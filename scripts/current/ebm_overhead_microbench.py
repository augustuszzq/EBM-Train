#!/usr/bin/env python3
"""Short overhead microbenchmarks for original CIFAR EBM long-K runs.

This script does not train a paper model. It isolates two costs:
  - ddp_train_only: model forward/backward/optimizer/DDP all-reduce, no sampling
  - sampling_only: K-step Langevin sampling, no optimizer/backward on parameters
  - attribution modes: single/local, independent multi-process, DDP full-step,
    and pipeline-stage synthetic variants.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch as t
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ebm_train_sync_mode_a import EnergyModel, langevin_sample


MODES = [
    "ddp_train_only",
    "sampling_only",
    "single_full_step",
    "single_sampling_only",
    "independent_full_step",
    "independent_sampling_only",
    "sampling_only_barrier",
    "ddp_full_step",
    "pipeline_stage_nocomm",
    "pipeline_stage_ring",
    "pipeline_full_step_ring",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("EBM overhead microbenchmark")
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64, help="global batch across ranks")
    ap.add_argument("--K", type=int, default=400)
    ap.add_argument("--pipe_stages", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--step_size", type=float, default=0.0025)
    ap.add_argument("--noise_std", type=float, default=0.01)
    ap.add_argument("--langevin_sign", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n_f", type=int, default=64)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--no_clamp_x", action="store_true")
    return ap.parse_args()


def mode_uses_dist(mode: str) -> bool:
    return mode in {
        "ddp_train_only",
        "sampling_only",
        "sampling_only_barrier",
        "ddp_full_step",
        "pipeline_stage_nocomm",
        "pipeline_stage_ring",
        "pipeline_full_step_ring",
    }


def setup_runtime(use_dist: bool) -> tuple[int, int, int, t.device]:
    if not use_dist:
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, t.cuda.device_count())))
        t.cuda.set_device(local_rank)
        return rank, world, local_rank, t.device("cuda", local_rank)
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    dist.init_process_group(backend="nccl")
    rank = int(dist.get_rank())
    world = int(dist.get_world_size())
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, t.cuda.device_count())))
    t.cuda.set_device(local_rank)
    return rank, world, local_rank, t.device("cuda", local_rank)


def make_writer(path: Path, fieldnames: list[str]) -> csv.DictWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer._file_handle = f  # type: ignore[attr-defined]
    return writer


def close_writer(writer: csv.DictWriter) -> None:
    f = getattr(writer, "_file_handle")
    f.close()


def sync_time(device: t.device) -> float:
    t.cuda.synchronize(device)
    return time.time()


def distributed_mean(value: float, device: t.device, use_dist: bool) -> float:
    if not use_dist:
        return float(value)
    buf = t.tensor([float(value)], dtype=t.float64, device=device)
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    return float((buf / dist.get_world_size()).item())


def run_ddp_train_only(
    args: argparse.Namespace,
    *,
    rank: int,
    world: int,
    local_batch: int,
    device: t.device,
    writer: csv.DictWriter,
) -> dict:
    model = EnergyModel(n_f=int(args.n_f)).to(device)
    ddp = DDP(model, device_ids=[device.index], output_device=device.index)
    opt = t.optim.Adam(ddp.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay), betas=(0.0, 0.999))
    pos = t.empty((local_batch, 3, 32, 32), device=device)
    neg = t.empty_like(pos)
    metric = {"iter_ms": 0.0, "train_ms": 0.0, "sample_ms": 0.0, "count": 0}
    wall0 = time.time()
    for step in range(int(args.steps)):
        pos.uniform_(-1.0, 1.0)
        neg.uniform_(-1.0, 1.0)
        start = sync_time(device)
        opt.zero_grad(set_to_none=True)
        f_pos = ddp(pos).mean()
        f_neg = ddp(neg).mean()
        objective = f_pos - f_neg
        loss = -objective if float(args.langevin_sign) > 0.0 else objective
        loss.backward()
        opt.step()
        end = sync_time(device)
        iter_ms = (end - start) * 1000.0
        if step >= int(args.warmup_steps):
            metric["iter_ms"] += iter_ms
            metric["train_ms"] += iter_ms
            metric["count"] += 1
        if step % max(1, int(args.log_every)) == 0 or step == int(args.steps) - 1:
            writer.writerow(
                {
                    "step": step,
                    "mode": args.mode,
                    "rank": rank,
                    "world_size": world,
                    "local_batch": local_batch,
                    "K": int(args.K),
                    "iter_ms": "%.6f" % iter_ms,
                    "train_ms": "%.6f" % iter_ms,
                    "sample_ms": "0.000000",
                    "samples_per_sec": "%.6f" % (float(args.batch_size) / max(1e-9, iter_ms / 1000.0)),
                    "f_pos": "%.8e" % float(f_pos.detach().item()),
                    "f_neg": "%.8e" % float(f_neg.detach().item()),
                    "loss": "%.8e" % float(loss.detach().item()),
                    "wallclock_sec": "%.6f" % (time.time() - wall0),
                }
            )
    return metric


def run_sampling_only(
    args: argparse.Namespace,
    *,
    rank: int,
    world: int,
    local_batch: int,
    device: t.device,
    writer: csv.DictWriter,
    barrier_each_step: bool = False,
) -> dict:
    model = EnergyModel(n_f=int(args.n_f)).to(device)
    model.eval()
    chain = t.empty((local_batch, 3, 32, 32), device=device).uniform_(-1.0, 1.0)
    metric = {"iter_ms": 0.0, "train_ms": 0.0, "sample_ms": 0.0, "count": 0}
    wall0 = time.time()
    for step in range(int(args.steps)):
        start = sync_time(device)
        out = langevin_sample(
            model=model,
            chain=chain,
            k_steps=int(args.K),
            langevin_sign=float(args.langevin_sign),
            step_size=float(args.step_size),
            noise_std=float(args.noise_std),
            clamp_x=(not bool(args.no_clamp_x)),
        )
        if barrier_each_step:
            dist.barrier()
        end = sync_time(device)
        iter_ms = (end - start) * 1000.0
        if step >= int(args.warmup_steps):
            metric["iter_ms"] += iter_ms
            metric["sample_ms"] += iter_ms
            metric["count"] += 1
        if step % max(1, int(args.log_every)) == 0 or step == int(args.steps) - 1:
            writer.writerow(
                {
                    "step": step,
                    "mode": args.mode,
                    "rank": rank,
                    "world_size": world,
                    "local_batch": local_batch,
                    "K": int(args.K),
                    "iter_ms": "%.6f" % iter_ms,
                    "train_ms": "0.000000",
                    "sample_ms": "%.6f" % iter_ms,
                    "comm_ms": "0.000000",
                    "samples_per_sec": "%.6f" % (float(args.batch_size) / max(1e-9, iter_ms / 1000.0)),
                    "f_pos": "nan",
                    "f_neg": "nan",
                    "loss": "nan",
                    "wallclock_sec": "%.6f" % (time.time() - wall0),
                    "max_abs_chain": "%.8e" % float(out.detach().abs().max().item()),
                }
            )
    return metric


def run_full_step(
    args: argparse.Namespace,
    *,
    rank: int,
    world: int,
    local_batch: int,
    device: t.device,
    writer: csv.DictWriter,
    use_ddp: bool,
) -> dict:
    model = EnergyModel(n_f=int(args.n_f)).to(device)
    train_model: nn.Module = model
    if use_ddp:
        train_model = DDP(model, device_ids=[device.index], output_device=device.index)
    opt = t.optim.Adam(train_model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay), betas=(0.0, 0.999))
    pos = t.empty((local_batch, 3, 32, 32), device=device)
    chain = t.empty_like(pos).uniform_(-1.0, 1.0)
    metric = {"iter_ms": 0.0, "train_ms": 0.0, "sample_ms": 0.0, "comm_ms": 0.0, "count": 0}
    wall0 = time.time()
    for step in range(int(args.steps)):
        pos.uniform_(-1.0, 1.0)
        iter_t0 = sync_time(device)
        sample_t0 = iter_t0
        chain_out = langevin_sample(
            model=model,
            chain=chain,
            k_steps=int(args.K),
            langevin_sign=float(args.langevin_sign),
            step_size=float(args.step_size),
            noise_std=float(args.noise_std),
            clamp_x=(not bool(args.no_clamp_x)),
        )
        sample_t1 = sync_time(device)
        opt.zero_grad(set_to_none=True)
        f_pos = train_model(pos).mean()
        f_neg = train_model(chain_out.detach()).mean()
        objective = f_pos - f_neg
        loss = -objective if float(args.langevin_sign) > 0.0 else objective
        loss.backward()
        opt.step()
        iter_t1 = sync_time(device)
        sample_ms = (sample_t1 - sample_t0) * 1000.0
        train_ms = (iter_t1 - sample_t1) * 1000.0
        iter_ms = (iter_t1 - iter_t0) * 1000.0
        if step >= int(args.warmup_steps):
            metric["iter_ms"] += iter_ms
            metric["sample_ms"] += sample_ms
            metric["train_ms"] += train_ms
            metric["count"] += 1
        if step % max(1, int(args.log_every)) == 0 or step == int(args.steps) - 1:
            writer.writerow(
                {
                    "step": step,
                    "mode": args.mode,
                    "rank": rank,
                    "world_size": world,
                    "local_batch": local_batch,
                    "K": int(args.K),
                    "pipe_stages": 1,
                    "slice_steps": int(args.K),
                    "iter_ms": "%.6f" % iter_ms,
                    "train_ms": "%.6f" % train_ms,
                    "sample_ms": "%.6f" % sample_ms,
                    "comm_ms": "0.000000",
                    "samples_per_sec": "%.6f" % (float(args.batch_size) / max(1e-9, iter_ms / 1000.0)),
                    "f_pos": "%.8e" % float(f_pos.detach().item()),
                    "f_neg": "%.8e" % float(f_neg.detach().item()),
                    "loss": "%.8e" % float(loss.detach().item()),
                    "wallclock_sec": "%.6f" % (time.time() - wall0),
                    "max_abs_chain": "%.8e" % float(chain_out.detach().abs().max().item()),
                }
            )
    return metric


def run_pipeline_stage(
    args: argparse.Namespace,
    *,
    rank: int,
    world: int,
    local_batch: int,
    device: t.device,
    writer: csv.DictWriter,
    do_ring: bool,
    do_full_step: bool,
) -> dict:
    pipe_stages = int(args.pipe_stages)
    if pipe_stages <= 0:
        raise ValueError("pipe_stages must be positive")
    if world != pipe_stages:
        raise ValueError("pipeline microbench expects world_size == pipe_stages")
    base = int(args.K) // pipe_stages
    rem = int(args.K) % pipe_stages
    slice_steps = base + (1 if rank < rem else 0)
    model = EnergyModel(n_f=int(args.n_f)).to(device)
    train_model: nn.Module = DDP(model, device_ids=[device.index], output_device=device.index) if do_full_step else model
    opt = t.optim.Adam(train_model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay), betas=(0.0, 0.999))
    pos = t.empty((local_batch, 3, 32, 32), device=device)
    chain = t.empty_like(pos).uniform_(-1.0, 1.0)
    recv = t.empty_like(chain)
    next_rank = (rank + 1) % world
    prev_rank = (rank - 1 + world) % world
    metric = {"iter_ms": 0.0, "train_ms": 0.0, "sample_ms": 0.0, "comm_ms": 0.0, "count": 0}
    wall0 = time.time()
    for step in range(int(args.steps)):
        pos.uniform_(-1.0, 1.0)
        iter_t0 = sync_time(device)
        sample_t0 = iter_t0
        chain_out = langevin_sample(
            model=model,
            chain=chain,
            k_steps=slice_steps,
            langevin_sign=float(args.langevin_sign),
            step_size=float(args.step_size),
            noise_std=float(args.noise_std),
            clamp_x=(not bool(args.no_clamp_x)),
        )
        sample_t1 = sync_time(device)
        comm_ms = 0.0
        if do_ring:
            comm_t0 = sample_t1
            tag = 910000 + (step % 100000)
            works = dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.irecv, recv, prev_rank, tag=tag),
                    dist.P2POp(dist.isend, chain_out.contiguous(), next_rank, tag=tag),
                ]
            )
            for work in works:
                work.wait()
            chain.copy_(recv)
            comm_t1 = sync_time(device)
            comm_ms = (comm_t1 - comm_t0) * 1000.0
        train_ms = 0.0
        f_pos = t.tensor(float("nan"), device=device)
        f_neg = t.tensor(float("nan"), device=device)
        loss = t.tensor(float("nan"), device=device)
        train_t0 = time.time()
        if do_full_step:
            opt.zero_grad(set_to_none=True)
            f_pos = train_model(pos).mean()
            f_neg = train_model(chain_out.detach()).mean()
            objective = f_pos - f_neg
            loss = -objective if float(args.langevin_sign) > 0.0 else objective
            loss.backward()
            opt.step()
            sync_time(device)
            train_ms = (time.time() - train_t0) * 1000.0
        iter_t1 = sync_time(device)
        sample_ms = (sample_t1 - sample_t0) * 1000.0
        iter_ms = (iter_t1 - iter_t0) * 1000.0
        if step >= int(args.warmup_steps):
            metric["iter_ms"] += iter_ms
            metric["sample_ms"] += sample_ms
            metric["comm_ms"] += comm_ms
            metric["train_ms"] += train_ms
            metric["count"] += 1
        if step % max(1, int(args.log_every)) == 0 or step == int(args.steps) - 1:
            writer.writerow(
                {
                    "step": step,
                    "mode": args.mode,
                    "rank": rank,
                    "world_size": world,
                    "local_batch": local_batch,
                    "K": int(args.K),
                    "pipe_stages": pipe_stages,
                    "slice_steps": slice_steps,
                    "iter_ms": "%.6f" % iter_ms,
                    "train_ms": "%.6f" % train_ms,
                    "sample_ms": "%.6f" % sample_ms,
                    "comm_ms": "%.6f" % comm_ms,
                    "samples_per_sec": "%.6f" % (float(args.batch_size) / max(1e-9, iter_ms / 1000.0)),
                    "f_pos": "%.8e" % float(f_pos.detach().item()),
                    "f_neg": "%.8e" % float(f_neg.detach().item()),
                    "loss": "%.8e" % float(loss.detach().item()),
                    "wallclock_sec": "%.6f" % (time.time() - wall0),
                    "max_abs_chain": "%.8e" % float(chain_out.detach().abs().max().item()),
                }
            )
    return metric


def main() -> int:
    args = parse_args()
    use_dist = mode_uses_dist(args.mode)
    rank, world, _local_rank, device = setup_runtime(use_dist=use_dist)
    if int(args.batch_size) % world != 0:
        raise ValueError("global batch_size must be divisible by world_size")
    local_batch = int(args.batch_size) // world
    t.manual_seed(int(args.seed) + rank * 1009)
    t.cuda.manual_seed_all(int(args.seed) + rank * 1009)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "step",
        "mode",
        "rank",
        "world_size",
        "local_batch",
        "K",
        "pipe_stages",
        "slice_steps",
        "iter_ms",
        "train_ms",
        "sample_ms",
        "comm_ms",
        "samples_per_sec",
        "f_pos",
        "f_neg",
        "loss",
        "wallclock_sec",
        "max_abs_chain",
    ]
    writer = make_writer(run_dir / ("metrics_rank%d.csv" % rank), fields)
    try:
        if args.mode == "ddp_train_only":
            metric = run_ddp_train_only(args, rank=rank, world=world, local_batch=local_batch, device=device, writer=writer)
        elif args.mode in {"sampling_only", "independent_sampling_only", "single_sampling_only"}:
            metric = run_sampling_only(
                args,
                rank=rank,
                world=world,
                local_batch=local_batch,
                device=device,
                writer=writer,
                barrier_each_step=False,
            )
        elif args.mode == "sampling_only_barrier":
            metric = run_sampling_only(
                args,
                rank=rank,
                world=world,
                local_batch=local_batch,
                device=device,
                writer=writer,
                barrier_each_step=True,
            )
        elif args.mode in {"single_full_step", "independent_full_step"}:
            metric = run_full_step(args, rank=rank, world=world, local_batch=local_batch, device=device, writer=writer, use_ddp=False)
        elif args.mode == "ddp_full_step":
            metric = run_full_step(args, rank=rank, world=world, local_batch=local_batch, device=device, writer=writer, use_ddp=True)
        elif args.mode == "pipeline_stage_nocomm":
            metric = run_pipeline_stage(args, rank=rank, world=world, local_batch=local_batch, device=device, writer=writer, do_ring=False, do_full_step=False)
        elif args.mode == "pipeline_stage_ring":
            metric = run_pipeline_stage(args, rank=rank, world=world, local_batch=local_batch, device=device, writer=writer, do_ring=True, do_full_step=False)
        elif args.mode == "pipeline_full_step_ring":
            metric = run_pipeline_stage(args, rank=rank, world=world, local_batch=local_batch, device=device, writer=writer, do_ring=True, do_full_step=True)
        else:
            raise RuntimeError("unsupported mode=%s" % args.mode)
    finally:
        close_writer(writer)

    count = max(1, int(metric["count"]))
    local_summary = {
        "mode": args.mode,
        "rank": rank,
        "world_size": world,
        "steps": int(args.steps),
        "warmup_steps": int(args.warmup_steps),
        "batch_size": int(args.batch_size),
        "local_batch": local_batch,
        "K": int(args.K),
        "mean_iter_ms": float(metric["iter_ms"]) / count,
        "mean_train_ms": float(metric["train_ms"]) / count,
        "mean_sample_ms": float(metric["sample_ms"]) / count,
        "mean_comm_ms": float(metric.get("comm_ms", 0.0)) / count,
    }
    local_summary["mean_samples_per_sec"] = float(args.batch_size) / max(1e-9, local_summary["mean_iter_ms"] / 1000.0)
    mean_iter_ms_global = distributed_mean(local_summary["mean_iter_ms"], device, use_dist=use_dist)
    if rank == 0:
        payload = dict(local_summary)
        payload["mean_iter_ms_across_ranks"] = mean_iter_ms_global
        payload["mean_samples_per_sec_across_ranks"] = float(args.batch_size) / max(1e-9, mean_iter_ms_global / 1000.0)
        payload["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with (run_dir / "summary.json").open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(json.dumps(payload, sort_keys=True), flush=True)
    if use_dist:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Single-GPU strict full-K control and serial pipeline emulation for Phase 1."""

import argparse
import csv
import os
import time
from datetime import datetime
from typing import List

import torch as t
import torch.nn as nn

try:
    from benchmark_runtime import (
        build_dataset,
        build_energy_model,
        energy_call,
        resolve_benchmark_runtime,
        unpack_batch,
    )
    from conditional_sampler import langevin_sample_conditional
    from conditional_chain import ChainState
    from conditional_replay import LabelReplayBuffer
except Exception:
    from polaris_ebm.scripts.current.benchmark_runtime import (
        build_dataset,
        build_energy_model,
        energy_call,
        resolve_benchmark_runtime,
        unpack_batch,
    )
    from polaris_ebm.scripts.current.conditional_sampler import langevin_sample_conditional
    from polaris_ebm.scripts.current.conditional_chain import ChainState
    from polaris_ebm.scripts.current.conditional_replay import LabelReplayBuffer

try:
    from ablation_common import compute_completion_aware_alpha, compute_k_slices, completion_ratios
except Exception:
    from polaris_ebm.scripts.current.ablation_common import (
        compute_completion_aware_alpha,
        compute_k_slices,
        completion_ratios,
    )

try:
    from ebm_train_sync_mode_a import (
        EnergyModel,
        build_cifar10,
        chain_state_from_buffers,
        iter_dataloader,
        langevin_sample,
        local_chain_summary,
        maybe_resume_from_checkpoint,
        state_to_cpu_payload,
        save_grid_png,
    )
except Exception:
    from polaris_ebm.scripts.current.ebm_train_sync_mode_a import (
        EnergyModel,
        build_cifar10,
        chain_state_from_buffers,
        iter_dataloader,
        langevin_sample,
        local_chain_summary,
        maybe_resume_from_checkpoint,
        state_to_cpu_payload,
        save_grid_png,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Single-GPU strict control and pipeline emulation", allow_abbrev=False)
    ap.add_argument("--train_mode", type=str, required=True, choices=["single_fullk", "single_pipe_emul"])
    ap.add_argument("--output_dir", type=str, default="./runs_ablation")
    ap.add_argument("--run_tag", type=str, default=os.environ.get("RUN_TAG", ""))
    ap.add_argument("--run_dir", type=str, default=os.environ.get("RUN_DIR", ""))
    ap.add_argument("--config", type=str, default=os.environ.get("CONFIG", ""))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data_dir", type=str, default="./data/cifar10")
    ap.add_argument("--n_f", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--save_dir", type=str, default="")
    ap.add_argument("--resume_ckpt", type=str, default=os.environ.get("RESUME_CKPT", ""))
    ap.add_argument("--vis_every", type=int, default=1000)
    ap.add_argument("--vis_num", type=int, default=64)
    ap.add_argument("--vis_nrow", type=int, default=8)
    ap.add_argument("--vis_dir", type=str, default="")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=0.0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--pipe_stages", type=int, default=4)
    ap.add_argument("--weight_mode", type=str, default="uniform", choices=["uniform", "deep_only", "last2_beta"])
    ap.add_argument("--last2_beta", type=float, default=0.01)
    ap.add_argument("--langevin_sign", type=float, default=1.0)
    ap.add_argument("--step_size", type=float, default=1.0)
    ap.add_argument("--noise_std", type=float, default=1e-2)
    ap.add_argument("--replay_capacity", type=int, default=0)
    ap.add_argument("--fresh_init", action="store_true")
    ap.add_argument("--no_clamp_x", action="store_true")
    ap.add_argument("--clamp_last_only", action="store_true")
    ap.add_argument("--pos_noise_std", type=float, default=3e-2)
    ap.add_argument("--no_clamp_pos", action="store_true")
    ap.add_argument("--debug_level", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--log_every", type=int, default=10)
    return ap.parse_args()


def should_log(step: int, debug_level: int, log_every: int) -> bool:
    if debug_level <= 0:
        return False
    if debug_level >= 2:
        return True
    return (step % max(1, int(log_every))) == 0


def resolve_run_dir(args: argparse.Namespace) -> str:
    if args.run_dir.strip():
        return os.path.abspath(args.run_dir)
    run_tag = args.run_tag.strip() or "%s_%s" % (
        args.train_mode,
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    return os.path.join(args.output_dir, run_tag)


def maybe_save_checkpoint(
    step: int,
    args: argparse.Namespace,
    run_dir: str,
    model: nn.Module,
    optimizer: t.optim.Optimizer,
    extra_state=None,
) -> None:
    if int(args.save_every) <= 0:
        return
    do_save = ((step + 1) % int(args.save_every)) == 0 or (step == int(args.steps) - 1)
    if not do_save:
        return
    ckpt_dir = args.save_dir.strip() or os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, "ckpt_step%d.pt" % step)
    t.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": int(step),
            "args": vars(args),
            "extra_state": extra_state,
        },
        path,
    )
    print("[CKPT] saved %s" % path, flush=True)


def build_stage_buffers(pipe_stages: int, batch_size: int, device: t.device) -> List[t.Tensor]:
    bufs = []
    for _ in range(int(pipe_stages)):
        bufs.append(t.empty((batch_size, 3, 32, 32), dtype=t.float32, device=device).uniform_(-1.0, 1.0))
    return bufs


def main() -> None:
    args = parse_args()
    if not t.cuda.is_available():
        raise RuntimeError("CUDA is required for single_pipe_emul")
    device = t.device("cuda")
    t.cuda.set_device(0)
    t.manual_seed(args.seed)
    t.cuda.manual_seed_all(args.seed)
    t.backends.cudnn.benchmark = True

    pipe_stages = 1 if args.train_mode == "single_fullk" else int(args.pipe_stages)
    if pipe_stages <= 0:
        raise RuntimeError(f"pipe_stages must be > 0, got {pipe_stages}")

    runtime = resolve_benchmark_runtime(config_path=str(args.config), data_dir=args.data_dir)

    run_dir = resolve_run_dir(args)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(args.save_dir.strip() or os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(args.vis_dir.strip() or os.path.join(run_dir, "vis"), exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics_rank0.csv")

    ds = build_dataset(runtime=runtime, cifar_builder=build_cifar10, train=True)
    dl = t.utils.data.DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        drop_last=True,
    )
    data_iter = iter_dataloader(dl)

    model = build_energy_model(runtime=runtime, n_f=args.n_f, unconditional_cls=EnergyModel).to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    resume_extra_state = {}
    start_step = maybe_resume_from_checkpoint(
        args=args,
        model=model,
        optimizer=optimizer,
        device=device,
        rank=0,
        extra_state_out=resume_extra_state,
    )

    k_slices = compute_k_slices(k=int(args.K), pipe_stages=pipe_stages)
    completion = completion_ratios(k_slices)
    stage_buffers = [
        t.empty((int(args.batch_size),) + runtime.image_shape, dtype=t.float32, device=device).uniform_(-1.0, 1.0)
        for _ in range(pipe_stages)
    ]
    stage_labels = [t.zeros((int(args.batch_size),), dtype=t.long, device=device) for _ in range(pipe_stages)] if runtime.conditional else None
    stage_chain_ids = [t.full((int(args.batch_size),), -1, dtype=t.long, device=device) for _ in range(pipe_stages)] if runtime.conditional else None
    stage_steps_done = [t.zeros((int(args.batch_size),), dtype=t.long, device=device) for _ in range(pipe_stages)] if runtime.conditional else None
    stage_valid = [t.ones((int(args.batch_size),), dtype=t.bool, device=device) for _ in range(pipe_stages)] if runtime.conditional else None
    labels_bootstrapped = not runtime.conditional
    replay = None
    if runtime.conditional:
        replay_capacity = int(args.replay_capacity) if int(args.replay_capacity) > 0 else max(int(args.batch_size) * 64, 256)
        replay = LabelReplayBuffer(
            capacity=replay_capacity,
            image_shape=runtime.image_shape,
            num_classes=runtime.num_classes,
        )
        replay_state = resume_extra_state.get("replay_state")
        if replay_state:
            replay.load_state_dict(replay_state)
        stage_states = resume_extra_state.get("stage_states") or []
        for s, payload in enumerate(stage_states[:pipe_stages]):
            state = ChainState.from_payload(payload).to(device)
            stage_buffers[s].copy_(state.x)
            stage_labels[s].copy_(state.y)
            if state.chain_id is not None:
                stage_chain_ids[s].copy_(state.chain_id)
            if state.steps_done is not None:
                stage_steps_done[s].copy_(state.steps_done)
            if state.valid is not None:
                stage_valid[s].copy_(state.valid)
        if stage_states:
            labels_bootstrapped = True

    header = [
        "step",
        "sample_ms",
        "train_ms",
        "iter_ms",
        "f_pos",
        "objective",
        "loss",
        "max_f_neg",
        "max_abs_chain",
        "chain_unique_count",
    ]
    header.extend("c_stage%d" % s for s in range(pipe_stages))
    header.extend("completion_stage%d" % s for s in range(pipe_stages))
    header.extend("alpha_stage%d" % s for s in range(pipe_stages))
    header.extend("f_neg_stage%d" % s for s in range(pipe_stages))

    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(header)

    print(
        "[Run] mode=%s out=%s batch=%d steps=%d K=%d pipe_stages=%d k_slices=[%s] completion=[%s] weight_mode=%s last2_beta=%.6f"
        % (
            args.train_mode,
            run_dir,
            int(args.batch_size),
            int(args.steps),
            int(args.K),
            pipe_stages,
            ",".join(str(int(x)) for x in k_slices),
            ",".join("%.4f" % x for x in completion),
            str(args.weight_mode),
            float(args.last2_beta),
        ),
        flush=True,
    )

    train_start = time.time()
    for step in range(start_step, int(args.steps)):
        iter_t0 = time.time()
        batch = next(data_iter)
        pos, pos_labels = unpack_batch(batch, device=device, conditional=runtime.conditional)
        if runtime.conditional and not labels_bootstrapped:
            for s in range(pipe_stages):
                stage_labels[s].copy_(pos_labels)
                stage_chain_ids[s].fill_(-1)
                stage_steps_done[s].zero_()
                stage_valid[s].fill_(True)
            if replay is not None:
                replay_state, _ = replay.sample(pos_labels.detach().cpu())
                replay_state = replay_state.to(device)
                stage_buffers[0].copy_(replay_state.x)
                stage_labels[0].copy_(replay_state.y)
                stage_chain_ids[0].copy_(replay_state.chain_id)
                stage_steps_done[0].copy_(replay_state.steps_done)
                stage_valid[0].copy_(replay_state.valid)
            labels_bootstrapped = True
        if args.fresh_init:
            if runtime.conditional and replay is not None:
                replay_state, _ = replay.sample(pos_labels.detach().cpu())
                replay_state = replay_state.to(device)
                stage_buffers[0].copy_(replay_state.x)
                stage_labels[0].copy_(replay_state.y)
                stage_chain_ids[0].copy_(replay_state.chain_id)
                stage_steps_done[0].copy_(replay_state.steps_done)
                stage_valid[0].copy_(replay_state.valid)
            else:
                stage_buffers[0].uniform_(-1.0, 1.0)
                if runtime.conditional:
                    stage_labels[0].copy_(pos_labels)
                    stage_chain_ids[0].fill_(-1)
                    stage_steps_done[0].zero_()
                    stage_valid[0].fill_(True)

        stage_outs: List[t.Tensor] = []
        sample_t0 = time.time()
        for s in range(pipe_stages):
            if runtime.conditional:
                out = langevin_sample_conditional(
                    model=model,
                    chain=stage_buffers[s],
                    labels=stage_labels[s],
                    k_steps=int(k_slices[s]),
                    langevin_sign=float(args.langevin_sign),
                    step_size=float(args.step_size),
                    noise_std=float(args.noise_std),
                    clamp_x=(not args.no_clamp_x),
                    clamp_last_only=bool(args.clamp_last_only),
                )
            else:
                out = langevin_sample(
                    model=model,
                    chain=stage_buffers[s],
                    k_steps=int(k_slices[s]),
                    langevin_sign=float(args.langevin_sign),
                    step_size=float(args.step_size),
                    noise_std=float(args.noise_std),
                    clamp_x=(not args.no_clamp_x),
                    clamp_last_only=bool(args.clamp_last_only),
                )
            stage_outs.append(out.detach().clone())
            if runtime.conditional:
                stage_steps_done[s].add_(int(k_slices[s]))
                stage_valid[s].fill_(True)
        sample_ms = (time.time() - sample_t0) * 1000.0

        train_t0 = time.time()
        if float(args.pos_noise_std) > 0.0:
            pos = pos + float(args.pos_noise_std) * t.randn_like(pos)
        if not args.no_clamp_pos:
            pos = pos.clamp(-1.0, 1.0)

        optimizer.zero_grad(set_to_none=True)
        f_pos = energy_call(model, pos, pos_labels).mean()
        f_neg_vals = [
            energy_call(model, x.detach(), stage_labels[s] if runtime.conditional else None).mean()
            for s, x in enumerate(stage_outs)
        ]
        if args.train_mode == "single_fullk":
            alpha_vec = [1.0]
        else:
            alpha_vec = compute_completion_aware_alpha(
                step=step,
                pipe_stages=pipe_stages,
                weight_mode=str(args.weight_mode),
                last2_beta=float(args.last2_beta),
            )
        # Completion-aware weighting acts only on the negative scalar energy terms.
        # Stage states are never averaged in image space and are only coupled through
        # chain buffers and updated parameters.
        weighted_f_neg = sum(float(alpha_vec[s]) * f_neg_vals[s] for s in range(pipe_stages))
        c_stage_vec = [float(pipe_stages) * float(alpha) for alpha in alpha_vec]
        objective = f_pos - weighted_f_neg
        loss_sign = -1.0 if float(args.langevin_sign) > 0.0 else 1.0
        loss = objective * loss_sign
        loss.backward()
        if float(args.max_grad_norm) > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
        optimizer.step()
        train_ms = (time.time() - train_t0) * 1000.0

        if runtime.conditional and replay is not None and bool((stage_chain_ids[0] >= 0).all().item()):
            replay.update_slots(
                ChainState(
                    x=stage_outs[0].detach(),
                    y=stage_labels[0].detach(),
                    chain_id=stage_chain_ids[0].detach(),
                    steps_done=stage_steps_done[0].detach(),
                    stage_id=0,
                    valid=stage_valid[0].detach(),
                ).to("cpu")
            )

        next_buffers = [x.clone() for x in stage_outs]
        next_labels = [y.clone() for y in stage_labels] if runtime.conditional else None
        next_chain_ids = [z.clone() for z in stage_chain_ids] if runtime.conditional else None
        next_steps_done = [z.clone() for z in stage_steps_done] if runtime.conditional else None
        next_valid = [z.clone() for z in stage_valid] if runtime.conditional else None
        for s in range(pipe_stages - 1, 0, -1):
            stage_buffers[s] = next_buffers[s - 1]
            if runtime.conditional:
                stage_labels[s] = next_labels[s - 1]
                stage_chain_ids[s] = next_chain_ids[s - 1]
                stage_steps_done[s] = next_steps_done[s - 1]
                stage_valid[s] = next_valid[s - 1]
        stage_buffers[0] = next_buffers[-1]
        if runtime.conditional:
            stage_labels[0] = next_labels[-1]
            stage_chain_ids[0] = next_chain_ids[-1]
            stage_steps_done[0] = next_steps_done[-1]
            stage_valid[0] = next_valid[-1]

        deepest = stage_outs[-1]
        max_f_neg = max(float(x.detach().item()) for x in f_neg_vals)
        max_abs_chain = max(float(x.detach().abs().max().item()) for x in stage_outs)
        chain_unique_count = max(int(local_chain_summary(x)[6]) for x in stage_outs)
        iter_ms = (time.time() - iter_t0) * 1000.0

        if args.vis_every > 0 and (step % int(args.vis_every) == 0):
            vis_dir = args.vis_dir.strip() or os.path.join(run_dir, "vis")
            os.makedirs(vis_dir, exist_ok=True)
            save_grid_png(
                deepest[: max(1, int(args.vis_num))],
                os.path.join(vis_dir, "step%06d_neg_stage%d.png" % (step, pipe_stages - 1)),
                nrow=max(1, int(args.vis_nrow)),
            )
            save_grid_png(
                pos[: max(1, int(args.vis_num))],
                os.path.join(vis_dir, "step%06d_pos.png" % step),
                nrow=max(1, int(args.vis_nrow)),
            )

        if should_log(step, args.debug_level, args.log_every):
            alpha_msg = ",".join("%.4f" % float(a) for a in alpha_vec)
            c_stage_msg = ",".join("%.4f" % float(x) for x in c_stage_vec)
            completion_msg = ",".join("%.4f" % float(x) for x in completion)
            fneg_msg = ",".join("%.4e" % float(v.detach().item()) for v in f_neg_vals)
            print(
                "[STEP] step=%d completion=[%s] alpha=[%s] c_stage=[%s] f_pos=%.4e f_neg=[%s] objective=%.4e loss=%.4e max_f_neg=%.4e max_abs_chain=%.4e unique_count=%d sample=%.2fms train=%.2fms"
                % (
                    step,
                    completion_msg,
                    alpha_msg,
                    c_stage_msg,
                    float(f_pos.detach().item()),
                    fneg_msg,
                    float(objective.detach().item()),
                    float(loss.detach().item()),
                    max_f_neg,
                    max_abs_chain,
                    chain_unique_count,
                    sample_ms,
                    train_ms,
                ),
                flush=True,
            )

        row = [
            step,
            "%.3f" % sample_ms,
            "%.3f" % train_ms,
            "%.3f" % iter_ms,
            "%.6e" % float(f_pos.detach().item()),
            "%.6e" % float(objective.detach().item()),
            "%.6e" % float(loss.detach().item()),
            "%.6e" % max_f_neg,
            "%.6e" % max_abs_chain,
            chain_unique_count,
        ]
        row.extend("%.6f" % float(x) for x in c_stage_vec)
        row.extend("%.6f" % float(x) for x in completion)
        row.extend("%.6f" % float(a) for a in alpha_vec)
        row.extend("%.6e" % float(v.detach().item()) for v in f_neg_vals)
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        maybe_save_checkpoint(
            step=step,
            args=args,
            run_dir=run_dir,
            model=model,
            optimizer=optimizer,
            extra_state=(
                {
                    "stage_states": [
                        state_to_cpu_payload(
                            chain_state_from_buffers(
                                x=stage_buffers[s],
                                y=stage_labels[s],
                                chain_id=stage_chain_ids[s],
                                steps_done=stage_steps_done[s],
                                valid=stage_valid[s],
                                stage_id=s,
                            )
                        )
                        for s in range(pipe_stages)
                    ],
                    "replay_state": (replay.state_dict() if replay is not None else None),
                }
                if runtime.conditional
                else None
            ),
        )

    print("[TRAIN] done in %.1fs" % (time.time() - train_start), flush=True)


if __name__ == "__main__":
    main()

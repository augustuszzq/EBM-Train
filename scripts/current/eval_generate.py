#!/usr/bin/env python3
"""Generate samples from a saved EBM checkpoint with unified Langevin settings."""
import argparse
import json
import math
import os
from typing import Dict, Tuple

import torch as t

try:
    from torchvision.utils import save_image
except Exception:
    save_image = None

try:
    from ebm_train_sync_mode_a import EnergyModel, langevin_sample
except Exception:
    from polaris_ebm.scripts.current.ebm_train_sync_mode_a import EnergyModel, langevin_sample

try:
    from benchmark_runtime import build_energy_model, make_eval_label_schedule, resolve_benchmark_runtime
    from conditional_sampler import langevin_sample_conditional
except Exception:
    from polaris_ebm.scripts.current.benchmark_runtime import (
        build_energy_model,
        make_eval_label_schedule,
        resolve_benchmark_runtime,
    )
    from polaris_ebm.scripts.current.conditional_sampler import langevin_sample_conditional


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Generate EBM samples from checkpoint")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--config", type=str, default="")
    ap.add_argument("--num_images", type=int, default=5000)
    ap.add_argument("--images_per_class", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--K_eval", type=int, default=200)
    ap.add_argument(
        "--langevin_sign",
        type=float,
        default=-1.0,
        help="Langevin drift sign: x += sign*step_size*grad + noise; use -1 for energy descent",
    )
    ap.add_argument("--step_size", type=float, default=0.2)
    ap.add_argument("--noise_std", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n_f", type=int, default=-1, help="override n_f from checkpoint args")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--no_clamp_x", action="store_true")
    ap.add_argument("--clamp_last_only", action="store_true")
    return ap.parse_args()


def select_device(mode: str) -> t.device:
    if mode == "cpu":
        return t.device("cpu")
    if mode == "cuda":
        if not t.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return t.device("cuda")
    return t.device("cuda" if t.cuda.is_available() else "cpu")


def load_checkpoint(path: str) -> Tuple[Dict, Dict]:
    obj = t.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        state = obj["model_state_dict"]
        meta = obj
    else:
        state = obj
        meta = {}
    return state, meta


def infer_n_f(args: argparse.Namespace, meta: Dict) -> int:
    if int(args.n_f) > 0:
        return int(args.n_f)
    a = meta.get("args", {})
    if isinstance(a, dict) and "n_f" in a:
        return int(a["n_f"])
    return 64


def to_01(x: t.Tensor) -> t.Tensor:
    return ((x.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)


def summarize_tensor(x: t.Tensor, unique_limit: int = 1_000_000, eps: float = 1e-6) -> Dict:
    y = x.detach().float().cpu()
    n = int(y.numel())
    if n <= 0:
        raise RuntimeError("cannot summarize empty tensor")
    if n <= unique_limit:
        unique_count = int(t.unique(y).numel())
        unique_mode = "exact"
    else:
        stride = max(1, n // unique_limit)
        ys = y.view(-1)[::stride]
        unique_count = int(t.unique(ys).numel())
        unique_mode = "sampled"
    return {
        "shape": list(y.shape),
        "dtype": str(x.dtype),
        "numel": n,
        "min": float(y.min().item()),
        "max": float(y.max().item()),
        "mean": float(y.mean().item()),
        "std": float(y.std(unbiased=False).item()),
        "frac_zero": float((y <= eps).float().mean().item()),
        "frac_one": float((y >= (1.0 - eps)).float().mean().item()),
        "unique_count": unique_count,
        "unique_mode": unique_mode,
    }


def main() -> None:
    args = parse_args()
    if args.num_images <= 0:
        raise RuntimeError(f"num_images must be > 0, got {args.num_images}")
    if args.batch <= 0:
        raise RuntimeError(f"batch must be > 0, got {args.batch}")
    if args.K_eval <= 0:
        raise RuntimeError(f"K_eval must be > 0, got {args.K_eval}")

    os.makedirs(args.out_dir, exist_ok=True)

    t.manual_seed(args.seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(args.seed)

    device = select_device(args.device)

    state, meta = load_checkpoint(args.ckpt)
    n_f = infer_n_f(args, meta)

    meta_args = meta.get("args", {}) if isinstance(meta.get("args", {}), dict) else {}
    runtime = resolve_benchmark_runtime(
        config_path=(args.config or meta_args.get("config", "")),
        data_dir=meta_args.get("data_dir", "./data/cifar10"),
    )
    model = build_energy_model(runtime=runtime, n_f=n_f, unconditional_cls=EnergyModel).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint/model mismatch: missing=%s unexpected=%s" % (str(missing), str(unexpected))
        )
    model.eval()

    all_chunks = []
    generated = 0
    labels_all = None
    if runtime.conditional:
        ipc = int(args.images_per_class) if int(args.images_per_class) > 0 else max(1, int(args.num_images) // runtime.num_classes)
        labels_all = make_eval_label_schedule(runtime.num_classes, ipc)[: args.num_images]

    with t.no_grad():
        while generated < args.num_images:
            bs = min(args.batch, args.num_images - generated)
            chain = t.empty((bs,) + runtime.image_shape, dtype=t.float32, device=device)
            chain.uniform_(-1.0, 1.0)
            if runtime.conditional:
                labels = labels_all[generated : generated + bs].to(device)
                out = langevin_sample_conditional(
                    model=model,
                    chain=chain,
                    labels=labels,
                    k_steps=args.K_eval,
                    langevin_sign=float(args.langevin_sign),
                    step_size=args.step_size,
                    noise_std=args.noise_std,
                    clamp_x=(not args.no_clamp_x),
                    clamp_last_only=bool(args.clamp_last_only),
                )
            else:
                out = langevin_sample(
                    model=model,
                    chain=chain,
                    k_steps=args.K_eval,
                    langevin_sign=float(args.langevin_sign),
                    step_size=args.step_size,
                    noise_std=args.noise_std,
                    clamp_x=(not args.no_clamp_x),
                    clamp_last_only=bool(args.clamp_last_only),
                )
            all_chunks.append(out.detach().cpu())
            generated += bs

    samples_m11 = t.cat(all_chunks, dim=0)
    samples_01 = to_01(samples_m11)
    m11_summary = summarize_tensor(samples_m11)
    x01_summary = summarize_tensor(samples_01)

    grid_count = min(256, samples_01.size(0))
    grid_nrow = max(1, int(math.sqrt(grid_count)))
    grid_path = os.path.join(args.out_dir, "grid.png")
    if save_image is None:
        raise RuntimeError("torchvision is required for grid.png output")
    save_image(samples_01[:grid_count], grid_path, nrow=grid_nrow)

    samples_path = os.path.join(args.out_dir, "samples.pt")
    payload = {
        "samples_01": samples_01,
        "samples_m11": samples_m11,
        "meta": {
                "ckpt": args.ckpt,
                "num_images": int(args.num_images),
                "batch": int(args.batch),
                "K_eval": int(args.K_eval),
                "langevin_sign": float(args.langevin_sign),
                "step_size": float(args.step_size),
                "noise_std": float(args.noise_std),
                "seed": int(args.seed),
                "n_f": int(n_f),
                "device": str(device),
                "no_clamp_x": bool(args.no_clamp_x),
                "clamp_last_only": bool(args.clamp_last_only),
            },
    }
    if labels_all is not None:
        payload["labels"] = labels_all.long().cpu()
    t.save(payload, samples_path)

    stats = {
        "ckpt": args.ckpt,
        "num_images": int(samples_01.size(0)),
        "K_eval": int(args.K_eval),
        "langevin_sign": float(args.langevin_sign),
        "step_size": float(args.step_size),
        "noise_std": float(args.noise_std),
        "no_clamp_x": bool(args.no_clamp_x),
        "clamp_last_only": bool(args.clamp_last_only),
        "seed": int(args.seed),
        "n_f": int(n_f),
        "device": str(device),
        "sample_m11_mean": float(samples_m11.mean().item()),
        "sample_m11_std": float(samples_m11.std(unbiased=False).item()),
        "sample_m11_min": float(samples_m11.min().item()),
        "sample_m11_max": float(samples_m11.max().item()),
        "sample_01_mean": float(samples_01.mean().item()),
        "sample_01_std": float(samples_01.std(unbiased=False).item()),
        "sample_01_min": float(samples_01.min().item()),
        "sample_01_max": float(samples_01.max().item()),
        "sample_m11_summary": m11_summary,
        "sample_01_summary": x01_summary,
        "grid_path": grid_path,
        "samples_path": samples_path,
    }
    stats_path = os.path.join(args.out_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print("[EVAL] wrote %s" % grid_path, flush=True)
    print("[EVAL] wrote %s" % samples_path, flush=True)
    print("[EVAL] wrote %s" % stats_path, flush=True)
    print("[EVAL] sample_m11_summary=%s" % json.dumps(m11_summary, sort_keys=True), flush=True)
    print("[EVAL] sample_01_summary=%s" % json.dumps(x01_summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

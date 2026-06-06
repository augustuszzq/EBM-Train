#!/usr/bin/env python3
"""Shared driver for ImageNet-32 benchmark wrappers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from benchmark_config import load_benchmark_config
    from imagenet32_data import get_benchmark_data_spec, materialize_imagenet32
except Exception:
    from polaris_ebm.scripts.current.benchmark_config import load_benchmark_config
    from polaris_ebm.scripts.current.imagenet32_data import get_benchmark_data_spec, materialize_imagenet32


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("ImageNet-32 benchmark driver")
    ap.add_argument(
        "mode",
        choices=[
            "single",
            "ddp",
            "pipeline",
            "ablation_screen",
            "mainline",
            "scaling",
            "eval_fid",
            "eval_conditional_acc",
        ],
    )
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--materialize-imagenet32", action="store_true")
    ap.add_argument("--cache-dir", type=str, default="")
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--materialize-max-examples", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_benchmark_config(args.config)
    spec = get_benchmark_data_spec(cfg.benchmark.name)
    payload = {
        "mode": args.mode,
        "config": str(Path(args.config).resolve()),
        "benchmark": {
            "name": spec.name,
            "conditional": spec.conditional,
            "num_classes": spec.num_classes,
            "image_size": spec.image_size,
        },
        "train": {
            "steps": cfg.train.steps,
            "batch_size": cfg.train.batch_size,
            "K": cfg.train.K,
            "lr": cfg.train.lr,
        },
        "pipeline": {
            "pipe_stages": cfg.pipeline.pipe_stages,
            "weight_mode": cfg.pipeline.weight_mode,
            "last2_beta": cfg.pipeline.last2_beta,
        },
    }
    if args.materialize_imagenet32 and spec.name == "imagenet32":
        payload["materialize"] = materialize_imagenet32(
            out_root=cfg.benchmark.data_root,
            split=cfg.benchmark.split_train,
            cache_dir=args.cache_dir,
            streaming=bool(args.streaming),
            max_examples=int(args.materialize_max_examples),
        )
    payload["dry_run"] = bool(args.dry_run)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

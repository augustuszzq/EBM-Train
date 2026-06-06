#!/usr/bin/env python3
"""Manifest builder for ImageNet-32 DRL-backbone external validation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
DATA_DIR = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/data/imagenet32")
FIELDS = [
    "phase",
    "group",
    "exp_id",
    "benchmark",
    "model_family",
    "story",
    "train_mode",
    "world_size",
    "pipe_stages",
    "K",
    "steps",
    "lr",
    "step_size",
    "noise_std",
    "weight_mode",
    "last2_beta",
    "seed",
    "num_nodes",
    "ppn",
    "batch_size",
    "save_every",
    "vis_every",
    "eval_images",
    "queue",
    "train_walltime",
    "data_dir",
    "config",
    "clamp_x",
    "skip_optimizer_until_full_diagonal",
    "grad_clip_norm",
    "submit",
    "notes",
]


def imagenet32_materialized_success(root: str | Path) -> bool:
    root = Path(root)
    root_success = root / "_SUCCESS.json"
    split_success = [root / "train" / "_SUCCESS.json", root / "val" / "_SUCCESS.json"]
    return root_success.exists() or all(path.exists() for path in split_success)


def _row(
    *,
    phase: str,
    exp_id: str,
    train_mode: str,
    steps: int,
    seed: int,
    config: str,
    submit: str,
    world_size: int = 4,
    pipe_stages: int = 1,
    weight_mode: str = "deep_only",
    num_nodes: int = 1,
    ppn: int = 4,
    train_walltime: str = "08:00:00",
    skip_optimizer_until_full_diagonal: bool = False,
    notes: str = "",
) -> dict[str, str]:
    return {
        "phase": phase,
        "group": "imagenet32_drl",
        "exp_id": exp_id,
        "benchmark": "imagenet32",
        "model_family": "drl_resnet_energy",
        "story": "DRL-style ResNet backbone with existing completion-aware EBM objective",
        "train_mode": train_mode,
        "world_size": str(int(world_size)),
        "pipe_stages": str(int(pipe_stages)),
        "K": "100",
        "steps": str(int(steps)),
        "lr": "5e-5",
        "step_size": "0.005",
        "noise_std": "0.005",
        "weight_mode": weight_mode,
        "last2_beta": "0.01",
        "seed": str(int(seed)),
        "num_nodes": str(int(num_nodes)),
        "ppn": str(int(ppn)),
        "batch_size": "128",
        "save_every": "5000",
        "vis_every": "5000",
        "eval_images": "50000",
        "queue": "preemptable",
        "train_walltime": train_walltime,
        "data_dir": str(DATA_DIR),
        "config": str(PROJECT_DIR / config),
        "clamp_x": "true",
        "skip_optimizer_until_full_diagonal": "true" if skip_optimizer_until_full_diagonal else "false",
        "grad_clip_norm": "",
        "submit": submit,
        "notes": notes,
    }


def build_drl_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    rows.append(
        _row(
            phase="phase0_smoke",
            exp_id="imagenet32_drl_ddp_k100_s5k_seed1",
            train_mode="ddp_fullk",
            steps=5000,
            seed=1,
            config="configs/imagenet32_drl_ddp_smoke.yaml",
            submit="no",
            train_walltime="04:00:00",
            notes="Phase 0 smoke: DDP full-K baseline, labels loaded but ignored by DRL backbone.",
        )
    )
    rows.append(
        _row(
            phase="phase0_smoke",
            exp_id="imagenet32_drl_pipe_p4_equal_k100_s5k_seed1",
            train_mode="pipe_strict",
            steps=5000,
            seed=1,
            config="configs/imagenet32_drl_pipeline_p4_equal_smoke.yaml",
            submit="no",
            pipe_stages=4,
            weight_mode="uniform",
            skip_optimizer_until_full_diagonal=True,
            train_walltime="04:00:00",
            notes="Phase 0 smoke: strict pipeline P4 equal with DRL backbone.",
        )
    )

    for train_mode, pipe_stages, weight_mode, config, suffix in [
        ("ddp_fullk", 1, "deep_only", "configs/imagenet32_drl_ddp_k100.yaml", "ddp"),
        ("pipe_strict", 4, "uniform", "configs/imagenet32_drl_pipeline_p4_equal_k100.yaml", "pipe_p4_equal"),
    ]:
        rows.append(
            _row(
                phase="phase1_early",
                exp_id=f"imagenet32_drl_{suffix}_k100_s150k_seed1",
                train_mode=train_mode,
                steps=150000,
                seed=1,
                config=config,
                submit="no",
                    pipe_stages=pipe_stages,
                    weight_mode=weight_mode,
                    skip_optimizer_until_full_diagonal=(train_mode == "pipe_strict"),
                    train_walltime="36:00:00",
                )
        )
        for seed in (1, 2, 3):
            rows.append(
                _row(
                    phase="phase2_300k",
                    exp_id=f"imagenet32_drl_{suffix}_k100_s300k_seed{seed}",
                    train_mode=train_mode,
                    steps=300000,
                    seed=seed,
                    config=config,
                    submit="no",
                    pipe_stages=pipe_stages,
                    weight_mode=weight_mode,
                    skip_optimizer_until_full_diagonal=(train_mode == "pipe_strict"),
                    train_walltime="72:00:00",
                )
            )
        for seed in (1, 2, 3):
            rows.append(
                _row(
                    phase="phase3_500k",
                    exp_id=f"imagenet32_drl_{suffix}_k100_s500k_seed{seed}",
                    train_mode=train_mode,
                    steps=500000,
                    seed=seed,
                    config=config,
                    submit="no",
                    pipe_stages=pipe_stages,
                    weight_mode=weight_mode,
                    skip_optimizer_until_full_diagonal=(train_mode == "pipe_strict"),
                    train_walltime="96:00:00",
                )
            )

    return rows


def write_manifest(path: str | Path, rows: list[dict[str, str]] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = rows if rows is not None else build_drl_rows()
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in payload:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser("Write ImageNet-32 DRL-backbone manifest")
    parser.add_argument(
        "--output",
        default=str(PROJECT_DIR / "experiments" / "imagenet32_drl_manifest.csv"),
    )
    args = parser.parse_args()
    rows = build_drl_rows()
    write_manifest(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the real ImageNet-32 Phase A preemptable manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
MANIFEST_PATH = PROJECT_DIR / "experiments" / "imagenet32_phasea_manifest.csv"
DATA_DIR = "/eagle/lc-mpi/Zhiqing/polaris_ebm/data/imagenet32"

HORIZONS = (
    {"steps": 1000, "tag": "s1k", "train_walltime_single": "02:00:00", "train_walltime_multi": "02:00:00"},
    {"steps": 5000, "tag": "s5k", "train_walltime_single": "04:00:00", "train_walltime_multi": "04:00:00"},
    {"steps": 20000, "tag": "s20k", "train_walltime_single": "10:00:00", "train_walltime_multi": "08:00:00"},
)


def base_row() -> Dict[str, str]:
    return {
        "submit": "1",
        "group": "PhaseA",
        "benchmark": "imagenet32",
        "story": "ImageNet-32 conditional infrastructure validation",
        "hypothesis": "Single, DDP, and strict pipeline should remain numerically stable under conditional state",
        "queue": "preemptable",
        "eval_images": "50000",
        "notes": "phase_a_smoke",
        "eval_walltime": "04:00:00",
        "save_every": "1000",
        "vis_every": "1000",
        "batch_size": "64",
        "sigma_pd": "0.03",
        "noise_std": "0.01",
        "langevin_sign": "1.0",
        "weight_decay": "0.0",
        "max_grad_norm": "0.0",
        "data_dir": DATA_DIR,
        "sync_fresh_init": "1",
        "debug_level": "1",
        "log_every": "50",
        "eval_batch": "256",
        "k_eval": "100",
        "step_size": "1.0",
        "lr": "1e-4",
        "last2_beta": "0.01",
        "seed": "1",
        "K": "100",
    }


def build_phasea_rows() -> List[Dict[str, str]]:
    base = base_row()
    rows: List[Dict[str, str]] = []
    for horizon in HORIZONS:
        steps = str(horizon["steps"])
        tag = str(horizon["tag"])

        rows.append(
            {
                **base,
                "exp_id": f"imagenet32_single_strict_k100_{tag}",
                "train_mode": "single_fullk",
                "config": str(PROJECT_DIR / "configs" / "imagenet32_single_strict.yaml"),
                "world_size": "1",
                "pipe_stages": "1",
                "weight_mode": "deep_only",
                "steps": steps,
                "num_nodes": "1",
                "ppn": "1",
                "train_walltime": str(horizon["train_walltime_single"]),
            }
        )
        rows.append(
            {
                **base,
                "exp_id": f"imagenet32_ddp_strict_k100_{tag}",
                "train_mode": "ddp_fullk",
                "config": str(PROJECT_DIR / "configs" / "imagenet32_ddp_strict.yaml"),
                "world_size": "4",
                "pipe_stages": "1",
                "weight_mode": "deep_only",
                "steps": steps,
                "num_nodes": "1",
                "ppn": "4",
                "train_walltime": str(horizon["train_walltime_multi"]),
            }
        )
        rows.append(
            {
                **base,
                "exp_id": f"imagenet32_pipeline_strict_p4_k100_{tag}",
                "train_mode": "pipe_strict",
                "config": str(PROJECT_DIR / "configs" / "imagenet32_pipeline_strict.yaml"),
                "world_size": "4",
                "pipe_stages": "4",
                "weight_mode": "last2_beta",
                "steps": steps,
                "num_nodes": "1",
                "ppn": "4",
                "train_walltime": str(horizon["train_walltime_multi"]),
            }
        )
    return rows


def write_manifest(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser("Write the ImageNet-32 Phase A manifest")
    ap.add_argument("--out", default=str(MANIFEST_PATH))
    args = ap.parse_args()
    rows = build_phasea_rows()
    write_manifest(Path(args.out), rows)
    print(f"[DONE] wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the requested 500k ablation manifest rows."""

import argparse
import csv
from pathlib import Path
from typing import Dict, List


MANIFEST_PATH = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/experiments/ablation_manifest_500k.csv")
DATA_DIR = "/eagle/lc-mpi/Zhiqing/ebm/data/cifar10"


def base_row() -> Dict[str, str]:
    return {
        "submit": "1",
        "group": "P500K",
        "story": "500k mainline followup",
        "hypothesis": "Longer training may separate stable baseline and pipeline behavior",
        "eval_images": "5000",
        "notes": "mainline_500k",
        "eval_walltime": "02:00:00",
        "queue": "preemptable",
        "save_every": "5000",
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
        "log_every": "10",
        "eval_batch": "256",
        "k_eval": "100",
        "step_size": "1.0",
        "lr": "1e-4",
        "last2_beta": "0.01",
        "steps": "500000",
    }


def build_500k_rows() -> List[Dict[str, str]]:
    base = base_row()
    rows = [
        {
            **base,
            "exp_id": "single_fullk_K100_seed1_500k",
            "train_mode": "single_fullk",
            "world_size": "1",
            "pipe_stages": "1",
            "K": "100",
            "weight_mode": "uniform",
            "seed": "1",
            "num_nodes": "1",
            "ppn": "1",
            "train_walltime": "18:00:00",
        },
        {
            **base,
            "exp_id": "ddp_fullk_K100_seed1_500k",
            "train_mode": "ddp_fullk",
            "world_size": "4",
            "pipe_stages": "0",
            "K": "100",
            "weight_mode": "uniform",
            "seed": "1",
            "num_nodes": "1",
            "ppn": "4",
            "train_walltime": "24:00:00",
        },
        {
            **base,
            "exp_id": "ddp_fullk_K100_seed2_500k",
            "train_mode": "ddp_fullk",
            "world_size": "4",
            "pipe_stages": "0",
            "K": "100",
            "weight_mode": "uniform",
            "seed": "2",
            "num_nodes": "1",
            "ppn": "4",
            "train_walltime": "24:00:00",
        },
        {
            **base,
            "exp_id": "ddp_fullk_K100_seed3_500k",
            "train_mode": "ddp_fullk",
            "world_size": "4",
            "pipe_stages": "0",
            "K": "100",
            "weight_mode": "uniform",
            "seed": "3",
            "num_nodes": "1",
            "ppn": "4",
            "train_walltime": "24:00:00",
        },
        {
            **base,
            "exp_id": "pipe_strict_P4_K100_beta001_lr1e4_seed1_500k",
            "train_mode": "pipe_strict",
            "world_size": "4",
            "pipe_stages": "4",
            "K": "100",
            "weight_mode": "last2_beta",
            "seed": "1",
            "num_nodes": "1",
            "ppn": "4",
            "train_walltime": "12:00:00",
        },
        {
            **base,
            "exp_id": "pipe_strict_P4_K100_beta001_lr1e4_seed2_500k",
            "train_mode": "pipe_strict",
            "world_size": "4",
            "pipe_stages": "4",
            "K": "100",
            "weight_mode": "last2_beta",
            "seed": "2",
            "num_nodes": "1",
            "ppn": "4",
            "train_walltime": "12:00:00",
        },
        {
            **base,
            "exp_id": "pipe_strict_P4_K100_beta001_lr1e4_seed3_500k",
            "train_mode": "pipe_strict",
            "world_size": "4",
            "pipe_stages": "4",
            "K": "100",
            "weight_mode": "last2_beta",
            "seed": "3",
            "num_nodes": "1",
            "ppn": "4",
            "train_walltime": "12:00:00",
        },
    ]
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
    ap = argparse.ArgumentParser("Write the 500k manifest")
    ap.add_argument("--out", default=str(MANIFEST_PATH))
    args = ap.parse_args()
    rows = build_500k_rows()
    write_manifest(Path(args.out), rows)
    print(f"[DONE] wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

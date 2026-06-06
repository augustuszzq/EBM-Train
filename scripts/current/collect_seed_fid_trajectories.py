#!/usr/bin/env python3
"""Aggregate seed trajectory re-evaluation points into one flat CSV."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


RUNS_ANALYSIS_DIR = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_analysis")
DEFAULT_OUT_CSV = RUNS_ANALYSIS_DIR / "seed_fid_trajectory_points.csv"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Collect seed trajectory FID points into one CSV")
    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Seed ids to include. Use --seeds 1 2 3 after seed3 finishes.",
    )
    ap.add_argument(
        "--out-csv",
        default=str(DEFAULT_OUT_CSV),
        help="Output CSV path",
    )
    return ap.parse_args()


def default_specs(seeds: Iterable[int]) -> List[Dict]:
    specs: List[Dict] = []
    for seed in seeds:
        specs.append(
            {
                "seed": seed,
                "method": "ddp",
                "label": f"ddp_seed{seed}_local",
                "trajectory_dir": str(RUNS_ANALYSIS_DIR / f"seed{seed}_fid_trajectory" / f"ddp_seed{seed}_local"),
            }
        )
        specs.append(
            {
                "seed": seed,
                "method": "pipeline",
                "label": f"pipeline_seed{seed}_local",
                "trajectory_dir": str(RUNS_ANALYSIS_DIR / f"seed{seed}_fid_trajectory" / f"pipeline_seed{seed}_local"),
            }
        )
    return specs


def collect_rows(specs: Iterable[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for spec in specs:
        trajectory_dir = Path(spec["trajectory_dir"])
        for metrics_path in sorted(trajectory_dir.glob("step*/metrics_compare.json")):
            payload = json.loads(metrics_path.read_text())
            step = int(metrics_path.parent.name.replace("step", ""))
            rows.append(
                {
                    "seed": spec["seed"],
                    "method": spec["method"],
                    "label": spec["label"],
                    "step": step,
                    "fid_inception": payload.get("fid_inception_baseline_vs_real"),
                    "fid_feature": payload.get("fid_feature_baseline_vs_real"),
                    "unique_ratio": payload.get("unique_ratio"),
                    "trajectory_dir": str(trajectory_dir),
                    "eval_dir": str(metrics_path.parent),
                    "metrics_path": str(metrics_path),
                }
            )
    rows.sort(key=lambda row: (row["seed"], row["method"], row["step"]))
    return rows


def write_rows_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "method",
        "label",
        "step",
        "fid_inception",
        "fid_feature",
        "unique_ratio",
        "trajectory_dir",
        "eval_dir",
        "metrics_path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> int:
    args = parse_args()
    rows = collect_rows(default_specs(args.seeds))
    out_csv = Path(args.out_csv)
    write_rows_csv(out_csv, rows)
    print(f"[DONE] wrote {len(rows)} rows to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

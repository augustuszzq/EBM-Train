#!/usr/bin/env python3
"""Aggregate ImageNet-32 result files into CSV/JSON summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Aggregate ImageNet-32 results")
    ap.add_argument("--runs-root", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    return ap.parse_args()


def collect_rows(runs_root: Path) -> List[Dict]:
    rows: List[Dict] = []
    for metrics_path in sorted(runs_root.rglob("metrics_compare.json")):
        payload = json.loads(metrics_path.read_text())
        cond_path = metrics_path.parent / "conditional_acc.json"
        cond_payload = json.loads(cond_path.read_text()) if cond_path.exists() else {}
        rows.append(
            {
                "exp_id": str(payload.get("exp_id", metrics_path.parent.parent.name)),
                "seed": payload.get("seed", ""),
                "final_fid": payload.get("final_fid", payload.get("fid_inception_baseline_vs_real", "")),
                "best_fid": payload.get("best_fid", payload.get("final_fid", payload.get("fid_inception_baseline_vs_real", ""))),
                "top1_conditional_acc": cond_payload.get("top1_conditional_acc", ""),
                "per_class_avg_acc": cond_payload.get("per_class_avg_acc", ""),
                "metrics_path": str(metrics_path),
                "conditional_acc_path": (str(cond_path) if cond_path.exists() else ""),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "exp_id",
                "seed",
                "final_fid",
                "best_fid",
                "top1_conditional_acc",
                "per_class_avg_acc",
                "metrics_path",
                "conditional_acc_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(runs_root)
    csv_path = out_dir / "imagenet32_summary.csv"
    json_path = out_dir / "imagenet32_summary.json"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2))
    print("[DONE] rows=%d csv=%s json=%s" % (len(rows), csv_path, json_path), flush=True)


if __name__ == "__main__":
    main()

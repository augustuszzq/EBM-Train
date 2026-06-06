#!/usr/bin/env python3
"""Aggregate replay trajectory outputs into point/summary/table artifacts."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    from ablation_replay_common import collect_replay_points, load_rows, summarize_replay_points, write_rows
except Exception:
    from polaris_ebm.scripts.current.ablation_replay_common import collect_replay_points, load_rows, summarize_replay_points, write_rows


PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Collect ablation replay outputs")
    ap.add_argument("--submitted_manifest", default=str(PROJECT_DIR / "experiments" / "ablation_replay_submitted.csv"))
    ap.add_argument("--out_dir", default="/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation_replay")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    replay_rows = load_rows(Path(args.submitted_manifest))
    points = collect_replay_points(replay_rows)
    summary = summarize_replay_points(points, replay_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points_csv = out_dir / "ablation_replay_points.csv"
    summary_csv = out_dir / "ablation_final_summary.csv"
    summary_json = out_dir / "ablation_final_summary.json"
    tables_md = out_dir / "ablation_tables_filled.md"

    point_fields = [
        "group",
        "exp_id",
        "story",
        "train_mode",
        "seed",
        "step",
        "fid_inception",
        "fid_feature",
        "unique_ratio",
        "trajectory_dir",
        "metrics_path",
    ]
    write_rows(points_csv, points, point_fields)

    summary_fields = [
        "group",
        "exp_id",
        "story",
        "train_mode",
        "seed",
        "steps",
        "expected_points",
        "available_points",
        "status",
        "final_step",
        "final_fid",
        "fid_feature",
        "unique_ratio",
        "best_step",
        "best_fid",
        "final_minus_best",
        "trajectory_dir",
        "run_dir",
    ]
    write_rows(summary_csv, summary, summary_fields)
    summary_json.write_text(json.dumps(summary, indent=2))

    subprocess.check_call(
        [
            sys.executable,
            str(PROJECT_DIR / "scripts" / "current" / "render_ablation_replay_tables.py"),
            "--summary_csv",
            str(summary_csv),
            "--out_md",
            str(tables_md),
        ]
    )
    print(f"[COLLECT] points={points_csv}")
    print(f"[COLLECT] summary={summary_csv}")
    print(f"[COLLECT] tables={tables_md}")


if __name__ == "__main__":
    main()

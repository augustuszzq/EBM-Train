#!/usr/bin/env python3
"""Collect pipeline-then-weighting 5k replay trajectories into one bundle."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from ablation_replay_common import collect_replay_points, load_rows, summarize_replay_points, write_rows
except Exception:
    from polaris_ebm.scripts.current.ablation_replay_common import (
        collect_replay_points,
        load_rows,
        summarize_replay_points,
        write_rows,
    )


PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
DEFAULT_MANIFEST = PROJECT_DIR / "experiments" / "pipeline_then_weighting_replay_manifest.csv"
DEFAULT_OUT_DIR = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_pipeline_then_weighting_replay")


def collect_rows_from_manifest(manifest_path: Path) -> Tuple[List[Dict], List[Dict]]:
    replay_rows = load_rows(manifest_path)
    points = collect_replay_points(replay_rows)
    summary = summarize_replay_points(points, replay_rows)
    return points, summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Collect pipeline-then-weighting replay trajectories")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    points, summary = collect_rows_from_manifest(manifest_path)

    points_csv = out_dir / "trajectory_points.csv"
    summary_csv = out_dir / "trajectory_summary.csv"
    summary_json = out_dir / "trajectory_summary.json"

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
    write_rows(points_csv, points, point_fields)
    write_rows(summary_csv, summary, summary_fields)
    summary_json.write_text(json.dumps(summary, indent=2))
    print(f"[COLLECT] points={points_csv}")
    print(f"[COLLECT] summary={summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

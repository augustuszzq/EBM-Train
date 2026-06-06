#!/usr/bin/env python3
"""Aggregate current CIFAR paper trajectories into a single fid-vs-step CSV."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_REGISTRY = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_final_bundle/master_experiment_registry.csv")
DEFAULT_OUT_DIR = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_final_bundle")


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def expected_points(row: Dict[str, str]) -> int:
    horizon = int(row.get("horizon_steps", "0") or "0")
    return horizon // 5000 if horizon > 0 else 0


def trajectory_csv(run_dir: str) -> Path:
    return Path(run_dir) / "trajectory" / "trajectory.csv"


def blocked_status(row: Dict[str, str]) -> str:
    if row.get("status") != "completed":
        return row.get("status", "")
    traj = trajectory_csv(row.get("canonical_run_dir", ""))
    if not traj.exists():
        return "missing_trajectory"
    points = load_rows(traj)
    if len(points) < expected_points(row):
        return "partial_trajectory"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser("Collect current CIFAR fid-vs-step table")
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    registry_path = Path(args.registry)
    out_dir = Path(args.out_dir)
    rows = [row for row in load_rows(registry_path) if row.get("ledger") == "current_cifar_paper"]

    point_rows: List[Dict[str, object]] = []
    blocked_rows: List[Dict[str, object]] = []
    coverage = {
        "total_current_cifar_logical_runs": len(rows),
        "completed_logical_runs": 0,
        "complete_trajectory_runs": 0,
        "blocked_or_incomplete_runs": 0,
    }

    for row in rows:
        status = row.get("status", "")
        traj_path = trajectory_csv(row.get("canonical_run_dir", ""))
        if status == "completed":
            coverage["completed_logical_runs"] += 1
        if status == "completed" and traj_path.exists():
            traj_rows = load_rows(traj_path)
            if len(traj_rows) >= expected_points(row):
                coverage["complete_trajectory_runs"] += 1
            for point in traj_rows:
                point_rows.append(
                    {
                        "ledger": row["ledger"],
                        "phase_group": row["phase_group"],
                        "family_id": row["family_id"],
                        "logical_run_id": row["logical_run_id"],
                        "benchmark": row["benchmark"],
                        "regime": row["regime"],
                        "stage_count": row["stage_count"],
                        "weighting": row["weighting"],
                        "K": row["K"],
                        "horizon_steps": row["horizon_steps"],
                        "seed": row["seed"],
                        "paper_role": row["paper_role"],
                        "canonical_run_dir": row["canonical_run_dir"],
                        "step": point.get("ordinal_step", point.get("step", "")),
                        "checkpoint_step": point.get("step", ""),
                        "fid_inception": point.get("fid_inception", ""),
                        "fid_feature": point.get("fid_feature", ""),
                        "unique_ratio": point.get("unique_ratio", ""),
                        "metrics_path": point.get("metrics_path", ""),
                    }
                )
        blocked = blocked_status(row)
        if blocked:
            coverage["blocked_or_incomplete_runs"] += 1
            blocked_rows.append(
                {
                    "phase_group": row["phase_group"],
                    "family_id": row["family_id"],
                    "logical_run_id": row["logical_run_id"],
                    "status": row["status"],
                    "blocked_reason": blocked,
                    "horizon_steps": row["horizon_steps"],
                    "seed": row["seed"],
                    "canonical_run_dir": row["canonical_run_dir"],
                    "notes": row.get("notes", ""),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    point_csv = out_dir / "current_cifar_fid_vs_step.csv"
    blocked_csv = out_dir / "current_cifar_fid_vs_step_blocked.csv"
    coverage_json = out_dir / "current_cifar_fid_vs_step_coverage.json"

    if point_rows:
        with point_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(point_rows[0].keys()))
            writer.writeheader()
            writer.writerows(point_rows)
    else:
        point_csv.write_text("")

    if blocked_rows:
        with blocked_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(blocked_rows[0].keys()))
            writer.writeheader()
            writer.writerows(blocked_rows)
    else:
        blocked_csv.write_text("")

    coverage_json.write_text(json.dumps(coverage, indent=2, sort_keys=True))
    print(json.dumps(coverage, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

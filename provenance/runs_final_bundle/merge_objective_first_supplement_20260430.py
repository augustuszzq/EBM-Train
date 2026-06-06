#!/usr/bin/env python3
import csv
import json
import shutil
import time
from pathlib import Path


BUNDLE = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_final_bundle")
FINAL = BUNDLE / "current_cifar_fid_vs_step.csv"
SUBMITTED = Path(
    "/eagle/lc-mpi/Zhiqing/polaris_ebm/experiments/objective_first_supplement_20260430_submitted.csv"
)


def family_id(row):
    if row["phase"] == "O5" and row["weight_mode"] == "uniform":
        return "pipe_P4_equal_K100"
    if row["weight_mode"] == "uniform":
        return "pipe_P4_equal"
    if row["weight_mode"] == "deep_only":
        return "pipe_P4_deepest"
    return "pipe_P4_other"


def weighting(row):
    if row["weight_mode"] == "uniform":
        return "equal"
    if row["weight_mode"] == "deep_only":
        return "deepest"
    return row["weight_mode"]


def paper_role(row):
    if row["phase"] == "O4" and row["weight_mode"] == "uniform":
        return "headline"
    if row["phase"] == "O5":
        return "control"
    return "appendix"


def logical_run_id(row, fam):
    return f"current_cifar_paper:cifar10:{fam}:seed{row['seed']}:h{row['steps']}"


def main():
    base_rows = list(csv.DictReader(FINAL.open()))
    if not base_rows:
        raise SystemExit(f"empty final CSV: {FINAL}")
    fieldnames = list(base_rows[0].keys())
    submitted_rows = list(csv.DictReader(SUBMITTED.open()))
    supplement_ids = {
        logical_run_id(row, family_id(row))
        for row in submitted_rows
    }
    rows = [row for row in base_rows if row.get("logical_run_id") not in supplement_ids]

    added = 0
    per_run = {}
    for row in submitted_rows:
        fam = family_id(row)
        logical = logical_run_id(row, fam)
        traj = Path(row["train_run_dir"]) / "trajectory" / "trajectory.csv"
        if not traj.exists():
            raise SystemExit(f"missing trajectory for {logical}: {traj}")
        traj_rows = list(csv.DictReader(traj.open()))
        expected = int(row["steps"]) // 5000
        if len(traj_rows) < expected:
            raise SystemExit(f"partial trajectory for {logical}: {len(traj_rows)}/{expected}")
        per_run[logical] = len(traj_rows)
        for point in traj_rows:
            rows.append(
                {
                    "ledger": "current_cifar_paper",
                    "phase_group": row["phase"],
                    "family_id": fam,
                    "logical_run_id": logical,
                    "benchmark": "cifar10",
                    "regime": "pipeline",
                    "stage_count": row["pipe_stages"],
                    "weighting": weighting(row),
                    "K": row["K"],
                    "horizon_steps": row["steps"],
                    "seed": row["seed"],
                    "paper_role": paper_role(row),
                    "canonical_run_dir": row["train_run_dir"],
                    "step": point.get("ordinal_step", point.get("step", "")),
                    "checkpoint_step": point.get("step", ""),
                    "fid_inception": point.get("fid_inception", ""),
                    "fid_feature": point.get("fid_feature", ""),
                    "unique_ratio": point.get("unique_ratio", ""),
                    "metrics_path": point.get("metrics_path", ""),
                }
            )
            added += 1

    backup = BUNDLE / (
        "current_cifar_fid_vs_step_pre_supplement_"
        + time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        + ".csv"
    )
    shutil.copy2(FINAL, backup)
    rows.sort(
        key=lambda row: (
            row.get("phase_group", ""),
            row.get("family_id", ""),
            int(row.get("horizon_steps") or 0),
            int(row.get("seed") or 0),
            int(float(row.get("step") or 0)),
        )
    )
    with FINAL.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ids = {row["logical_run_id"] for row in rows}
    coverage = {
        "total_current_cifar_logical_runs": len(ids),
        "completed_logical_runs": len(ids),
        "complete_trajectory_runs": len(ids),
        "blocked_or_incomplete_runs": 0,
        "supplemental_logical_runs": len(supplement_ids),
        "supplemental_rows_added": added,
        "supplemental_points_by_run": per_run,
        "backup_before_supplement": str(backup),
    }
    (BUNDLE / "current_cifar_fid_vs_step_coverage.json").write_text(
        json.dumps(coverage, indent=2, sort_keys=True)
    )
    print(json.dumps(coverage, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

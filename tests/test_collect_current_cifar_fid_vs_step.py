import csv
import json
import subprocess
from pathlib import Path


SCRIPT = "/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/collect_current_cifar_fid_vs_step.py"


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_collect_current_cifar_fid_vs_step(tmp_path):
    run_done = tmp_path / "run_done"
    traj_dir = run_done / "trajectory"
    traj_dir.mkdir(parents=True)
    write_csv(
        traj_dir / "trajectory.csv",
        [
            {"step": "4999", "ordinal_step": "5000", "fid_inception": "100.0", "fid_feature": "90.0", "unique_ratio": "0.9", "metrics_path": "m1"},
            {"step": "9999", "ordinal_step": "10000", "fid_inception": "80.0", "fid_feature": "70.0", "unique_ratio": "0.8", "metrics_path": "m2"},
        ],
    )
    run_partial = tmp_path / "run_partial"
    (run_partial / "trajectory").mkdir(parents=True)
    registry = tmp_path / "registry.csv"
    write_csv(
        registry,
        [
            {
                "ledger": "current_cifar_paper",
                "phase_group": "O1",
                "family_id": "single_P1_terminal",
                "logical_run_id": "a",
                "benchmark": "cifar10",
                "regime": "single",
                "stage_count": "1",
                "weighting": "terminal",
                "K": "100",
                "horizon_steps": "10000",
                "seed": "1",
                "status": "completed",
                "paper_role": "headline",
                "canonical_run_dir": str(run_done),
                "notes": "",
            },
            {
                "ledger": "current_cifar_paper",
                "phase_group": "O4",
                "family_id": "pipe_P4_equal",
                "logical_run_id": "b",
                "benchmark": "cifar10",
                "regime": "pipeline",
                "stage_count": "4",
                "weighting": "equal",
                "K": "100",
                "horizon_steps": "500000",
                "seed": "2",
                "status": "completed",
                "paper_role": "headline",
                "canonical_run_dir": str(run_partial),
                "notes": "",
            },
        ],
    )
    out_dir = tmp_path / "out"
    subprocess.check_call(["python3", SCRIPT, "--registry", str(registry), "--out-dir", str(out_dir)])
    point_rows = list(csv.DictReader((out_dir / "current_cifar_fid_vs_step.csv").open()))
    blocked_rows = list(csv.DictReader((out_dir / "current_cifar_fid_vs_step_blocked.csv").open()))
    coverage = json.loads((out_dir / "current_cifar_fid_vs_step_coverage.json").read_text())
    assert len(point_rows) == 2
    assert point_rows[0]["logical_run_id"] == "a"
    assert len(blocked_rows) == 1
    assert blocked_rows[0]["logical_run_id"] == "b"
    assert coverage["complete_trajectory_runs"] == 1

import csv
from pathlib import Path

from scripts.current.backfill_current_cifar_trajectories import needs_backfill, selected_rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_needs_backfill_completed_without_trajectory(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    row = {
        "ledger": "current_cifar_paper",
        "status": "completed",
        "canonical_run_dir": str(run_dir),
        "horizon_steps": "300000",
    }
    assert needs_backfill(row) is True


def test_selected_rows_round_robin_partition(tmp_path):
    rows = []
    for idx in range(4):
        run_dir = tmp_path / ("run%d" % idx)
        run_dir.mkdir()
        rows.append(
            {
                "ledger": "current_cifar_paper",
                "status": "completed",
                "canonical_run_dir": str(run_dir),
                "horizon_steps": "300000",
                "logical_run_id": "run-%d" % idx,
                "family_id": "fam-%d" % idx,
            }
        )
    registry = tmp_path / "registry.csv"
    write_csv(registry, rows)
    part0 = selected_rows(registry, rank_index=0, rank_count=2, targets=[])
    part1 = selected_rows(registry, rank_index=1, rank_count=2, targets=[])
    assert [row["logical_run_id"] for row in part0] == ["run-0", "run-2"]
    assert [row["logical_run_id"] for row in part1] == ["run-1", "run-3"]

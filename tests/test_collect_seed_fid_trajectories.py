import csv
import json
from pathlib import Path

from polaris_ebm.scripts.current.collect_seed_fid_trajectories import (
    collect_rows,
    write_rows_csv,
)


def write_metrics(path: Path, fid: float, feature: float, unique_ratio: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "fid_inception_baseline_vs_real": fid,
                "fid_feature_baseline_vs_real": feature,
                "unique_ratio": unique_ratio,
            }
        )
    )


def test_collect_rows_and_write_csv(tmp_path):
    seed1_ddp = tmp_path / "seed1_fid_trajectory" / "ddp_seed1_local"
    seed1_pipe = tmp_path / "seed1_fid_trajectory" / "pipeline_seed1_local"
    seed2_ddp = tmp_path / "seed2_fid_trajectory" / "ddp_seed2_local"
    write_metrics(seed1_ddp / "step005000" / "metrics_compare.json", 120.0, 0.9, 1.0)
    write_metrics(seed1_ddp / "step010000" / "metrics_compare.json", 110.0, 0.8, 1.0)
    write_metrics(seed1_pipe / "step005000" / "metrics_compare.json", 130.0, 1.1, 0.99)
    write_metrics(seed2_ddp / "step005000" / "metrics_compare.json", 100.0, 0.7, 1.0)

    specs = [
        {"seed": 1, "method": "ddp", "label": "ddp_seed1", "trajectory_dir": str(seed1_ddp)},
        {"seed": 1, "method": "pipeline", "label": "pipe_seed1", "trajectory_dir": str(seed1_pipe)},
        {"seed": 2, "method": "ddp", "label": "ddp_seed2", "trajectory_dir": str(seed2_ddp)},
    ]

    rows = collect_rows(specs)

    assert [(row["seed"], row["method"], row["step"]) for row in rows] == [
        (1, "ddp", 5000),
        (1, "ddp", 10000),
        (1, "pipeline", 5000),
        (2, "ddp", 5000),
    ]
    assert rows[0]["fid_inception"] == 120.0
    assert rows[2]["label"] == "pipe_seed1"

    out_csv = tmp_path / "trajectory_points.csv"
    write_rows_csv(out_csv, rows)

    written_rows = list(csv.DictReader(out_csv.open()))
    assert len(written_rows) == 4
    assert written_rows[0]["seed"] == "1"
    assert written_rows[0]["method"] == "ddp"
    assert written_rows[0]["step"] == "5000"
    assert written_rows[0]["fid_inception"] == "120.0"

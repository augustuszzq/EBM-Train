import csv
import json
from pathlib import Path

from polaris_ebm.scripts.current.ablation_replay_common import write_rows
from polaris_ebm.scripts.current.build_pipeline_then_weighting_replay_manifest import build_rows
from polaris_ebm.scripts.current.collect_pipeline_then_weighting_replay import collect_rows_from_manifest


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_build_rows_prefers_local_run_and_uses_run_trajectory_dir(tmp_path: Path) -> None:
    pbs_run = tmp_path / "pbs_run"
    local_run = tmp_path / "local_run"
    other_run = tmp_path / "other_run"

    write_text(pbs_run / "train.log", "[DONE]\n")
    write_text(local_run / "train.log", "[DONE]\n")
    write_text(other_run / "train.log", "[DONE]\n")
    write_text(local_run / "checkpoints" / "ckpt_step4999.pt", "")
    write_text(local_run / "checkpoints" / "ckpt_step9999.pt", "")
    write_text(other_run / "checkpoints" / "ckpt_step4999.pt", "")

    pbs_rows = [
        {
            "phase": "E1",
            "group": "E1",
            "exp_id": "exp_a",
            "story": "pbs duplicate",
            "train_mode": "single_fullk",
            "seed": "1",
            "K": "100",
            "steps": "10000",
            "step_size": "1.0",
            "weight_mode": "deep_only",
            "train_run_dir": str(pbs_run),
        },
        {
            "phase": "E2",
            "group": "E2",
            "exp_id": "exp_b",
            "story": "pbs only",
            "train_mode": "single_pipe_emul",
            "seed": "1",
            "K": "100",
            "steps": "5000",
            "step_size": "1.0",
            "weight_mode": "uniform",
            "train_run_dir": str(other_run),
        },
    ]
    local_rows = [
        {
            "phase": "E1",
            "group": "E1",
            "exp_id": "exp_a",
            "story": "local preferred",
            "train_mode": "single_fullk",
            "seed": "1",
            "K": "100",
            "steps": "10000",
            "step_size": "1.0",
            "weight_mode": "deep_only",
            "train_run_dir": str(local_run),
        }
    ]

    rows = build_rows(pbs_rows, local_rows)
    assert [row["exp_id"] for row in rows] == ["exp_a", "exp_b"]
    assert rows[0]["run_dir"] == str(local_run)
    assert rows[0]["trajectory_dir"] == str(local_run / "trajectory")
    assert rows[0]["expected_points"] == 2
    assert rows[1]["run_dir"] == str(other_run)
    assert rows[1]["expected_points"] == 1


def test_collect_rows_from_manifest_aggregates_points_and_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    trajectory_dir = run_dir / "trajectory"
    manifest = tmp_path / "replay_manifest.csv"

    write_json(
        trajectory_dir / "step005000" / "metrics_compare.json",
        {
            "fid_inception_baseline_vs_real": 120.0,
            "fid_feature_baseline_vs_real": 1.2,
            "unique_ratio": 1.0,
        },
    )
    write_json(
        trajectory_dir / "step010000" / "metrics_compare.json",
        {
            "fid_inception_baseline_vs_real": 90.0,
            "fid_feature_baseline_vs_real": 0.9,
            "unique_ratio": 1.0,
        },
    )

    rows = [
        {
            "group": "E2",
            "exp_id": "exp_c",
            "story": "trajectory",
            "train_mode": "single_pipe_emul",
            "seed": 1,
            "K": "100",
            "steps": 10000,
            "step_size": "1.0",
            "weight_mode": "uniform",
            "run_dir": str(run_dir),
            "trajectory_dir": str(trajectory_dir),
            "expected_points": 2,
            "replay_mode": "existing",
            "replay_walltime": "02:00:00",
            "replay_queue": "local",
            "replay_job_id": "",
        }
    ]
    write_rows(manifest, rows, list(rows[0].keys()))

    points, summary = collect_rows_from_manifest(manifest)
    assert len(points) == 2
    assert [point["step"] for point in points] == [5000, 10000]
    assert summary[0]["exp_id"] == "exp_c"
    assert summary[0]["best_step"] == 10000
    assert summary[0]["best_fid"] == 90.0
    assert summary[0]["final_step"] == 10000
    assert summary[0]["status"] == "done"

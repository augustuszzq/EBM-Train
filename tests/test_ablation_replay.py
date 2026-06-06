import json
from pathlib import Path

from polaris_ebm.scripts.current.ablation_replay_common import (
    build_replay_rows,
    collect_replay_points,
    summarize_replay_points,
    trajectory_progress_summary,
)


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ckpt")


def write_metrics(path: Path, fid: float, feature: float, unique_ratio: float = 1.0) -> None:
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


def test_build_replay_rows_uses_override_and_counts_expected_points(tmp_path):
    run_dir = tmp_path / "run_a"
    ckpt_dir = run_dir / "checkpoints"
    touch(ckpt_dir / "ckpt_step4999.pt")
    touch(ckpt_dir / "ckpt_step9999.pt")
    touch(ckpt_dir / "ckpt_step14999.pt")
    touch(ckpt_dir / "ckpt_step19999.pt")

    rows = build_replay_rows(
        source_rows=[
            {
                "group": "A1",
                "exp_id": "A1_case",
                "story": "story",
                "train_mode": "single_fullk",
                "steps": "20000",
                "seed": "1",
                "K": "100",
                "step_size": "1.0",
                "weight_mode": "uniform",
                "train_run_dir": str(run_dir),
            }
        ],
        replay_root=tmp_path / "replay_root",
        overrides={"A1_case": str(tmp_path / "existing_traj")},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["trajectory_dir"] == str(tmp_path / "existing_traj")
    assert row["expected_points"] == 4
    assert row["replay_walltime"] == "02:00:00"


def test_build_replay_rows_keeps_final_only_rows_without_checkpoints(tmp_path):
    run_dir = tmp_path / "run_final_only"
    (run_dir / "eval").mkdir(parents=True)
    write_metrics(run_dir / "eval" / "metrics_compare.json", fid=61.5, feature=0.5)

    rows = build_replay_rows(
        source_rows=[
            {
                "group": "A0",
                "exp_id": "A0_single_ref",
                "story": "ref",
                "train_mode": "single_fullk",
                "steps": "300000",
                "seed": "1",
                "K": "100",
                "step_size": "1.0",
                "weight_mode": "uniform",
                "train_run_dir": str(run_dir),
            }
        ],
        replay_root=tmp_path / "replay_root",
        overrides={},
    )

    assert len(rows) == 1
    assert rows[0]["replay_mode"] == "final_only"
    assert rows[0]["trajectory_dir"] == ""
    assert rows[0]["expected_points"] == 0


def test_progress_and_summary_detect_partial_and_best_step(tmp_path):
    traj_dir = tmp_path / "traj"
    write_metrics(traj_dir / "step005000" / "metrics_compare.json", fid=120.0, feature=0.9)
    write_metrics(traj_dir / "step010000" / "metrics_compare.json", fid=80.0, feature=0.7)

    progress = trajectory_progress_summary(traj_dir, expected_points=4)
    assert progress["available_points"] == 2
    assert progress["status"] == "partial"
    assert progress["final_step"] == 10000
    assert progress["final_fid"] == 80.0

    replay_rows = [
        {
            "exp_id": "demo_case",
            "group": "A1",
            "seed": 1,
            "trajectory_dir": str(traj_dir),
            "expected_points": 4,
        }
    ]
    points = collect_replay_points(replay_rows)
    assert len(points) == 2
    summary = summarize_replay_points(points, replay_rows)
    assert len(summary) == 1
    assert summary[0]["exp_id"] == "demo_case"
    assert summary[0]["best_step"] == 10000
    assert summary[0]["best_fid"] == 80.0
    assert summary[0]["final_minus_best"] == 0.0


def test_summary_falls_back_to_final_metrics_when_no_points_exist(tmp_path):
    run_dir = tmp_path / "run_final_only"
    write_metrics(run_dir / "eval" / "metrics_compare.json", fid=61.5, feature=0.42, unique_ratio=1.0)

    replay_rows = [
        {
            "exp_id": "final_only_case",
            "group": "A0",
            "seed": 1,
            "steps": 300000,
            "trajectory_dir": "",
            "expected_points": 0,
            "run_dir": str(run_dir),
        }
    ]
    summary = summarize_replay_points([], replay_rows)
    assert len(summary) == 1
    assert summary[0]["status"] == "final_only"
    assert summary[0]["final_fid"] == 61.5
    assert summary[0]["fid_feature"] == 0.42
    assert summary[0]["best_step"] == ""

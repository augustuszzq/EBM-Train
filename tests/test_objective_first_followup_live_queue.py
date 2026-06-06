from pathlib import Path

from polaris_ebm.scripts.current.objective_first_followup_live_queue import (
    build_specs,
    has_ready_checkpoint,
    launchable_specs,
)


def test_build_specs_keeps_only_submitted_rows(tmp_path):
    rows = [
        {
            "submit": "yes",
            "train_job_id": "123",
            "train_run_dir": str(tmp_path / "run_a"),
            "steps": "300000",
            "exp_id": "O1_a",
        },
        {
            "submit": "no",
            "train_job_id": "",
            "train_run_dir": str(tmp_path / "run_b"),
            "steps": "300000",
            "exp_id": "O2_b",
        },
    ]
    specs = build_specs(rows)
    assert len(specs) == 1
    assert specs[0]["exp_id"] == "O1_a"
    assert specs[0]["expected_points"] == "60"
    assert specs[0]["trajectory_dir"].endswith("/trajectory")


def test_launchable_specs_requires_checkpoint_and_incomplete_progress(tmp_path):
    run_dir = tmp_path / "run"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    row = {
        "exp_id": "O1_a",
        "run_dir": str(run_dir),
        "trajectory_dir": str(tmp_path / "traj"),
        "expected_points": "60",
    }
    assert not has_ready_checkpoint(run_dir)
    assert launchable_specs([row], active_out_dirs=set()) == []

    (ckpt_dir / "ckpt_step4999.pt").write_bytes(b"")
    launchable = launchable_specs([row], active_out_dirs=set())
    assert len(launchable) == 1
    assert launchable[0]["exp_id"] == "O1_a"

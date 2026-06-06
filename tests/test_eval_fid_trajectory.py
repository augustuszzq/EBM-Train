from pathlib import Path

from polaris_ebm.scripts.current.eval_fid_trajectory import (
    discover_trajectory_checkpoints,
    resolve_eval_python,
    summarize_trajectory_rows,
)


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ckpt")


def test_discover_trajectory_checkpoints_sorts_all_5000_step_checkpoints(tmp_path):
    ckpt_dir = tmp_path / "run" / "artifacts" / "checkpoints"
    touch(ckpt_dir / "ckpt_step14999.pt")
    touch(ckpt_dir / "ckpt_step4999.pt")
    touch(ckpt_dir / "ckpt_step9999.pt")
    touch(ckpt_dir / "ckpt_step299999.pt")

    ckpts = discover_trajectory_checkpoints(tmp_path / "run")

    assert [item["step"] for item in ckpts] == [4999, 9999, 14999, 299999]
    assert [item["ordinal_step"] for item in ckpts] == [5000, 10000, 15000, 300000]


def test_summarize_trajectory_rows_reports_best_step_and_final_match_delta():
    rows = [
        {"step": 4999, "ordinal_step": 5000, "fid_inception": 180.0},
        {"step": 9999, "ordinal_step": 10000, "fid_inception": 120.0},
        {"step": 14999, "ordinal_step": 15000, "fid_inception": 90.0},
        {"step": 19999, "ordinal_step": 20000, "fid_inception": 95.0},
    ]

    summary = summarize_trajectory_rows(rows, reference_final_fid=94.5)

    assert summary["num_points"] == 4
    assert summary["best_step"] == 15000
    assert summary["best_fid"] == 90.0
    assert summary["final_step"] == 20000
    assert summary["final_fid"] == 95.0
    assert summary["final_match_delta"] == 0.5
    assert summary["first_sub100_step"] == 15000
    assert summary["first_sub70_step"] is None


def test_resolve_eval_python_prefers_explicit_env_override(monkeypatch):
    monkeypatch.setenv("EVAL_PYTHON", "/tmp/custom-python")

    assert resolve_eval_python() == "/tmp/custom-python"

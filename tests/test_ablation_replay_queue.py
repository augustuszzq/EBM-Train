from pathlib import Path

from polaris_ebm.scripts.current.ablation_replay_queue import (
    available_launch_gpus,
    build_launch_spec,
    launchable_replay_rows,
    parse_active_out_dirs,
    parse_busy_gpu_indices,
)


def write_metrics(path: Path, fid: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"fid_inception_baseline_vs_real": %s}' % fid)


def test_parse_active_out_dirs_extracts_running_trajectory_dirs():
    ps_output = "\n".join(
        [
            "123 /home/x/python scripts/current/eval_fid_trajectory.py --run-dir /runs/a --out-dir /traj/a --label a",
            "124 /home/x/python scripts/other.py --out-dir /ignore/me",
            "125 /home/x/python scripts/current/eval_fid_trajectory.py --out-dir /traj/b --run-dir /runs/b",
        ]
    )
    assert parse_active_out_dirs(ps_output) == {"/traj/a", "/traj/b"}


def test_parse_busy_gpu_indices_maps_gpu_uuid_to_indices():
    gpu_csv = "\n".join(
        [
            "0, GPU-aaa",
            "1, GPU-bbb",
            "2, GPU-ccc",
        ]
    )
    apps_csv = "\n".join(
        [
            "GPU-bbb, 101, python, 800 MiB",
            "GPU-ccc, 102, python, 900 MiB",
        ]
    )
    assert parse_busy_gpu_indices(gpu_csv, apps_csv) == {"1", "2"}


def test_launchable_replay_rows_skips_done_final_only_and_active(tmp_path):
    active_dir = tmp_path / "active"
    done_dir = tmp_path / "done"
    pending_dir = tmp_path / "pending"
    write_metrics(done_dir / "step005000" / "metrics_compare.json", fid=100.0)
    write_metrics(done_dir / "step010000" / "metrics_compare.json", fid=90.0)
    write_metrics(done_dir / "step015000" / "metrics_compare.json", fid=80.0)
    write_metrics(done_dir / "step020000" / "metrics_compare.json", fid=70.0)
    write_metrics(active_dir / "step005000" / "metrics_compare.json", fid=110.0)

    rows = [
        {"exp_id": "final_only", "trajectory_dir": "", "expected_points": "0"},
        {"exp_id": "done_case", "trajectory_dir": str(done_dir), "expected_points": "4"},
        {"exp_id": "active_case", "trajectory_dir": str(active_dir), "expected_points": "4"},
        {"exp_id": "pending_case", "trajectory_dir": str(pending_dir), "expected_points": "4"},
    ]

    launchable = launchable_replay_rows(rows, active_out_dirs={str(active_dir)})
    assert [row["exp_id"] for row in launchable] == ["pending_case"]


def test_build_launch_spec_uses_gpu_and_skip_existing(tmp_path):
    row = {
        "exp_id": "demo_case",
        "run_dir": "/runs/demo",
        "trajectory_dir": str(tmp_path / "traj"),
        "seed": "3",
    }
    spec = build_launch_spec(
        row=row,
        gpu="1",
        project_dir=Path("/proj"),
        eval_python="/env/python",
        data_dir="/data/cifar10",
    )
    assert spec["log_path"] == str(tmp_path / "traj" / "run.log")
    assert spec["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert spec["env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert "--skip-existing" in spec["cmd"]
    assert spec["cmd"][0] == "/env/python"
    assert spec["cmd"][1] == "/proj/scripts/current/eval_fid_trajectory.py"


def test_available_launch_gpus_respects_active_parallel_slots():
    gpu_ids = ["0", "1", "2", "3"]
    busy = set()
    active = {"/traj/a", "/traj/b"}
    assert available_launch_gpus(
        gpu_ids=gpu_ids,
        busy_gpu_indices=busy,
        active_out_dirs=active,
        max_parallel=4,
    ) == ["0", "1"]

    assert available_launch_gpus(
        gpu_ids=gpu_ids,
        busy_gpu_indices={"0"},
        active_out_dirs=active,
        max_parallel=4,
    ) == ["1", "2"]

    assert available_launch_gpus(
        gpu_ids=gpu_ids,
        busy_gpu_indices=set(),
        active_out_dirs={"a", "b", "c", "d"},
        max_parallel=4,
    ) == []

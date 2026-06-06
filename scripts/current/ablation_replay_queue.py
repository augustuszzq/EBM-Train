#!/usr/bin/env python3
"""Run replay trajectories locally with bounded parallelism."""

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Set

try:
    from ablation_replay_common import load_rows, trajectory_progress_summary
except Exception:
    from polaris_ebm.scripts.current.ablation_replay_common import load_rows, trajectory_progress_summary


PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
DEFAULT_MANIFEST = PROJECT_DIR / "experiments" / "ablation_replay_submitted.csv"
DEFAULT_EVAL_PYTHON = "/home/kevienzzq/.conda/envs/llm-env/bin/python"
DEFAULT_DATA_DIR = "/eagle/lc-mpi/Zhiqing/ebm/data/cifar10"
DEFAULT_STATE_JSON = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation_replay/local_queue_runner_state.json")


def parse_active_out_dirs(ps_output: str) -> Set[str]:
    active: Set[str] = set()
    for raw_line in ps_output.splitlines():
        line = raw_line.strip()
        if not line or "eval_fid_trajectory.py" not in line:
            continue
        try:
            tokens = shlex.split(line)
        except Exception:
            tokens = line.split()
        if "--out-dir" not in tokens:
            continue
        idx = tokens.index("--out-dir")
        if idx + 1 < len(tokens):
            active.add(tokens[idx + 1])
    return active


def parse_busy_gpu_indices(gpu_csv: str, apps_csv: str) -> Set[str]:
    uuid_to_index: Dict[str, str] = {}
    for line in gpu_csv.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0] and parts[1]:
            uuid_to_index[parts[1]] = parts[0]
    busy: Set[str] = set()
    for line in apps_csv.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not parts or not parts[0] or parts[0].startswith("No running"):
            continue
        gpu_uuid = parts[0]
        if gpu_uuid in uuid_to_index:
            busy.add(uuid_to_index[gpu_uuid])
    return busy


def available_launch_gpus(
    gpu_ids: List[str],
    busy_gpu_indices: Set[str],
    active_out_dirs: Set[str],
    max_parallel: int,
) -> List[str]:
    active_count = len(active_out_dirs)
    slots = max(0, max_parallel - active_count)
    if slots == 0:
        return []
    return [gpu for gpu in gpu_ids if gpu not in busy_gpu_indices][:slots]


def launchable_replay_rows(rows: Iterable[Dict[str, str]], active_out_dirs: Set[str]) -> List[Dict[str, str]]:
    launchable: List[Dict[str, str]] = []
    for row in rows:
        trajectory_dir = row.get("trajectory_dir", "").strip()
        expected_points = int(row.get("expected_points", "0") or "0")
        if not trajectory_dir or expected_points <= 0:
            continue
        if trajectory_dir in active_out_dirs:
            continue
        progress = trajectory_progress_summary(Path(trajectory_dir), expected_points)
        if progress["status"] != "done":
            launchable.append(row)
    return launchable


def build_launch_spec(
    row: Dict[str, str],
    gpu: str,
    project_dir: Path,
    eval_python: str,
    data_dir: str,
) -> Dict[str, object]:
    trajectory_dir = Path(row["trajectory_dir"])
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "EVAL_PYTHON": eval_python,
            "PYTHONPATH": "/eagle/lc-mpi/Zhiqing" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        }
    )
    cmd = [
        eval_python,
        str(project_dir / "scripts" / "current" / "eval_fid_trajectory.py"),
        "--run-dir",
        row["run_dir"],
        "--out-dir",
        row["trajectory_dir"],
        "--step-interval",
        "5000",
        "--eval-images",
        "5000",
        "--eval-batch",
        "256",
        "--k-eval",
        "100",
        "--step-size",
        row.get("step_size", "1.0") or "1.0",
        "--noise-std",
        "0.01",
        "--langevin-sign",
        "1.0",
        "--eval-seed",
        row.get("seed", "1") or "1",
        "--device",
        "cuda",
        "--data-dir",
        data_dir,
        "--label",
        row["exp_id"],
        "--skip-existing",
    ]
    return {
        "cmd": cmd,
        "env": env,
        "log_path": str(trajectory_dir / "run.log"),
    }


def run_text(cmd: List[str]) -> str:
    out = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.STDOUT)
    return out


def query_active_out_dirs() -> Set[str]:
    ps_output = run_text(["ps", "-eo", "pid=,args="])
    return parse_active_out_dirs(ps_output)


def query_busy_gpu_indices() -> Set[str]:
    gpu_csv = run_text(["nvidia-smi", "--query-gpu=index,gpu_uuid", "--format=csv,noheader"])
    apps_csv = run_text(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader"])
    return parse_busy_gpu_indices(gpu_csv, apps_csv)


def write_state(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Run local replay jobs with bounded parallelism")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--poll-seconds", type=int, default=20)
    ap.add_argument("--max-parallel", type=int, default=2)
    ap.add_argument("--eval-python", default=DEFAULT_EVAL_PYTHON)
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--state-json", default=str(DEFAULT_STATE_JSON))
    ap.add_argument("--once", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    max_parallel = min(args.max_parallel, len(gpu_ids))
    manifest_path = Path(args.manifest)
    state_path = Path(args.state_json)

    print(f"[QUEUE] manifest={manifest_path}")
    print(f"[QUEUE] gpus={','.join(gpu_ids)} max_parallel={max_parallel}")

    while True:
        rows = load_rows(manifest_path)
        active_out_dirs = query_active_out_dirs()
        busy_gpu_indices = query_busy_gpu_indices()
        pending_rows = launchable_replay_rows(rows, active_out_dirs=active_out_dirs)
        available_gpus = available_launch_gpus(
            gpu_ids=gpu_ids,
            busy_gpu_indices=busy_gpu_indices,
            active_out_dirs=active_out_dirs,
            max_parallel=max_parallel,
        )
        launched = []

        for gpu, row in zip(available_gpus, pending_rows):
            spec = build_launch_spec(
                row=row,
                gpu=gpu,
                project_dir=PROJECT_DIR,
                eval_python=args.eval_python,
                data_dir=args.data_dir,
            )
            log_path = Path(spec["log_path"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("ab") as logf:
                proc = subprocess.Popen(
                    spec["cmd"],
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env=spec["env"],
                    preexec_fn=os.setsid,
                )
            launched.append({"exp_id": row["exp_id"], "pid": proc.pid, "gpu": gpu, "trajectory_dir": row["trajectory_dir"]})
            print(f"[LAUNCH] gpu={gpu} exp_id={row['exp_id']} pid={proc.pid}")

        state = {
            "manifest": str(manifest_path),
            "gpus": gpu_ids,
            "busy_gpus": sorted(busy_gpu_indices),
            "active_out_dirs": sorted(active_out_dirs),
            "pending_exp_ids": [row["exp_id"] for row in pending_rows],
            "launched": launched,
        }
        write_state(state_path, state)

        if not pending_rows and not (set(gpu_ids) & busy_gpu_indices):
            print("[DONE] no pending replay rows remain")
            return 0
        if args.once:
            print("[ONCE] exiting after single scheduling pass")
            return 0

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

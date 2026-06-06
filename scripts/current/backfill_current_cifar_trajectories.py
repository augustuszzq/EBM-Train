#!/usr/bin/env python3
"""Backfill per-5k FID trajectories for completed current-CIFAR paper runs."""

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List

try:
    from ablation_replay_common import trajectory_progress_summary
except Exception:
    try:
        from scripts.current.ablation_replay_common import trajectory_progress_summary
    except Exception:
        from polaris_ebm.scripts.current.ablation_replay_common import trajectory_progress_summary


PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
DEFAULT_REGISTRY = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_final_bundle/master_experiment_registry.csv")
DEFAULT_EVAL_PYTHON = "/home/kevienzzq/.conda/envs/llm-env/bin/python"
STEP_INTERVAL = 5000


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def expected_points(row: Dict[str, str]) -> int:
    horizon = int(row.get("horizon_steps", "0") or "0")
    if horizon <= 0:
        return 0
    return horizon // STEP_INTERVAL


def trajectory_dir_for(row: Dict[str, str]) -> Path:
    return Path(row["canonical_run_dir"]) / "trajectory"


def trajectory_csv_for(row: Dict[str, str]) -> Path:
    return trajectory_dir_for(row) / "trajectory.csv"


def needs_backfill(row: Dict[str, str]) -> bool:
    if row.get("ledger") != "current_cifar_paper":
        return False
    if row.get("status") != "completed":
        return False
    run_dir = row.get("canonical_run_dir", "").strip()
    if not run_dir or not Path(run_dir).exists():
        return False
    expected = expected_points(row)
    if expected <= 0:
        return False
    progress = trajectory_progress_summary(trajectory_dir_for(row), expected)
    return progress["status"] != "done"


def selected_rows(
    registry_path: Path,
    rank_index: int,
    rank_count: int,
    targets: List[str],
) -> List[Dict[str, str]]:
    rows = [row for row in load_rows(registry_path) if needs_backfill(row)]
    rows.sort(key=lambda row: row["logical_run_id"])
    if targets:
        wanted = set(targets)
        rows = [row for row in rows if row["logical_run_id"] in wanted or row["family_id"] in wanted]
    if rank_count > 1:
        rows = [row for idx, row in enumerate(rows) if idx % rank_count == rank_index]
    return rows


def launch(row: Dict[str, str], gpu: str, eval_python: str) -> subprocess.Popen:
    run_dir = Path(row["canonical_run_dir"])
    trajectory_dir = trajectory_dir_for(row)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
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
        str(PROJECT_DIR / "scripts" / "current" / "eval_fid_trajectory.py"),
        "--run-dir",
        str(run_dir),
        "--out-dir",
        str(trajectory_dir),
        "--step-interval",
        str(STEP_INTERVAL),
        "--eval-images",
        "5000",
        "--eval-batch",
        "256",
        "--k-eval",
        str(row.get("K", "100") or "100"),
        "--step-size",
        "1.0",
        "--noise-std",
        "0.01",
        "--langevin-sign",
        "1.0",
        "--eval-seed",
        str(row.get("seed", "1") or "1"),
        "--device",
        "cuda",
        "--data-dir",
        "/eagle/lc-mpi/Zhiqing/ebm/data/cifar10",
        "--label",
        row["logical_run_id"],
        "--skip-existing",
    ]
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    log_path = trajectory_dir / "backfill_run.log"
    with log_path.open("ab") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            preexec_fn=os.setsid,
        )
    print("[LAUNCH] gpu=%s logical_run_id=%s pid=%s" % (gpu, row["logical_run_id"], proc.pid), flush=True)
    return proc


def write_state(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Backfill current CIFAR paper trajectories")
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--rank-index", type=int, default=0)
    ap.add_argument("--rank-count", type=int, default=1)
    ap.add_argument("--poll-seconds", type=int, default=15)
    ap.add_argument("--eval-python", default=DEFAULT_EVAL_PYTHON)
    ap.add_argument("--state-json", default="")
    ap.add_argument("--targets", nargs="*", default=[])
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    registry_path = Path(args.registry)
    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    rows = selected_rows(
        registry_path=registry_path,
        rank_index=args.rank_index,
        rank_count=args.rank_count,
        targets=args.targets,
    )
    print(
        "[QUEUE] registry=%s rank=%d/%d targets=%d gpus=%s"
        % (registry_path, args.rank_index, args.rank_count, len(rows), ",".join(gpu_ids)),
        flush=True,
    )
    active: Dict[str, subprocess.Popen] = {}
    gpu_for: Dict[str, str] = {}
    next_idx = 0
    while next_idx < len(rows) or active:
        while next_idx < len(rows) and len(active) < len(gpu_ids):
            gpu = next(g for g in gpu_ids if g not in gpu_for)
            row = rows[next_idx]
            proc = launch(row, gpu, args.eval_python)
            active[row["logical_run_id"]] = proc
            gpu_for[gpu] = row["logical_run_id"]
            next_idx += 1
        done = []
        for logical_run_id, proc in active.items():
            rc = proc.poll()
            if rc is not None:
                print("[DONE] logical_run_id=%s rc=%s" % (logical_run_id, rc), flush=True)
                done.append((logical_run_id, rc))
        for logical_run_id, _rc in done:
            del active[logical_run_id]
            for gpu, owner in list(gpu_for.items()):
                if owner == logical_run_id:
                    del gpu_for[gpu]
        if args.state_json:
            pending_ids = [row["logical_run_id"] for row in rows[next_idx:]]
            write_state(
                Path(args.state_json),
                {
                    "registry": str(registry_path),
                    "rank_index": args.rank_index,
                    "rank_count": args.rank_count,
                    "gpus": gpu_ids,
                    "active": sorted(active.keys()),
                    "pending": pending_ids,
                },
            )
        if active or next_idx < len(rows):
            time.sleep(args.poll_seconds)
    print("[DONE] backfill queue empty", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

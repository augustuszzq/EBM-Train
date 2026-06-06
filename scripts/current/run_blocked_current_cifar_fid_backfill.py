#!/usr/bin/env python3
"""Backfill missing current-CIFAR FID trajectories from the blocked table.

The collector marks completed logical runs as blocked when their canonical
``trajectory/trajectory.csv`` is absent or partial. This runner evaluates those
blocked rows locally with a small fixed number of GPU lanes and writes progress
to a JSON state file.
"""

import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List


PROJECT = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
BLOCKED_CSV = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_final_bundle/current_cifar_fid_vs_step_blocked.csv")
STATE_JSON = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_final_bundle/current_cifar_blocked_backfill_state.json")
PYTHON = "/home/kevienzzq/.conda/envs/llm-env/bin/python"
EVAL = PROJECT / "scripts" / "current" / "eval_fid_trajectory.py"


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def trajectory_points(out_dir: Path) -> List[int]:
    points = []
    for metrics in out_dir.glob("step*/metrics_compare.json"):
        match = re.search(r"step(\d+)", metrics.parent.name)
        if match:
            points.append(int(match.group(1)))
    return sorted(set(points))


def select_eval_run_dir(canonical_run_dir: Path) -> Path:
    merged = canonical_run_dir / "trajectory_merged_run"
    if (merged / "checkpoints").exists():
        return merged
    return canonical_run_dir


def build_targets() -> List[Dict[str, str]]:
    targets = []
    for row in read_rows(BLOCKED_CSV):
        if row.get("status") != "completed":
            continue
        canonical = Path(row["canonical_run_dir"])
        run_dir = select_eval_run_dir(canonical)
        out_dir = canonical / "trajectory"
        expected = int(row["horizon_steps"]) // 5000
        points = trajectory_points(out_dir)
        if len(points) >= expected:
            continue
        targets.append(
            {
                "logical_run_id": row["logical_run_id"],
                "family_id": row["family_id"],
                "seed": row["seed"],
                "horizon_steps": row["horizon_steps"],
                "run_dir": str(run_dir),
                "out_dir": str(out_dir),
                "expected_points": str(expected),
                "completed_points": str(len(points)),
            }
        )
    return targets


def launch(target: Dict[str, str], gpu: str):
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "EVAL_PYTHON": PYTHON,
            "PYTHONPATH": "/eagle/lc-mpi/Zhiqing" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        }
    )
    out_dir = Path(target["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "blocked_backfill.log"
    cmd = [
        PYTHON,
        str(EVAL),
        "--run-dir",
        target["run_dir"],
        "--label",
        target["logical_run_id"],
        "--out-dir",
        target["out_dir"],
        "--step-interval",
        "5000",
        "--eval-images",
        "5000",
        "--device",
        "cuda",
        "--eval-seed",
        target["seed"],
        "--skip-existing",
    ]
    log = log_path.open("ab")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
    return proc, log, str(log_path)


def write_state(queue: List[Dict[str, str]], active: Dict[str, object], completed: List[Dict[str, str]]) -> None:
    STATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "queue_remaining": len(queue),
        "active": {
            gpu: {
                "pid": proc.pid,
                "logical_run_id": target["logical_run_id"],
                "out_dir": target["out_dir"],
                "log_path": log_path,
                "points": len(trajectory_points(Path(target["out_dir"]))),
                "expected_points": int(target["expected_points"]),
            }
            for gpu, (proc, _log, target, log_path) in active.items()
        },
        "completed": completed,
    }
    STATE_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    gpus = [gpu.strip() for gpu in os.environ.get("BACKFILL_GPUS", "0,1").split(",") if gpu.strip()]
    if not gpus:
        raise SystemExit("BACKFILL_GPUS resolved to an empty GPU list")
    queue = build_targets()
    active: Dict[str, object] = {}
    completed: List[Dict[str, str]] = []
    print("[QUEUE] targets=%d gpus=%s" % (len(queue), ",".join(gpus)), flush=True)
    while queue or active:
        for gpu in gpus:
            if gpu in active or not queue:
                continue
            target = queue.pop(0)
            proc, log, log_path = launch(target, gpu)
            active[gpu] = (proc, log, target, log_path)
            print("[LAUNCH] gpu=%s pid=%s logical_run_id=%s" % (gpu, proc.pid, target["logical_run_id"]), flush=True)
        write_state(queue, active, completed)
        time.sleep(30)
        for gpu, (proc, log, target, log_path) in list(active.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log.close()
            points = len(trajectory_points(Path(target["out_dir"])))
            completed.append(
                {
                    "logical_run_id": target["logical_run_id"],
                    "returncode": str(rc),
                    "points": str(points),
                    "expected_points": target["expected_points"],
                    "out_dir": target["out_dir"],
                    "log_path": log_path,
                }
            )
            print(
                "[DONE] gpu=%s rc=%s points=%s/%s logical_run_id=%s"
                % (gpu, rc, points, target["expected_points"], target["logical_run_id"]),
                flush=True,
            )
            del active[gpu]
            if rc != 0:
                write_state(queue, active, completed)
                return rc
    write_state(queue, active, completed)
    print("[DONE] blocked current-CIFAR backfill complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

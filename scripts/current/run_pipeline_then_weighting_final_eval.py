#!/usr/bin/env python3
"""Run final local evals for completed pipeline-then-weighting runs."""

import argparse
import csv
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List


PROJECT_DIR = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm")
PBS_SUBMITTED = PROJECT_DIR / "experiments" / "pipeline_then_weighting_manifest_submitted.csv"
LOCAL_SUBMITTED = PROJECT_DIR / "experiments" / "pipeline_then_weighting_local_submitted.csv"
EVAL_WRAPPER = PROJECT_DIR / "scripts" / "current" / "run_eval_ablation_case.sh"


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def merged_rows() -> Dict[str, Dict[str, str]]:
    rows = {row["exp_id"]: dict(row) for row in load_rows(PBS_SUBMITTED)}
    for row in load_rows(LOCAL_SUBMITTED):
        rows[row["exp_id"]] = dict(row)
    return rows


def train_done(run_dir: Path) -> bool:
    log = run_dir / "train.log"
    return log.exists() and "[DONE]" in log.read_text(errors="ignore")


def has_final_eval(run_dir: Path) -> bool:
    return (run_dir / "trajectory" / "step300000" / "metrics_compare.json").exists() or (
        run_dir / "eval" / "metrics_compare.json"
    ).exists()


def missing_eval_rows() -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for exp_id, row in sorted(merged_rows().items()):
        run_dir = Path(row["train_run_dir"])
        if not run_dir.exists():
            continue
        if not train_done(run_dir):
            continue
        if has_final_eval(run_dir):
            continue
        out.append(row)
    return out


def launch(row: Dict[str, str], gpu: str) -> subprocess.Popen:
    run_dir = Path(row["train_run_dir"])
    env = os.environ.copy()
    env.update(
        {
            "PROJECT_DIR": str(PROJECT_DIR),
            "GROUP": row["group"],
            "EXP_ID": row["exp_id"],
            "TRAIN_MODE": row["train_mode"],
            "RUN_DIR": str(run_dir),
            "EVAL_DIR": str(run_dir / "eval"),
            "EVAL_IMAGES": row.get("eval_images", "5000") or "5000",
            "EVAL_BATCH": row.get("eval_batch", "256") or "256",
            "K_EVAL": row.get("k_eval", "100") or "100",
            "STEP_SIZE": row.get("step_size", "1.0") or "1.0",
            "NOISE_STD": row.get("noise_std", "0.01") or "0.01",
            "LANGEVIN_SIGN": row.get("langevin_sign", "1.0") or "1.0",
            "EVAL_SEED": row.get("seed", "1") or "1",
            "DATA_DIR": row.get("data_dir", "/eagle/lc-mpi/Zhiqing/ebm/data/cifar10") or "/eagle/lc-mpi/Zhiqing/ebm/data/cifar10",
            "CUDA_VISIBLE_DEVICES": gpu,
        }
    )
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    log_path = eval_dir / "final_eval_launcher.log"
    with log_path.open("ab") as logf:
        proc = subprocess.Popen(
            ["/usr/bin/bash", str(EVAL_WRAPPER)],
            stdout=logf,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            preexec_fn=os.setsid,
        )
    print(f"[LAUNCH] gpu={gpu} exp_id={row['exp_id']} pid={proc.pid} run_dir={run_dir}", flush=True)
    return proc


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Run local final evals for completed pipeline-then-weighting runs")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--poll-seconds", type=int, default=15)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    pending = missing_eval_rows()
    print(f"[QUEUE] pending_final_eval={len(pending)} gpus={','.join(gpus)}", flush=True)
    active: Dict[str, subprocess.Popen] = {}
    gpu_for: Dict[str, str] = {}
    idx = 0
    while idx < len(pending) or active:
        while idx < len(pending) and len(active) < len(gpus):
            gpu = next(g for g in gpus if g not in gpu_for)
            row = pending[idx]
            proc = launch(row, gpu)
            active[row["exp_id"]] = proc
            gpu_for[gpu] = row["exp_id"]
            idx += 1
        done = []
        for exp_id, proc in active.items():
            rc = proc.poll()
            if rc is not None:
                print(f"[DONE] exp_id={exp_id} rc={rc}", flush=True)
                done.append((exp_id, rc))
        for exp_id, _ in done:
            del active[exp_id]
            for gpu, owner in list(gpu_for.items()):
                if owner == exp_id:
                    del gpu_for[gpu]
        if active or idx < len(pending):
            time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

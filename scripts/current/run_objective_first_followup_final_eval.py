#!/usr/bin/env python3
"""Run local final evals for completed objective-first follow-up runs."""

import argparse
import csv
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List


PROJECT_DIR = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm")
SUBMITTED = PROJECT_DIR / "experiments" / "objective_first_followup_manifest_submitted.csv"
EVAL_WRAPPER = PROJECT_DIR / "scripts" / "current" / "run_eval_ablation_case.sh"


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def selected_rows(targets: List[str]) -> List[Dict[str, str]]:
    wanted = set(targets)
    rows = []
    for row in load_rows(SUBMITTED):
        if row.get("exp_id", "") not in wanted:
            continue
        run_dir = row.get("train_run_dir", "").strip()
        if not run_dir or not Path(run_dir).exists():
            phase = row.get("phase", "").strip()
            exp_id = row.get("exp_id", "").strip()
            if phase and exp_id:
                base = PROJECT_DIR / "runs_objective_first_followup" / phase
                matches = sorted(base.glob(f"{exp_id}_*"))
                if matches:
                    run_dir = str(matches[-1])
        if not run_dir:
            continue
        item = dict(row)
        item["train_run_dir"] = run_dir
        rows.append(item)
    return rows


def train_done(run_dir: Path) -> bool:
    job_info = run_dir / "job_info.json"
    if not job_info.exists():
        return False
    try:
        import json

        payload = json.loads(job_info.read_text())
    except Exception:
        return False
    return payload.get("train_status") == "succeeded"


def has_final_eval(run_dir: Path) -> bool:
    return (run_dir / "eval" / "metrics_compare.json").exists()


def pending_rows(targets: List[str]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in selected_rows(targets):
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
            "DATA_DIR": row.get("data_dir", "/eagle/lc-mpi/Zhiqing/ebm/data/cifar10")
            or "/eagle/lc-mpi/Zhiqing/ebm/data/cifar10",
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
    print(f"[LAUNCH] gpu={gpu} exp_id={row['exp_id']} pid={proc.pid}", flush=True)
    return proc


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Run local final evals for completed objective-first follow-up runs")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--poll-seconds", type=int, default=15)
    ap.add_argument("--targets", nargs="+", required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    pending = pending_rows(args.targets)
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
    print("[DONE] final eval queue empty", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Launch selected pipeline-then-weighting runs locally on specific GPUs."""

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
DEFAULT_SUBMITTED = PROJECT_DIR / "experiments" / "pipeline_then_weighting_manifest_submitted.csv"
DEFAULT_OUT_MANIFEST = PROJECT_DIR / "experiments" / "pipeline_then_weighting_local_submitted.csv"
DEFAULT_PAYLOAD_DIR = PROJECT_DIR / "experiments" / "_submission_payloads" / "pipeline_then_weighting_local"
LAUNCHER = PROJECT_DIR / "scripts" / "current" / "pbs_run_ablation_case.pbs"


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Launch local pipeline-then-weighting runs")
    ap.add_argument("--submitted-manifest", default=str(DEFAULT_SUBMITTED))
    ap.add_argument("--out-manifest", default=str(DEFAULT_OUT_MANIFEST))
    ap.add_argument("--exp-id", action="append", required=True)
    ap.add_argument("--gpu", action="append", required=True)
    ap.add_argument("--run-suffix", default="local")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.exp_id) != len(args.gpu):
        raise SystemExit("--exp-id and --gpu must have the same count")

    source_rows = {row["exp_id"]: row for row in load_rows(Path(args.submitted_manifest))}
    out_rows: List[Dict[str, str]] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    for exp_id, gpu in zip(args.exp_id, args.gpu):
        if exp_id not in source_rows:
            raise SystemExit(f"exp_id not found in submitted manifest: {exp_id}")
        row = dict(source_rows[exp_id])
        run_dir = (
            PROJECT_DIR
            / "runs_pipeline_then_weighting"
            / row["phase"]
            / f"{exp_id}_{args.run_suffix}_{timestamp}"
        )
        eval_dir = run_dir / "eval"
        payload_dir = DEFAULT_PAYLOAD_DIR
        payload_dir.mkdir(parents=True, exist_ok=True)
        payload_path = payload_dir / f"{exp_id}_{args.run_suffix}_{timestamp}.json"

        payload = {
            "phase": row["phase"],
            "group": row["group"],
            "exp_id": exp_id,
            "story": row.get("story", ""),
            "train_mode": row["train_mode"],
            "world_size": row["world_size"],
            "pipe_stages": row["pipe_stages"],
            "K": row["K"],
            "steps": row["steps"],
            "lr": row["lr"],
            "step_size": row["step_size"],
            "weight_mode": row["weight_mode"],
            "last2_beta": row["last2_beta"],
            "seed": row["seed"],
            "num_nodes": row["num_nodes"],
            "ppn": row["ppn"],
            "eval_images": row["eval_images"],
            "notes": row.get("notes", ""),
            "run_dir": str(run_dir),
            "eval_dir": str(eval_dir),
            "gpu": str(gpu),
            "launched_at": datetime.now(timezone.utc).isoformat(),
        }
        payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

        env = os.environ.copy()
        env.update(
            {
                "PROJECT_DIR": str(PROJECT_DIR),
                "GROUP": row["group"],
                "EXP_ID": exp_id,
                "STORY": row.get("story", ""),
                "TRAIN_MODE": row["train_mode"],
                "WORLD_SIZE": row["world_size"],
                "PIPE_STAGES": row["pipe_stages"],
                "K": row["K"],
                "STEPS": row["steps"],
                "LR": row["lr"],
                "STEP_SIZE": row["step_size"],
                "WEIGHT_MODE": row["weight_mode"],
                "LAST2_BETA": row["last2_beta"],
                "SEED": row["seed"],
                "NUM_NODES": row["num_nodes"],
                "PPN": row["ppn"],
                "EVAL_IMAGES": row["eval_images"],
                "NOTES": row.get("notes", ""),
                "RUN_DIR": str(run_dir),
                "EVAL_DIR": str(eval_dir),
                "MANIFEST_ROW_PATH": str(payload_path),
                "SAVE_EVERY": row.get("save_every", "5000") or "5000",
                "VIS_EVERY": row.get("vis_every", "1000") or "1000",
                "BATCH_SIZE": row.get("batch_size", "64") or "64",
                "SIGMA_PD": row.get("sigma_pd", "0.03") or "0.03",
                "NOISE_STD": row.get("noise_std", "0.01") or "0.01",
                "LANGEVIN_SIGN": row.get("langevin_sign", "1.0") or "1.0",
                "WEIGHT_DECAY": row.get("weight_decay", "0.0") or "0.0",
                "MAX_GRAD_NORM": row.get("max_grad_norm", "0.0") or "0.0",
                "DATA_DIR": row.get("data_dir", "/eagle/lc-mpi/Zhiqing/ebm/data/cifar10"),
                "SYNC_FRESH_INIT": row.get("sync_fresh_init", "1") or "1",
                "DEBUG_LEVEL": row.get("debug_level", "1") or "1",
                "LOG_EVERY": row.get("log_every", "10") or "10",
                "EVAL_BATCH": row.get("eval_batch", "256") or "256",
                "K_EVAL": row.get("k_eval", "100") or "100",
                "EVAL_SEED": row.get("seed", "1"),
                "CUDA_VISIBLE_DEVICES": str(gpu),
            }
        )

        launch_log = run_dir / "launcher.log"
        launch_log.parent.mkdir(parents=True, exist_ok=True)
        with launch_log.open("ab") as log_handle:
            proc = subprocess.Popen(
                ["/usr/bin/bash", str(LAUNCHER)],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                preexec_fn=os.setsid,
            )

        row["submitted_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        row["train_run_dir"] = str(run_dir)
        row["eval_run_dir"] = str(eval_dir)
        row["train_job_id"] = ""
        row["eval_job_id"] = ""
        row["local_pid"] = str(proc.pid)
        row["local_gpu"] = str(gpu)
        row["launch_log"] = str(launch_log)
        out_rows.append(row)
        time.sleep(0.2)

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    write_rows(Path(args.out_manifest), out_rows, fieldnames)
    print(f"out_manifest={args.out_manifest}")
    for row in out_rows:
        print(f"{row['exp_id']}: pid={row['local_pid']} gpu={row['local_gpu']} run_dir={row['train_run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

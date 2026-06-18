#!/usr/bin/env python3
"""Submit/resume the two CelebA-HQ 256 300k validation runs.

This is intentionally narrow: it watches only the DDP full-K and strict P4
pipeline runs for the CelebA-HQ 256 external-validation branch.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
RUN_ROOT = PROJECT_DIR / "runs_celebahq256_preflight" / "smoke"
PBS_SCRIPT = PROJECT_DIR / "scripts" / "current" / "pbs_run_ablation_case.pbs"
DATA_DIR = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/data/celeba/celeba-lmdb")
CKPT_RE = re.compile(r"ckpt_step(\d+)\.pt$")


@dataclass(frozen=True)
class RunSpec:
    exp_id: str
    train_mode: str
    pipe_stages: int
    weight_mode: str
    config: Path
    world_size: int = 4
    num_nodes: int = 1
    ppn: int = 4


def run_specs(stamp: str | None) -> list[RunSpec]:
    suffix = "_{stamp}" if stamp else "_*"
    del suffix
    return [
        RunSpec(
            exp_id="celebahq256_ddp_smoke_b64_k100_s300000_seed1",
            train_mode="ddp_fullk",
            pipe_stages=1,
            weight_mode="deep_only",
            config=PROJECT_DIR / "configs" / "celebahq256_ddp_strict.yaml",
        ),
        RunSpec(
            exp_id="celebahq256_pipe_p4_smoke_b64_k100_s300000_seed1",
            train_mode="pipe_strict",
            pipe_stages=4,
            weight_mode="uniform",
            config=PROJECT_DIR / "configs" / "celebahq256_pipeline_strict.yaml",
        ),
    ]


def run_dir_for(spec: RunSpec, stamp: str | None) -> Path:
    if stamp:
        return RUN_ROOT / f"{spec.exp_id}_{stamp}"
    matches = sorted(RUN_ROOT.glob(f"{spec.exp_id}_*"))
    if matches:
        return matches[-1]
    new_stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return RUN_ROOT / f"{spec.exp_id}_{new_stamp}"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(path)


def latest_metric_step(run_dir: Path) -> int | None:
    path = run_dir / "metrics_rank0.csv"
    if not path.exists():
        return None
    last: list[str] | None = None
    with path.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0] != "step":
                last = row
    if not last:
        return None
    try:
        return int(float(last[0]))
    except ValueError:
        return None


def latest_checkpoint(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    best: tuple[int, Path] | None = None
    for path in ckpt_dir.glob("ckpt_step*.pt"):
        match = CKPT_RE.search(path.name)
        if not match:
            continue
        step = int(match.group(1))
        if best is None or step > best[0]:
            best = (step, path)
    return best[1] if best else None


def job_exists(job_id: str) -> bool:
    if not job_id:
        return False
    proc = subprocess.run(
        ["qstat", "-f", job_id],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def write_manifest_row(
    spec: RunSpec,
    run_dir: Path,
    args: argparse.Namespace,
    resume_ckpt: str,
) -> Path:
    row = {
        "K": args.k,
        "batch_size": args.batch_size,
        "benchmark": "celebahq256",
        "config": str(spec.config),
        "data_dir": str(DATA_DIR),
        "exp_id": spec.exp_id,
        "group": args.group,
        "last2_beta": 0.01,
        "lr": args.lr,
        "max_grad_norm": args.max_grad_norm,
        "noise_std": args.noise_std,
        "num_nodes": spec.num_nodes,
        "pipe_stages": spec.pipe_stages,
        "ppn": spec.ppn,
        "resume_ckpt": resume_ckpt,
        "seed": args.seed,
        "step_size": args.step_size,
        "steps": args.steps,
        "story": "CelebA-HQ 256 K100 300k external validation",
        "submitted_at_epoch": int(time.time()),
        "train_mode": spec.train_mode,
        "weight_mode": spec.weight_mode,
        "world_size": spec.world_size,
    }
    path = run_dir / "manifest_row_source.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(row, f, indent=2, sort_keys=True)
    return path


def qsub_vars(payload: dict[str, str]) -> str:
    return ",".join(f"{k}={v}" for k, v in payload.items())


def submit(spec: RunSpec, run_dir: Path, args: argparse.Namespace) -> str:
    ckpt = latest_checkpoint(run_dir)
    resume_ckpt = "latest" if ckpt else ""
    manifest_row = write_manifest_row(spec, run_dir, args, resume_ckpt)
    envs = {
        "PROJECT_DIR": str(PROJECT_DIR),
        "GROUP": args.group,
        "EXP_ID": spec.exp_id,
        "STORY": "CelebA-HQ256 300k watched",
        "TRAIN_MODE": spec.train_mode,
        "WORLD_SIZE": str(spec.world_size),
        "PIPE_STAGES": str(spec.pipe_stages),
        "K": str(args.k),
        "STEPS": str(args.steps),
        "LR": str(args.lr),
        "STEP_SIZE": str(args.step_size),
        "NOISE_STD": str(args.noise_std),
        "WEIGHT_MODE": spec.weight_mode,
        "LAST2_BETA": "0.01",
        "SEED": str(args.seed),
        "NUM_NODES": str(spec.num_nodes),
        "PPN": str(spec.ppn),
        "EVAL_IMAGES": "0",
        "RUN_DIR": str(run_dir),
        "MANIFEST_ROW_PATH": str(manifest_row),
        "SAVE_EVERY": str(args.save_every),
        "VIS_EVERY": str(args.vis_every),
        "BATCH_SIZE": str(args.batch_size),
        "SIGMA_PD": "0.0",
        "LANGEVIN_SIGN": "-1.0",
        "WEIGHT_DECAY": "0.0",
        "MAX_GRAD_NORM": str(args.max_grad_norm),
        "DATA_DIR": str(DATA_DIR),
        "CONFIG": str(spec.config),
        "SYNC_FRESH_INIT": "1",
        "DEBUG_LEVEL": str(args.debug_level),
        "LOG_EVERY": str(args.log_every),
        "RESUME_CKPT": resume_ckpt,
    }
    cmd = [
        "qsub",
        "-A",
        args.account,
        "-q",
        args.queue,
        "-l",
        "select=1:system=polaris",
        "-l",
        f"walltime={args.walltime}",
        "-N",
        args.job_name,
        "-v",
        qsub_vars(envs),
        str(PBS_SCRIPT),
    ]
    if args.dry_run:
        print("[DRY_RUN]", " ".join(cmd))
        return "DRY_RUN"
    proc = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
    return proc.stdout.strip().split()[0]


def append_attempt(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = [
        "submitted_at",
        "exp_id",
        "train_mode",
        "job_id",
        "run_dir",
        "latest_metric_step",
        "latest_checkpoint",
        "queue",
        "walltime",
        "status",
    ]
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", default=os.environ.get("PBS_ACCOUNT", "sbi-fair"))
    parser.add_argument("--queue", default="preemptable")
    parser.add_argument("--walltime", default="72:00:00")
    parser.add_argument("--job-name", default="chq300")
    parser.add_argument("--steps", type=int, default=300000)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--step-size", type=float, default=0.001)
    parser.add_argument("--noise-std", type=float, default=0.005)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--vis-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--debug-level", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--group", default="celebahq256_300k_watch")
    parser.add_argument("--stamp", default="20260613_220359")
    parser.add_argument("--state-json", type=Path, default=RUN_ROOT / "logs" / "celebahq256_300k_watch_state.json")
    parser.add_argument("--attempts-csv", type=Path, default=RUN_ROOT / "summaries" / "celebahq256_300k_resume_attempts.csv")
    parser.add_argument("--max-new", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    state = load_json(args.state_json)
    submitted = 0
    records: dict[str, Any] = state.setdefault("runs", {})
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for spec in run_specs(args.stamp):
        run_dir = run_dir_for(spec, args.stamp)
        latest_step = latest_metric_step(run_dir)
        ckpt = latest_checkpoint(run_dir)
        rec = records.setdefault(spec.exp_id, {})
        current_job = rec.get("job_id", "")

        if latest_step is not None and latest_step >= args.steps - 1:
            rec.update(
                {
                    "status": "completed",
                    "run_dir": str(run_dir),
                    "latest_metric_step": latest_step,
                    "latest_checkpoint": str(ckpt or ""),
                    "updated_at": now,
                }
            )
            print(f"[COMPLETE] {spec.exp_id} step={latest_step}")
            continue

        if job_exists(current_job):
            rec.update(
                {
                    "status": "active",
                    "run_dir": str(run_dir),
                    "latest_metric_step": latest_step,
                    "latest_checkpoint": str(ckpt or ""),
                    "updated_at": now,
                }
            )
            print(f"[ACTIVE] {spec.exp_id} job={current_job} step={latest_step}")
            continue

        if submitted >= args.max_new:
            rec.update(
                {
                    "status": "needs_submit_deferred",
                    "run_dir": str(run_dir),
                    "latest_metric_step": latest_step,
                    "latest_checkpoint": str(ckpt or ""),
                    "updated_at": now,
                }
            )
            print(f"[DEFER] {spec.exp_id} max_new={args.max_new}")
            continue

        job_id = submit(spec, run_dir, args)
        submitted += 1
        rec.update(
            {
                "status": "submitted",
                "job_id": job_id,
                "run_dir": str(run_dir),
                "latest_metric_step": latest_step,
                "latest_checkpoint": str(ckpt or ""),
                "queue": args.queue,
                "walltime": args.walltime,
                "updated_at": now,
            }
        )
        append_attempt(
            args.attempts_csv,
            {
                "submitted_at": now,
                "exp_id": spec.exp_id,
                "train_mode": spec.train_mode,
                "job_id": job_id,
                "run_dir": str(run_dir),
                "latest_metric_step": latest_step,
                "latest_checkpoint": str(ckpt or ""),
                "queue": args.queue,
                "walltime": args.walltime,
                "status": "submitted",
            },
        )
        print(f"[SUBMITTED] {spec.exp_id} job={job_id} step={latest_step} ckpt={ckpt or ''}")

    save_json(args.state_json, state)
    print(f"resume_needed_submitted={submitted}")
    print(f"state_json={args.state_json}")
    print(f"attempts_csv={args.attempts_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

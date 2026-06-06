#!/usr/bin/env python3
"""Submit successors for interrupted ImageNet-32 DRL prime 20k gates.

This is scheduler plumbing only. It does not change the method, objective,
sampler, or model config. It watches logical runs in
experiments/imagenet32_drl_prime20k_submitted.csv and appends a new scheduler
attempt when a logical run is incomplete and has no active PBS job.
"""

import argparse
import csv
import getpass
import json
import os
import re
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
SUBMITTED = PROJECT_DIR / "experiments" / "imagenet32_drl_prime20k_submitted.csv"
ATTEMPTS = PROJECT_DIR / "experiments" / "imagenet32_drl_prime20k_auto_attempts.csv"
PAYLOAD_DIR = PROJECT_DIR / "experiments" / "_submission_payloads" / "imagenet32_drl_prime20k"
PBS_SCRIPT = PROJECT_DIR / "scripts" / "current" / "pbs_run_imagenet32_drl_case.pbs"
RUN_ROOT = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_imagenet32_drl_prime/S2_20k_gate")

ACTIVE_STATES = {"R", "Q", "H", "W", "S", "E", "B"}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def qstat_states():
    user = os.environ.get("USER") or getpass.getuser()
    try:
        out = subprocess.check_output(["qstat", "-u", user]).decode("utf-8", "replace")
    except Exception:
        return {}
    states = {}
    for line in out.splitlines():
        if ".polaris-pbs" not in line:
            continue
        parts = line.split()
        if not parts:
            continue
        jid = parts[0].split(".")[0]
        state = ""
        for tok in reversed(parts):
            if tok in ACTIVE_STATES:
                state = tok
                break
        if state:
            states[jid] = state
    return states


def read_rows(path):
    if not path.exists():
        return [], []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def bool01(value):
    return "1" if str(value).strip().lower() in {"1", "true", "yes", "y"} else "0"


def job_num(job_id):
    return (job_id or "").split(".", 1)[0].strip()


def latest_logged_step(run_dir):
    metrics = run_dir / "metrics_rank0.csv"
    if not metrics.exists():
        return -1
    try:
        last = subprocess.check_output(["tail", "-n", "1", str(metrics)]).decode("utf-8", "replace").strip()
    except Exception:
        return -1
    if not last or last.startswith("step,"):
        return -1
    try:
        return int(float(last.split(",", 1)[0]))
    except Exception:
        return -1


def latest_checkpoint(run_dir):
    candidates = []
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    for path in ckpt_dir.glob("ckpt_step*.pt"):
        match = re.search(r"ckpt_step(\d+)\.pt$", path.name)
        if not match:
            continue
        candidates.append((int(match.group(1)), path))
    for _, path in sorted(candidates, reverse=True):
        if zipfile.is_zipfile(path):
            return path
    return None


def group_logical_rows(rows):
    grouped = {}
    for row in rows:
        if row.get("phase") != "S2_20k_gate":
            continue
        if row.get("submit", "yes").strip().lower() not in {"yes", "true", "1"}:
            continue
        exp_id = row.get("exp_id", "")
        if not exp_id:
            continue
        grouped.setdefault(exp_id, []).append(row)
    return grouped


def logical_status(rows, states):
    target = int(float(rows[-1].get("steps") or 20000))
    active = []
    best_step = -1
    best_ckpt = None
    best_ckpt_step = -1
    latest_row = rows[-1]
    latest_job = job_num(latest_row.get("train_job_id", ""))

    for row in rows:
        jid = job_num(row.get("train_job_id", ""))
        if jid and states.get(jid) in ACTIVE_STATES:
            active.append("%s:%s" % (jid, states[jid]))
        run_dir = Path(row.get("train_run_dir", ""))
        step = latest_logged_step(run_dir)
        if step > best_step:
            best_step = step
        ckpt = latest_checkpoint(run_dir)
        if ckpt is not None:
            match = re.search(r"ckpt_step(\d+)\.pt$", ckpt.name)
            ckpt_step = int(match.group(1)) if match else -1
            if ckpt_step > best_ckpt_step:
                best_ckpt_step = ckpt_step
                best_ckpt = ckpt

    return {
        "target": target,
        "active": active,
        "best_step": best_step,
        "best_ckpt": best_ckpt,
        "latest_row": latest_row,
        "latest_job": latest_job,
    }


def append_csv(path, fieldnames, row):
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def qsub_successor(row, attempt_index, resume_ckpt, dry_run):
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    exp_id = row["exp_id"]
    run_dir = RUN_ROOT / ("%s_auto%d_%s" % (exp_id, attempt_index, ts))
    payload_path = PAYLOAD_DIR / ("%s_auto%d_%s.json" % (exp_id, attempt_index, ts))
    submitted_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    new_row = dict(row)
    new_row["submitted_at"] = submitted_at
    new_row["train_run_dir"] = str(run_dir)
    new_row["train_job_id"] = ""
    new_row["payload_path"] = str(payload_path)
    mode = "resume" if resume_ckpt else "fresh"
    new_row["notes"] = (
        (new_row.get("notes") or "")
        + " Auto watcher %s attempt %d at %s%s."
        % (mode, attempt_index, submitted_at, (" from " + str(resume_ckpt)) if resume_ckpt else "")
    ).strip()

    payload = dict(new_row)
    payload["auto_watcher"] = True
    payload["resume_ckpt"] = str(resume_ckpt or "")

    if not dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)
        with payload_path.open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    env = {
        "PROJECT_DIR": str(PROJECT_DIR),
        "GROUP": row["group"],
        "EXP_ID": exp_id,
        "TRAIN_MODE": row["train_mode"],
        "WORLD_SIZE": row["world_size"],
        "PIPE_STAGES": row["pipe_stages"],
        "K": row["K"],
        "STEPS": row["steps"],
        "LR": row["lr"],
        "LR_WARMUP_STEPS": row["lr_warmup_steps"],
        "STEP_SIZE": row["step_size"],
        "NOISE_STD": row["noise_std"],
        "SAMPLER_PRIME": row["sampler_prime"],
        "SAMPLER_TEMPERATURE": row["sampler_temperature"],
        "ENERGY_LOSS_SCALE": row["energy_loss_scale"],
        "WEIGHT_MODE": row["weight_mode"],
        "LAST2_BETA": row["last2_beta"],
        "SEED": row["seed"],
        "NUM_NODES": row["num_nodes"],
        "PPN": row["ppn"],
        "EVAL_IMAGES": row["eval_images"],
        "RUN_DIR": str(run_dir),
        "MANIFEST_ROW_PATH": str(payload_path),
        "SAVE_EVERY": row["save_every"],
        "VIS_EVERY": row["vis_every"],
        "BATCH_SIZE": row["batch_size"],
        "SIGMA_PD": "0.03",
        "DATA_DIR": row["data_dir"],
        "CONFIG": row["config"],
        "MAX_GRAD_NORM": row["max_grad_norm"],
        "GRAD_CLIP_NORM": row["grad_clip_norm"],
        "DIAGNOSTIC_FIRST_STEPS": row["diagnostic_first_steps"],
        "DIAGNOSTIC_LOG_EVERY": row["diagnostic_log_every"],
        "CRASH_ABS_ENERGY": row["crash_abs_energy"],
        "CRASH_MAX_ABS_CHAIN_UNCLAMPED": row["crash_max_abs_chain_unclamped"],
        "CRASH_GRAD_NORM": row["crash_grad_norm"],
        "SOFT_ABS_ENERGY": row["soft_abs_energy"],
        "SOFT_GRAD_NORM": row["soft_grad_norm"],
        "HARD_GRAD_NORM": row["hard_grad_norm"],
        "CLIP_ACTIVE_FRACTION": row["clip_active_fraction"],
        "SOFT_GUARD_STOP": bool01(row["soft_guard_stop"]),
        "SKIP_OPTIMIZER_UNTIL_FULL_DIAGONAL": bool01(row["skip_optimizer_until_full_diagonal"]),
        "SYNC_FRESH_INIT": "1",
        "DEBUG_LEVEL": "1",
        "LOG_EVERY": row["diagnostic_log_every"],
        "RESUME_CKPT": str(resume_ckpt or ""),
    }
    env_arg = ",".join("%s=%s" % (k, v) for k, v in env.items())
    jobname = "im32p20s01w" if str(row.get("energy_loss_scale")) == "0.1" else "im32p20s1w"
    cmd = [
        "qsub",
        "-q",
        row.get("queue") or "preemptable",
        "-N",
        jobname,
        "-l",
        "select=1:system=polaris",
        "-l",
        "walltime=%s" % (row.get("train_walltime") or "24:00:00"),
        "-o",
        str(run_dir / "pbs.out"),
        "-v",
        env_arg,
        str(PBS_SCRIPT),
    ]

    if dry_run:
        return new_row, "DRY_RUN:" + " ".join(cmd)

    out = subprocess.check_output(cmd, cwd=str(PROJECT_DIR)).decode("utf-8", "replace").strip()
    new_row["train_job_id"] = out
    payload["train_job_id"] = out
    with payload_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return new_row, out


def append_attempt(row):
    fieldnames = [
        "submitted_at",
        "exp_id",
        "train_job_id",
        "train_run_dir",
        "payload_path",
        "notes",
    ]
    append_csv(ATTEMPTS, fieldnames, row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows, fieldnames = read_rows(SUBMITTED)
    if not rows:
        print("submitted_count=0")
        return 0

    states = qstat_states()
    grouped = group_logical_rows(rows)
    submitted_count = 0

    for exp_id in sorted(grouped):
        if submitted_count >= args.max_new:
            break
        group = grouped[exp_id]
        status = logical_status(group, states)
        target = status["target"]
        best_step = status["best_step"]
        if best_step >= target - 1:
            print("%s complete step=%s/%s" % (exp_id, best_step, target))
            continue
        if status["active"]:
            print("%s active=%s step=%s/%s" % (exp_id, ",".join(status["active"]), best_step, target))
            continue

        new_row, job_id = qsub_successor(
            row=status["latest_row"],
            attempt_index=len(group),
            resume_ckpt=status["best_ckpt"],
            dry_run=args.dry_run,
        )
        mode = "resume" if status["best_ckpt"] else "fresh"
        print(
            "%s submitted mode=%s step=%s/%s ckpt=%s job=%s"
            % (exp_id, mode, best_step, target, status["best_ckpt"] or "", job_id)
        )
        if not args.dry_run:
            append_csv(SUBMITTED, fieldnames, new_row)
            append_attempt(
                {
                    "submitted_at": new_row["submitted_at"],
                    "exp_id": exp_id,
                    "train_job_id": new_row["train_job_id"],
                    "train_run_dir": new_row["train_run_dir"],
                    "payload_path": new_row["payload_path"],
                    "notes": new_row["notes"],
                }
            )
        submitted_count += 1

    print("submitted_count=%d" % submitted_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

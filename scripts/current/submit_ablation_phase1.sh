#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python3 - "$@" <<'PY'
import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def truthy(text: str) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes", "y"}


def load_rows(path: Path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def env_payload(row, run_dir: str, eval_dir: str, manifest_row_path: str, project_dir: str):
    payload = {
        "PROJECT_DIR": project_dir,
        "GROUP": row["group"],
        "EXP_ID": row["exp_id"],
        "STORY": row.get("story", ""),
        "TRAIN_MODE": row["train_mode"],
        "WORLD_SIZE": row["world_size"],
        "PIPE_STAGES": row["pipe_stages"],
        "K": row["K"],
        "STEPS": row["steps"],
        "LR": row["lr"],
        "STEP_SIZE": row["step_size"],
        "WEIGHT_MODE": row["weight_mode"],
        "LAST2_BETA": row.get("last2_beta", "0.01") or "0.01",
        "SEED": row["seed"],
        "NUM_NODES": row["num_nodes"],
        "PPN": row["ppn"],
        "EVAL_IMAGES": row["eval_images"],
        "NOTES": row.get("notes", ""),
        "RUN_DIR": run_dir,
        "EVAL_DIR": eval_dir,
        "MANIFEST_ROW_PATH": manifest_row_path,
        "SAVE_EVERY": row.get("save_every", "5000") or "5000",
        "VIS_EVERY": row.get("vis_every", "1000") or "1000",
        "BATCH_SIZE": row.get("batch_size", "64") or "64",
        "SIGMA_PD": row.get("sigma_pd", "0.03") or "0.03",
        "NOISE_STD": row.get("noise_std", "0.01") or "0.01",
        "LANGEVIN_SIGN": row.get("langevin_sign", "1.0") or "1.0",
        "WEIGHT_DECAY": row.get("weight_decay", "0.0") or "0.0",
        "MAX_GRAD_NORM": row.get("max_grad_norm", "0.0") or "0.0",
        "DATA_DIR": row.get("data_dir", "/eagle/lc-mpi/Zhiqing/ebm/data/cifar10") or "/eagle/lc-mpi/Zhiqing/ebm/data/cifar10",
        "SYNC_FRESH_INIT": row.get("sync_fresh_init", "1") or "1",
        "DEBUG_LEVEL": row.get("debug_level", "1") or "1",
        "LOG_EVERY": row.get("log_every", "10") or "10",
        "EVAL_BATCH": row.get("eval_batch", "256") or "256",
        "K_EVAL": row.get("k_eval", "100") or "100",
        "EVAL_SEED": row.get("eval_seed", row["seed"]) or row["seed"],
    }
    return payload


def qsub_command(script: Path, queue: str, walltime: str, num_nodes: str, envs: dict, job_name: str, depend: str = ""):
    cmd = [
        "qsub",
        "-q",
        queue,
        "-N",
        job_name,
        "-l",
        f"select={num_nodes}:system=polaris",
        "-l",
        f"walltime={walltime}",
    ]
    if depend:
        cmd.extend(["-W", f"depend=afterok:{depend}"])
    env_text = ",".join(f"{k}={v}" for k, v in envs.items())
    cmd.extend(["-v", env_text, str(script)])
    return cmd


def run_cmd(cmd, dry_run: bool):
    if dry_run:
        return ""
    out = subprocess.check_output(cmd, universal_newlines=True)
    return out.strip().split()[0]


parser = argparse.ArgumentParser("Submit Phase 1 ablation cases")
parser.add_argument("--manifest", default=str(Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/experiments/ablation_manifest_phase1.csv")))
parser.add_argument("--submitted-manifest", default=str(Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/experiments/ablation_manifest_phase1_submitted.csv")))
parser.add_argument("--eval-mode", choices=["pbs", "local"], default="local")
parser.add_argument("--runs-root", default="/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation")
parser.add_argument("--payload-subdir", default="phase1")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

project_dir = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
runs_root = Path(args.runs_root)
payload_root = project_dir / "experiments" / "_submission_payloads" / args.payload_subdir
payload_root.mkdir(parents=True, exist_ok=True)

manifest_path = Path(args.manifest)
submitted_path = Path(args.submitted_manifest)
rows = load_rows(manifest_path)
existing_rows = load_rows(submitted_path) if submitted_path.exists() else []
existing_by_exp_id = {row["exp_id"]: dict(row) for row in existing_rows}
submitted_rows = []
group_jobs = defaultdict(list)
submitted_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
submit_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
extra_fields = ["submitted_at", "train_run_dir", "eval_run_dir", "train_job_id", "eval_job_id"]

for idx, row in enumerate(rows, start=1):
    row = dict(row)
    submit_flag = truthy(row.get("submit", "1"))
    payload_path = payload_root / f"{idx:02d}_{row['exp_id']}.json"
    existing = dict(existing_by_exp_id.get(row["exp_id"], {}))
    out_row = dict(row)
    for name in extra_fields:
        out_row[name] = existing.get(name, "")

    run_dir = out_row.get("train_run_dir", "").strip() or row.get("reference_run_dir", "").strip()
    eval_dir = out_row.get("eval_run_dir", "").strip() or row.get("reference_eval_dir", "").strip()
    if submit_flag and not run_dir:
        run_dir = str(runs_root / row["group"] / f"{row['exp_id']}_{submit_tag}")
    if submit_flag and not eval_dir:
        eval_dir = str(Path(run_dir) / "eval")

    payload = dict(row)
    payload["submitted_at"] = out_row.get("submitted_at", "") or submitted_at
    payload["train_run_dir"] = run_dir
    payload["eval_run_dir"] = eval_dir
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    out_row["train_run_dir"] = run_dir
    out_row["eval_run_dir"] = eval_dir

    if not submit_flag:
        submitted_rows.append(out_row)
        continue

    queue = row.get("queue", "preemptable") or "preemptable"
    queue = row.get("queue", "preemptable") or "preemptable"
    train_env = env_payload(
        row=row,
        run_dir=run_dir,
        eval_dir=eval_dir,
        manifest_row_path=str(payload_path),
        project_dir=str(project_dir),
    )
    eval_env = dict(train_env)
    train_job_name = f"abtr{idx:02d}"
    eval_job_name = f"abev{idx:02d}"
    train_job_id = out_row.get("train_job_id", "").strip()
    if not train_job_id:
        train_cmd = qsub_command(
            script=project_dir / "scripts" / "current" / "pbs_run_ablation_case.pbs",
            queue=queue,
            walltime=row.get("train_walltime", "04:00:00") or "04:00:00",
            num_nodes=row["num_nodes"],
            envs=train_env,
            job_name=train_job_name,
        )
        train_job_id = run_cmd(train_cmd, dry_run=args.dry_run)
        out_row["train_job_id"] = train_job_id if not args.dry_run else f"DRYRUN-TRAIN-{idx:02d}"
        out_row["submitted_at"] = submitted_at

    eval_job_id = out_row.get("eval_job_id", "").strip()
    if args.eval_mode == "pbs" and not eval_job_id:
        eval_cmd = qsub_command(
            script=project_dir / "scripts" / "current" / "pbs_eval_ablation_case.pbs",
            queue=queue,
            walltime=row.get("eval_walltime", "02:00:00") or "02:00:00",
            num_nodes="1",
            envs=eval_env,
            job_name=eval_job_name,
            depend=(train_job_id if not args.dry_run else "DRYRUN-TRAIN"),
        )
        try:
            eval_job_id = run_cmd(eval_cmd, dry_run=args.dry_run)
        except Exception:
            submitted_rows.append(out_row)
            fieldnames = list(submitted_rows[0].keys()) if submitted_rows else list(out_row.keys())
            remaining_rows = []
            for tail_idx in range(idx, len(rows)):
                tail_row = dict(rows[tail_idx])
                for name in extra_fields:
                    tail_row[name] = existing_by_exp_id.get(tail_row["exp_id"], {}).get(name, "")
                remaining_rows.append(tail_row)
            write_rows(submitted_path, submitted_rows + remaining_rows, fieldnames)
            raise
        out_row["eval_job_id"] = eval_job_id if not args.dry_run else f"DRYRUN-EVAL-{idx:02d}"
        out_row["submitted_at"] = submitted_at

    submitted_rows.append(out_row)
    group_jobs[row["group"]].append((row["exp_id"], out_row["train_job_id"], out_row["eval_job_id"]))

fieldnames = list(submitted_rows[0].keys()) if submitted_rows else []
write_rows(submitted_path, submitted_rows, fieldnames)

print(f"manifest={manifest_path}")
print(f"submitted_manifest={submitted_path}")
print(f"dry_run={int(args.dry_run)}")
print(f"train_jobs={sum(1 for r in submitted_rows if r.get('train_job_id'))}")
print(f"eval_jobs={sum(1 for r in submitted_rows if r.get('eval_job_id'))}")
for group in sorted(group_jobs):
    print(f"[{group}]")
    for exp_id, train_job_id, eval_job_id in group_jobs[group]:
        print(f"  {exp_id}: train={train_job_id} eval={eval_job_id}")
PY

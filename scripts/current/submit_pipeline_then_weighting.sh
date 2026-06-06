#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY=/home/kevienzzq/.conda/envs/llm-env/bin/python
REPO_PYTHONPATH="/eagle/lc-mpi/Zhiqing:/lus/eagle/projects/lc-mpi/Zhiqing${PYTHONPATH:+:${PYTHONPATH}}"

PYTHONPATH="${REPO_PYTHONPATH}" "${PY}" - "$@" <<'PY'
import argparse
import csv
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def truthy(text: str) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes", "y"}


def clean_job_id(value: str) -> str:
    text = (value or "").strip()
    if text.startswith("DRYRUN-"):
        return ""
    return text


def live_job_id(value: str) -> str:
    job_id = clean_job_id(value)
    if not job_id:
        return ""
    try:
        out = subprocess.check_output(["qstat", "-xf", job_id], universal_newlines=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        return ""
    state = ""
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("job_state ="):
            state = line.split("=", 1)[1].strip()
            break
    if state in {"Q", "R", "H", "B", "W"}:
        return job_id
    return ""


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
    return {
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
    try:
        out = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        output = exc.output or ""
        if "per-project limit" in output:
            raise RuntimeError("QUEUE_PROJECT_LIMIT_REACHED")
        raise
    return out.strip().split()[0]


parser = argparse.ArgumentParser("Submit pipeline-then-weighting experiments")
parser.add_argument("--manifest", default=str(Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/experiments/pipeline_then_weighting_manifest.csv")))
parser.add_argument("--submitted-manifest", default=str(Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/experiments/pipeline_then_weighting_manifest_submitted.csv")))
parser.add_argument("--runs-root", default="/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_pipeline_then_weighting")
parser.add_argument("--payload-subdir", default="pipeline_then_weighting")
parser.add_argument("--eval-mode", choices=("local", "pbs"), default="local")
parser.add_argument("--max-live-train-jobs", type=int, default=0)
parser.add_argument("--force-resubmit", action="store_true")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

project_dir = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm")
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
current_live_train_jobs = 0
queue_project_limit_reached = False

extra_fields = ["submitted_at", "train_run_dir", "eval_run_dir", "train_job_id", "eval_job_id"]

for idx, row in enumerate(rows, start=1):
    row = dict(row)
    submit_flag = truthy(row.get("submit", "no"))
    payload_path = payload_root / f"{idx:02d}_{row['exp_id']}.json"
    existing = dict(existing_by_exp_id.get(row["exp_id"], {}))
    out_row = dict(row)
    for name in extra_fields:
        out_row[name] = existing.get(name, "")

    existing_train_job = live_job_id(out_row.get("train_job_id", ""))
    existing_eval_job = "" if args.eval_mode == "local" else live_job_id(out_row.get("eval_job_id", ""))
    historical_train_job = clean_job_id(out_row.get("train_job_id", ""))
    run_dir = out_row.get("train_run_dir", "").strip()
    eval_dir = out_row.get("eval_run_dir", "").strip()
    if existing_train_job:
        current_live_train_jobs += 1
    if not existing_train_job:
        run_dir = ""
        eval_dir = ""
    if submit_flag and not run_dir:
        run_dir = str(runs_root / row["phase"] / f"{row['exp_id']}_{submit_tag}")
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
        out_row["train_job_id"] = ""
        out_row["eval_job_id"] = ""
        submitted_rows.append(out_row)
        continue

    if historical_train_job and not existing_train_job and not args.force_resubmit:
        submitted_rows.append(out_row)
        continue

    if (
        args.max_live_train_jobs > 0
        and not existing_train_job
        and current_live_train_jobs >= args.max_live_train_jobs
    ):
        submitted_rows.append(out_row)
        continue

    if queue_project_limit_reached and not existing_train_job:
        submitted_rows.append(out_row)
        continue

    queue = row.get("queue", "preemptable") or "preemptable"
    train_env = env_payload(
        row=row,
        run_dir=run_dir,
        eval_dir=eval_dir,
        manifest_row_path=str(payload_path),
        project_dir=str(project_dir),
    )
    eval_env = dict(train_env)
    train_job_id = existing_train_job
    if not train_job_id:
        try:
            train_cmd = qsub_command(
                script=project_dir / "scripts" / "current" / "pbs_run_pipeline_then_weighting_case.pbs",
                queue=queue,
                walltime=row.get("train_walltime", "04:00:00") or "04:00:00",
                num_nodes=row["num_nodes"],
                envs=train_env,
                job_name=f"ptwtr{idx:02d}",
            )
            train_job_id = run_cmd(train_cmd, dry_run=args.dry_run)
            out_row["submitted_at"] = submitted_at
            current_live_train_jobs += 1
        except RuntimeError as exc:
            if str(exc) != "QUEUE_PROJECT_LIMIT_REACHED":
                raise
            queue_project_limit_reached = True
            submitted_rows.append(out_row)
            continue

    if args.eval_mode == "pbs":
        eval_job_id = existing_eval_job
        if not eval_job_id:
            eval_cmd = qsub_command(
                script=project_dir / "scripts" / "current" / "pbs_eval_pipeline_then_weighting_case.pbs",
                queue=queue,
                walltime=row.get("eval_walltime", "02:00:00") or "02:00:00",
                num_nodes="1",
                envs=eval_env,
                job_name=f"ptwev{idx:02d}",
                depend=(train_job_id if not args.dry_run else "DRYRUN-TRAIN"),
            )
            eval_job_id = run_cmd(eval_cmd, dry_run=args.dry_run)
            out_row["submitted_at"] = submitted_at
    else:
        eval_job_id = ""

    out_row["train_job_id"] = train_job_id if not args.dry_run else f"DRYRUN-TRAIN-{idx:02d}"
    if args.eval_mode == "pbs":
        out_row["eval_job_id"] = eval_job_id if not args.dry_run else f"DRYRUN-EVAL-{idx:02d}"
    else:
        out_row["eval_job_id"] = ""
    submitted_rows.append(out_row)
    group_jobs[row["group"]].append((row["exp_id"], out_row["train_job_id"], out_row["eval_job_id"]))

fieldnames = list(submitted_rows[0].keys()) if submitted_rows else []
if not args.dry_run:
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

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
from datetime import datetime, timezone
from pathlib import Path

from polaris_ebm.scripts.current.imagenet32_phasea_manifest import build_phasea_rows, write_manifest


def load_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def env_payload(row, run_dir: str, manifest_row_path: str, project_dir: str):
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
        "MANIFEST_ROW_PATH": manifest_row_path,
        "SAVE_EVERY": row.get("save_every", "1000") or "1000",
        "VIS_EVERY": row.get("vis_every", "1000") or "1000",
        "BATCH_SIZE": row.get("batch_size", "64") or "64",
        "SIGMA_PD": row.get("sigma_pd", "0.03") or "0.03",
        "NOISE_STD": row.get("noise_std", "0.01") or "0.01",
        "LANGEVIN_SIGN": row.get("langevin_sign", "1.0") or "1.0",
        "WEIGHT_DECAY": row.get("weight_decay", "0.0") or "0.0",
        "MAX_GRAD_NORM": row.get("max_grad_norm", "0.0") or "0.0",
        "DATA_DIR": row["data_dir"],
        "CONFIG": row["config"],
        "SYNC_FRESH_INIT": row.get("sync_fresh_init", "1") or "1",
        "DEBUG_LEVEL": row.get("debug_level", "1") or "1",
        "LOG_EVERY": row.get("log_every", "50") or "50",
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
    out = subprocess.check_output(cmd, universal_newlines=True)
    return out.strip().split()[0]


def clean_job_id(value: str) -> str:
    text = (value or "").strip()
    if text.startswith("DRYRUN-"):
        return ""
    return text


parser = argparse.ArgumentParser("Submit ImageNet-32 Phase A training jobs")
parser.add_argument("--manifest", default=str(Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/experiments/imagenet32_phasea_manifest.csv")))
parser.add_argument("--submitted-manifest", default=str(Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/experiments/imagenet32_phasea_submitted.csv")))
parser.add_argument("--runs-root", default="/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_imagenet32_phasea")
parser.add_argument("--payload-subdir", default="imagenet32_phasea")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

project_dir = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
manifest_path = Path(args.manifest)
submitted_path = Path(args.submitted_manifest)
runs_root = Path(args.runs_root)
payload_root = project_dir / "experiments" / "_submission_payloads" / args.payload_subdir
payload_root.mkdir(parents=True, exist_ok=True)

rows = build_phasea_rows()
write_manifest(manifest_path, rows)

existing_rows = load_rows(submitted_path)
existing_by_exp_id = {row["exp_id"]: dict(row) for row in existing_rows}

submitted_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
submit_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
materialize_job_id = ""
if existing_rows:
    materialize_job_id = clean_job_id(existing_rows[0].get("materialize_job_id", ""))

if not materialize_job_id:
    materialize_env = {
        "PROJECT_DIR": str(project_dir),
        "OUT_ROOT": "/eagle/lc-mpi/Zhiqing/polaris_ebm/data/imagenet32",
        "CACHE_DIR": "/eagle/lc-mpi/Zhiqing/.cache/huggingface/datasets",
        "STREAMING": "0",
        "TRAIN_MAX_EXAMPLES": "0",
        "VAL_MAX_EXAMPLES": "0",
    }
    materialize_cmd = qsub_command(
        script=project_dir / "scripts" / "current" / "pbs_materialize_imagenet32.pbs",
        queue="preemptable",
        walltime="08:00:00",
        num_nodes="1",
        envs=materialize_env,
        job_name="imnet32prep",
    )
    materialize_job_id = run_cmd(materialize_cmd, dry_run=args.dry_run)

submitted_rows = []
for idx, row in enumerate(rows, start=1):
    row = dict(row)
    existing = dict(existing_by_exp_id.get(row["exp_id"], {}))
    existing_job_id = clean_job_id(existing.get("train_job_id", ""))
    run_dir = str(runs_root / row["group"] / f"{row['exp_id']}_{submit_tag}")
    if existing_job_id:
        run_dir = existing.get("train_run_dir", "").strip() or run_dir
    payload_path = payload_root / f"{idx:02d}_{row['exp_id']}.json"
    payload = dict(row)
    payload["submitted_at"] = submitted_at
    payload["train_run_dir"] = run_dir
    payload["materialize_job_id"] = materialize_job_id if not args.dry_run else "DRYRUN-MATERIALIZE"
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    out_row = dict(row)
    out_row["submitted_at"] = existing.get("submitted_at", "") or submitted_at
    out_row["train_run_dir"] = run_dir
    out_row["materialize_job_id"] = materialize_job_id if not args.dry_run else "DRYRUN-MATERIALIZE"
    train_job_id = existing_job_id
    if not train_job_id:
        envs = env_payload(row, run_dir=run_dir, manifest_row_path=str(payload_path), project_dir=str(project_dir))
        train_cmd = qsub_command(
            script=project_dir / "scripts" / "current" / "pbs_run_ablation_case.pbs",
            queue=row.get("queue", "preemptable") or "preemptable",
            walltime=row.get("train_walltime", "04:00:00") or "04:00:00",
            num_nodes=row["num_nodes"],
            envs=envs,
            job_name=f"imtr{idx:02d}",
            depend=(materialize_job_id if not args.dry_run else "DRYRUN-MATERIALIZE"),
        )
        train_job_id = run_cmd(train_cmd, dry_run=args.dry_run)
    out_row["train_job_id"] = train_job_id if not args.dry_run else f"DRYRUN-TRAIN-{idx:02d}"
    submitted_rows.append(out_row)

fieldnames = list(submitted_rows[0].keys()) if submitted_rows else []
write_rows(submitted_path, submitted_rows, fieldnames)

print(f"manifest={manifest_path}")
print(f"submitted_manifest={submitted_path}")
print(f"dry_run={int(args.dry_run)}")
print(f"materialize_job_id={materialize_job_id if not args.dry_run else 'DRYRUN-MATERIALIZE'}")
print(f"train_jobs={sum(1 for r in submitted_rows if r.get('train_job_id'))}")
for row in submitted_rows:
    print(f"{row['exp_id']}: train={row['train_job_id']}")
PY

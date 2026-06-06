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

from polaris_ebm.scripts.current.imagenet32_drl_manifest import (
    FIELDS,
    build_drl_rows,
    imagenet32_materialized_success,
    write_manifest,
)
from polaris_ebm.scripts.current.imagenet32_drl_stability_manifest import (
    FIELDS as STABILITY_FIELDS,
    build_stability_rows,
    write_manifest as write_stability_manifest,
)


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


def qsub_command(script: Path, row: dict, envs: dict, job_name: str):
    cmd = [
        "qsub",
        "-q",
        row.get("queue", "preemptable") or "preemptable",
        "-N",
        job_name,
        "-l",
        f"select={row['num_nodes']}:system=polaris",
        "-l",
        f"walltime={row.get('train_walltime', '04:00:00') or '04:00:00'}",
    ]
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


def env_payload(row, run_dir: str, manifest_row_path: str, project_dir: str):
    return {
        "PROJECT_DIR": project_dir,
        "GROUP": row["group"],
        "EXP_ID": row["exp_id"],
        "STORY": "imagenet32_drl_backbone_external_validation",
        "TRAIN_MODE": row["train_mode"],
        "WORLD_SIZE": row["world_size"],
        "PIPE_STAGES": row["pipe_stages"],
        "K": row["K"],
        "STEPS": row["steps"],
        "LR": row["lr"],
        "LR_WARMUP_STEPS": row.get("lr_warmup_steps", "0") or "0",
        "STEP_SIZE": row["step_size"],
        "NOISE_STD": row.get("noise_std", "0.01") or "0.01",
        "SAMPLER_PRIME": row.get("sampler_prime", "0.0") or "0.0",
        "SAMPLER_TEMPERATURE": row.get("sampler_temperature", "1.0") or "1.0",
        "ENERGY_LOSS_SCALE": row.get("energy_loss_scale", "1.0") or "1.0",
        "WEIGHT_MODE": row["weight_mode"],
        "LAST2_BETA": row.get("last2_beta", "0.01") or "0.01",
        "SEED": row["seed"],
        "NUM_NODES": row["num_nodes"],
        "PPN": row["ppn"],
        "EVAL_IMAGES": row["eval_images"],
        "NOTES": "",
        "RUN_DIR": run_dir,
        "MANIFEST_ROW_PATH": manifest_row_path,
        "SAVE_EVERY": row.get("save_every", "5000") or "5000",
        "VIS_EVERY": row.get("vis_every", "5000") or "5000",
        "BATCH_SIZE": row.get("batch_size", "128") or "128",
        "DATA_DIR": row["data_dir"],
        "CONFIG": row["config"],
        "DEBUG_LEVEL": "1",
        "LOG_EVERY": row.get("diagnostic_log_every", "50") or "50",
        "MAX_GRAD_NORM": row.get("max_grad_norm", "0.0") or "0.0",
        "GRAD_CLIP_NORM": row.get("grad_clip_norm", "") or "",
        "DIAGNOSTIC_FIRST_STEPS": row.get("diagnostic_first_steps", "200") or "200",
        "DIAGNOSTIC_LOG_EVERY": row.get("diagnostic_log_every", "50") or "50",
        "CRASH_ABS_ENERGY": row.get("crash_abs_energy", "1000.0") or "1000.0",
        "CRASH_MAX_ABS_CHAIN_UNCLAMPED": row.get("crash_max_abs_chain_unclamped", "2.5") or "2.5",
        "CRASH_GRAD_NORM": row.get("crash_grad_norm", "10000.0") or "10000.0",
        "SOFT_ABS_ENERGY": row.get("soft_abs_energy", "10.0") or "10.0",
        "SOFT_GRAD_NORM": row.get("soft_grad_norm", "100.0") or "100.0",
        "HARD_GRAD_NORM": row.get("hard_grad_norm", "500.0") or "500.0",
        "CLIP_ACTIVE_FRACTION": row.get("clip_active_fraction", "0.25") or "0.25",
        "SOFT_GUARD_STOP": "1" if (row.get("soft_guard_stop", "false").lower() == "true") else "0",
        "SKIP_OPTIMIZER_UNTIL_FULL_DIAGONAL": "1"
        if (row.get("skip_optimizer_until_full_diagonal", "false").lower() == "true")
        else "0",
    }


parser = argparse.ArgumentParser("Submit ImageNet-32 DRL-backbone jobs")
parser.add_argument("--manifest", default="/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/experiments/imagenet32_drl_manifest.csv")
parser.add_argument("--submitted-manifest", default="/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/experiments/imagenet32_drl_submitted.csv")
parser.add_argument("--runs-root", default="/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_imagenet32_drl")
parser.add_argument("--payload-subdir", default="imagenet32_drl")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument(
    "--stability",
    action="store_true",
    help="use the short stability diagnostic manifest instead of the long validation manifest",
)
args = parser.parse_args()

project_dir = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
data_root = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/data/imagenet32")
if not imagenet32_materialized_success(data_root):
    raise SystemExit(f"ImageNet-32 materialized dataset is missing split success files under: {data_root}")

manifest_path = Path(args.manifest)
submitted_path = Path(args.submitted_manifest)
runs_root = Path(args.runs_root)
payload_root = project_dir / "experiments" / "_submission_payloads" / args.payload_subdir
payload_root.mkdir(parents=True, exist_ok=True)

if args.stability:
    rows = build_stability_rows()
    write_stability_manifest(manifest_path, rows)
    field_base = STABILITY_FIELDS
else:
    rows = build_drl_rows()
    write_manifest(manifest_path, rows)
    field_base = FIELDS
rows_to_submit = [row for row in rows if row.get("submit", "").lower() == "yes"]
existing_rows = load_rows(submitted_path)
existing_by_exp_id = {row["exp_id"]: dict(row) for row in existing_rows}

submitted_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
submit_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
submitted_rows = []
for idx, row in enumerate(rows_to_submit, start=1):
    row = dict(row)
    existing = dict(existing_by_exp_id.get(row["exp_id"], {}))
    existing_job_id = clean_job_id(existing.get("train_job_id", ""))
    run_dir = existing.get("train_run_dir", "").strip() or str(runs_root / row["phase"] / f"{row['exp_id']}_{submit_tag}")
    payload_path = payload_root / f"{idx:02d}_{row['exp_id']}.json"
    payload = dict(row)
    payload["submitted_at"] = submitted_at
    payload["train_run_dir"] = run_dir
    payload_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    train_job_id = existing_job_id
    if not train_job_id:
        envs = env_payload(row, run_dir=run_dir, manifest_row_path=str(payload_path), project_dir=str(project_dir))
        cmd = qsub_command(
            script=project_dir / "scripts" / "current" / "pbs_run_imagenet32_drl_case.pbs",
            row=row,
            envs=envs,
            job_name=f"im32drl{idx:02d}",
        )
        train_job_id = run_cmd(cmd, dry_run=args.dry_run)

    out_row = dict(row)
    out_row["submitted_at"] = existing.get("submitted_at", "") or submitted_at
    out_row["train_run_dir"] = run_dir
    out_row["train_job_id"] = train_job_id if not args.dry_run else f"DRYRUN-TRAIN-{idx:02d}"
    submitted_rows.append(out_row)

fieldnames = field_base + ["submitted_at", "train_run_dir", "train_job_id"]
write_rows(submitted_path, submitted_rows, fieldnames)

print(f"manifest={manifest_path}")
print(f"submitted_manifest={submitted_path}")
print(f"dry_run={int(args.dry_run)}")
print(f"submit_yes_rows={len(rows_to_submit)}")
for row in submitted_rows:
    print(f"{row['exp_id']}: train={row['train_job_id']} run_dir={row['train_run_dir']}")
PY

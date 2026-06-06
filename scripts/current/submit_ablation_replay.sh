#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHONPATH="${PROJECT_DIR}/scripts/current:/eagle/lc-mpi/Zhiqing:${PYTHONPATH:-}" python3 - "$@" <<'PY'
import argparse
import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    from ablation_replay_common import load_rows, trajectory_progress_summary, write_rows
except Exception:
    from polaris_ebm.scripts.current.ablation_replay_common import load_rows, trajectory_progress_summary, write_rows


def qsub_job(script: Path, row: dict, dry_run: bool) -> str:
    env_text = ",".join(
        [
            f"PROJECT_DIR=/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm",
            f"RUN_DIR={row['run_dir']}",
            f"OUT_DIR={row['trajectory_dir']}",
            f"LABEL={row['exp_id']}",
            "STEP_INTERVAL=5000",
            "EVAL_IMAGES=5000",
            "EVAL_BATCH=256",
            "K_EVAL=100",
            f"STEP_SIZE={row.get('step_size', '1.0')}",
            "NOISE_STD=0.01",
            "LANGEVIN_SIGN=1.0",
            f"EVAL_SEED={row.get('seed', '1')}",
            "DEVICE=cuda",
            "SKIP_EXISTING=1",
        ]
    )
    cmd = [
        "qsub",
        "-q",
        row.get("replay_queue", "capacity"),
        "-N",
        ("rpl_" + row["exp_id"])[:15],
        "-l",
        "select=1:system=polaris",
        "-l",
        f"walltime={row.get('replay_walltime', '06:00:00')}",
        "-v",
        env_text,
        str(script),
    ]
    if dry_run:
        return "DRYRUN-" + row["exp_id"]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)
    return out.strip().split()[0]


parser = argparse.ArgumentParser("Submit missing ablation replay jobs")
parser.add_argument("--manifest", default="/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/experiments/ablation_replay_manifest.csv")
parser.add_argument("--submitted-manifest", default="/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/experiments/ablation_replay_submitted.csv")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

project_dir = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
rows = load_rows(Path(args.manifest))
existing_rows = load_rows(Path(args.submitted_manifest)) if Path(args.submitted_manifest).exists() else []
existing_by_exp = {row["exp_id"]: row for row in existing_rows}
submitted_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

out_rows = []
for row in rows:
    row = dict(row)
    existing = existing_by_exp.get(row["exp_id"], {})
    row["replay_submit_error"] = ""
    if existing:
        row["replay_job_id"] = existing.get("replay_job_id", row.get("replay_job_id", ""))
        row["submitted_at"] = existing.get("submitted_at", "")
        row["replay_submit_error"] = existing.get("replay_submit_error", "")
    if row.get("replay_job_id", "").startswith("DRYRUN-"):
        row["replay_job_id"] = ""
        row["submitted_at"] = ""
        row["replay_submit_error"] = ""

    progress = trajectory_progress_summary(Path(row["trajectory_dir"]), int(row["expected_points"]))
    row["available_points"] = progress["available_points"]
    row["replay_status"] = progress["status"]

    if row.get("replay_mode") == "qsub" and progress["status"] == "missing" and not row.get("replay_job_id", "").strip():
        try:
            job_id = qsub_job(project_dir / "scripts" / "current" / "pbs_eval_fid_trajectory.pbs", row, args.dry_run)
            row["replay_job_id"] = job_id
            row["submitted_at"] = submitted_at
            row["replay_submit_error"] = ""
        except subprocess.CalledProcessError as exc:
            row["replay_submit_error"] = (exc.output or str(exc)).strip().replace("\n", " | ")

    out_rows.append(row)

fieldnames = list(out_rows[0].keys()) if out_rows else []
write_rows(Path(args.submitted_manifest), out_rows, fieldnames)
print(f"[DONE] wrote {len(out_rows)} rows to {args.submitted_manifest}")
print(f"submitted_jobs={sum(1 for row in out_rows if row.get('replay_job_id'))}")
PY

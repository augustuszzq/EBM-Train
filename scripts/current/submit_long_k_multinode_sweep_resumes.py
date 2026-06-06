#!/usr/bin/env python3
"""Submit resume successors for unfinished multi-node long-K sweep runs."""

import argparse
import csv
import re
import shutil
import subprocess
import time
from pathlib import Path


DEFAULT_BUNDLE = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_long_k_multinode_sweep")


ATTEMPT_FIELDS = [
    "exp_id",
    "attempt_role",
    "job_id",
    "previous_job_id",
    "pbs_path",
    "train_run_dir",
    "pbs_output_log",
    "submitted_at",
    "status_hint",
]

RESUME_FIELDS = [
    "old_job_id",
    "new_job_id",
    "exp_id",
    "latest_step_before_resume",
    "latest_checkpoint_before_resume",
    "pbs_path",
    "train_run_dir",
    "pbs_output_log",
    "preserved_old_pbs_output",
    "submitted_at",
    "status",
    "error",
]


def read_csv(path):
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def append_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def qstat_states(user):
    try:
        output = subprocess.check_output(["qstat", "-u", user], universal_newlines=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        return None, exc.output
    states = {}
    for line in output.splitlines():
        match = re.match(
            r"^(\d+)\.\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\d+\s+\d+\s+\S+\s+\S+\s+([RQUEH])\s+",
            line,
        )
        if match:
            states[match.group(1)] = match.group(2)
    return states, output


def latest_step(run_dir):
    path = Path(run_dir) / "metrics_rank0.csv"
    step = -1
    if not path.exists():
        return step
    with path.open(errors="ignore") as f:
        for line in f:
            if not line.strip() or line.startswith("step,"):
                continue
            try:
                step = int(float(line.split(",", 1)[0]))
            except Exception:
                pass
    return step


def latest_checkpoint(run_dir):
    ckpts = []
    for path in (Path(run_dir) / "checkpoints").glob("ckpt_step*.pt"):
        match = re.search(r"ckpt_step(\d+)\.pt$", path.name)
        if match:
            ckpts.append((int(match.group(1)), str(path)))
    if not ckpts:
        return ""
    return max(ckpts)[1]


def normalize_attempts(bundle_root, submitted_rows):
    attempts_path = bundle_root / "summaries" / "multinode_k_sweep_scheduler_attempts.csv"
    attempts = read_csv(attempts_path)
    if attempts:
        return attempts
    attempts = []
    for row in submitted_rows:
        attempts.append(
            {
                "exp_id": row["exp_id"],
                "attempt_role": "original",
                "job_id": row["train_job_id"],
                "previous_job_id": "",
                "pbs_path": row["pbs_path"],
                "train_run_dir": row["train_run_dir"],
                "pbs_output_log": row["pbs_output_log"],
                "submitted_at": row.get("submitted_at", ""),
                "status_hint": row.get("status", "submitted"),
            }
        )
    write_csv(attempts_path, attempts, ATTEMPT_FIELDS)
    return attempts


def latest_attempt_by_exp(attempts):
    latest = {}
    for row in attempts:
        latest[row["exp_id"]] = row
    return latest


def main():
    parser = argparse.ArgumentParser("Submit resume successors for multi-node long-K sweep")
    parser.add_argument("--bundle-root", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--user", default="kevienzzq")
    parser.add_argument("--max-new", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bundle_root = Path(args.bundle_root)
    submitted_path = bundle_root / "summaries" / "multinode_k_sweep_submitted.csv"
    submitted_rows = read_csv(submitted_path)
    submitted_by_exp = dict((row["exp_id"], row) for row in submitted_rows)
    attempts = normalize_attempts(bundle_root, submitted_rows)
    latest = latest_attempt_by_exp(attempts)
    states, qstat_output = qstat_states(args.user)
    if states is None:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        per_run_path = bundle_root / "summaries" / ("multinode_k_sweep_resume_submitted_%s.csv" % stamp)
        write_csv(per_run_path, [], RESUME_FIELDS)
        print("resume_skipped=qstat_unavailable")
        print("resume_needed_submitted=0")
        print("resume_csv=%s" % per_run_path)
        print("qstat_error=%s" % str(qstat_output).strip().replace("\n", " | "))
        return 0

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    resume_rows = []
    new_attempts = []

    for exp_id in sorted(latest):
        if len(resume_rows) >= int(args.max_new):
            break
        attempt = latest[exp_id]
        base = submitted_by_exp.get(exp_id)
        if not base:
            continue
        job_id = attempt["job_id"].split(".")[0]
        state = states.get(job_id, "not_in_qstat")
        target_steps = int(base.get("steps", "20000"))
        step = latest_step(base["train_run_dir"])
        if state in ("Q", "R") or step >= target_steps - 1:
            continue

        old_log = Path(base["pbs_output_log"])
        preserved = ""
        if old_log.exists() and not args.dry_run:
            preserved_path = old_log.with_name("%s.prev_%s_%s" % (old_log.name, job_id, stamp))
            shutil.copy2(str(old_log), str(preserved_path))
            preserved = str(preserved_path)

        status = "submitted"
        error = ""
        if args.dry_run:
            new_job_id = "DRYRUN-%s" % exp_id
            status = "dry_run"
        else:
            try:
                new_job_id = subprocess.check_output(
                    ["qsub", base["pbs_path"]],
                    universal_newlines=True,
                    stderr=subprocess.STDOUT,
                ).strip().split()[0]
            except subprocess.CalledProcessError as exc:
                new_job_id = ""
                status = "qsub_failed"
                error = exc.output.strip()

        resume_row = {
            "old_job_id": attempt["job_id"],
            "new_job_id": new_job_id,
            "exp_id": exp_id,
            "latest_step_before_resume": str(step),
            "latest_checkpoint_before_resume": latest_checkpoint(base["train_run_dir"]),
            "pbs_path": base["pbs_path"],
            "train_run_dir": base["train_run_dir"],
            "pbs_output_log": base["pbs_output_log"],
            "preserved_old_pbs_output": preserved,
            "submitted_at": now,
            "status": status,
            "error": error,
        }
        resume_rows.append(resume_row)
        if new_job_id:
            new_attempts.append(
                {
                    "exp_id": exp_id,
                    "attempt_role": "resume",
                    "job_id": new_job_id,
                    "previous_job_id": attempt["job_id"],
                    "pbs_path": base["pbs_path"],
                    "train_run_dir": base["train_run_dir"],
                    "pbs_output_log": base["pbs_output_log"],
                    "submitted_at": now,
                    "status_hint": status,
                }
            )

    per_run_path = bundle_root / "summaries" / ("multinode_k_sweep_resume_submitted_%s.csv" % stamp)
    write_csv(per_run_path, resume_rows, RESUME_FIELDS)
    if not args.dry_run:
        append_csv(bundle_root / "summaries" / "multinode_k_sweep_resume_submitted.csv", resume_rows, RESUME_FIELDS)
        append_csv(bundle_root / "summaries" / "multinode_k_sweep_scheduler_attempts.csv", new_attempts, ATTEMPT_FIELDS)

    print("resume_needed_submitted=%d" % len(resume_rows))
    print("resume_csv=%s" % per_run_path)
    for row in resume_rows:
        print(
            "%s %s old=%s step=%s ckpt=%s status=%s"
            % (
                row["new_job_id"] or "<none>",
                row["exp_id"],
                row["old_job_id"].split(".")[0],
                row["latest_step_before_resume"],
                row["latest_checkpoint_before_resume"] or "<none>",
                row["status"],
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())

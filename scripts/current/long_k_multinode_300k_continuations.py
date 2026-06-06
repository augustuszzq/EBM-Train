#!/usr/bin/env python3
"""Submit and maintain 300k continuations for completed M0 long-K runs."""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


BUNDLE_ROOT = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_long_k_multinode_sweep")
PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
TARGET_STEPS = 300000

FIELDS = [
    "phase",
    "source_exp_id",
    "exp_id",
    "scale_group",
    "train_mode",
    "world_size",
    "pipe_stages",
    "stage_slices",
    "K",
    "steps",
    "lr",
    "step_size",
    "noise_std",
    "weight_mode",
    "last2_beta",
    "seed",
    "num_nodes",
    "ppn",
    "batch_size",
    "local_batch",
    "save_every",
    "vis_every",
    "eval_images",
    "queue",
    "train_walltime",
    "data_dir",
    "source_run_dir",
    "source_checkpoint",
    "config_path",
    "pbs_path",
    "manifest_row_path",
    "run_dir",
    "pbs_output_log",
]

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
    return max(ckpts)[1] if ckpts else ""


def qstat_states(user):
    try:
        out = subprocess.check_output(["qstat", "-u", user], universal_newlines=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        return None, exc.output
    states = {}
    for line in out.splitlines():
        match = re.match(
            r"^(\d+)\.\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\d+\s+\d+\s+\S+\s+\S+\s+([RQUEH])\s+",
            line,
        )
        if match:
            states[match.group(1)] = match.group(2)
    return states, out


def copy_seed_checkpoint(source_ckpt, run_dir):
    if not source_ckpt:
        return ""
    source = Path(source_ckpt)
    dst_dir = Path(run_dir) / "checkpoints"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / source.name
    if not dst.exists():
        shutil.copy2(str(source), str(dst))
    src_rank = source.parent / "rank_state"
    if src_rank.exists():
        dst_rank = dst_dir / "rank_state"
        dst_rank.mkdir(parents=True, exist_ok=True)
        for src in src_rank.glob(source.stem + "_rank*.pt"):
            out = dst_rank / src.name
            if not out.exists():
                shutil.copy2(str(src), str(out))
    return str(dst)


def config_text(row):
    slices = ", ".join(row["stage_slices"].split(";"))
    return "\n".join(
        [
            "benchmark:",
            "  name: cifar10",
            "  conditional: false",
            "  image_size: 32",
            "  data_root: %s" % row["data_dir"],
            "  split_train: train",
            "  split_val: train",
            "train:",
            "  steps: %s" % row["steps"],
            "  batch_size: %s" % row["batch_size"],
            "  K: %s" % row["K"],
            "  lr: %s" % row["lr"],
            "  step_size: %s" % row["step_size"],
            "pipeline:",
            "  pipe_stages: %s" % row["pipe_stages"],
            "  weight_mode: %s" % row["weight_mode"],
            "  stage_slices: [%s]" % slices,
            "m1_300k_continuation:",
            "  source_exp_id: %s" % row["source_exp_id"],
            "  source_checkpoint: %s" % row["source_checkpoint"],
            "  objective_note: weights are applied to negative energy scalars, not images",
            "",
        ]
    )


def pbs_text(row):
    env = {
        "PROJECT_DIR": str(PROJECT_DIR),
        "GROUP": "long_k_multinode_300k",
        "EXP_ID": row["exp_id"],
        "STORY": "300k continuation from completed M0 long-K multinode checkpoint",
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
        "NOTES": "M1 300k continuation from %s" % row["source_exp_id"],
        "RUN_DIR": row["run_dir"],
        "MANIFEST_ROW_PATH": row["manifest_row_path"],
        "SAVE_EVERY": row["save_every"],
        "VIS_EVERY": row["vis_every"],
        "BATCH_SIZE": row["batch_size"],
        "NOISE_STD": row["noise_std"],
        "LANGEVIN_SIGN": "1.0",
        "WEIGHT_DECAY": "0.0",
        "MAX_GRAD_NORM": "0.0",
        "DATA_DIR": row["data_dir"],
        "CONFIG": row["config_path"],
        "SYNC_FRESH_INIT": "1",
        "DEBUG_LEVEL": "1",
        "LOG_EVERY": "10",
        "RESUME_CKPT": "auto",
    }
    exports = "\n".join("export %s=%s" % (k, shell_quote(v)) for k, v in env.items())
    name = row["exp_id"].replace("_", "")[:15]
    return "\n".join(
        [
            "#!/bin/bash",
            "#PBS -A lc-mpi",
            "#PBS -l select=%s:system=polaris" % row["num_nodes"],
            "#PBS -l walltime=%s" % row["train_walltime"],
            "#PBS -l filesystems=home:eagle",
            "#PBS -q %s" % row["queue"],
            "#PBS -N %s" % name,
            "#PBS -j oe",
            "#PBS -o %s" % row["pbs_output_log"],
            "",
            "set -euo pipefail",
            exports,
            "cp %s %s" % (shell_quote(row["config_path"]), shell_quote(str(Path(row["run_dir"]) / "experiment_config.yaml"))),
            "exec /usr/bin/bash %s" % shell_quote(str(PROJECT_DIR / "scripts" / "current" / "pbs_run_ablation_case.pbs")),
            "",
        ]
    )


def shell_quote(value):
    import shlex

    return shlex.quote(str(value))


def build_rows(include_partial=False):
    registry = {r["exp_id"]: r for r in read_csv(BUNDLE_ROOT / "summaries" / "multinode_k_sweep_master_registry.csv")}
    submitted = {r["exp_id"]: r for r in read_csv(BUNDLE_ROOT / "summaries" / "multinode_k_sweep_submitted.csv")}
    rows = []
    for exp_id, base in sorted(registry.items()):
        source_run = submitted[exp_id]["train_run_dir"]
        source_step = latest_step(source_run)
        if (not include_partial) and source_step < int(base["steps"]) - 1:
            continue
        ckpt = latest_checkpoint(source_run)
        if (not include_partial) and not ckpt:
            continue
        row = dict(base)
        row["phase"] = "M1_300k"
        row["source_exp_id"] = exp_id
        row["exp_id"] = "M1_%s_to300k" % exp_id
        row["steps"] = str(TARGET_STEPS)
        row["source_run_dir"] = source_run
        row["source_checkpoint"] = ckpt
        row["source_latest_step"] = str(source_step)
        row["train_walltime"] = "72:00:00"
        row["config_path"] = str(BUNDLE_ROOT / "configs" / ("%s.yaml" % row["exp_id"]))
        row["pbs_path"] = str(BUNDLE_ROOT / "pbs" / ("%s.pbs" % row["exp_id"]))
        row["manifest_row_path"] = str(BUNDLE_ROOT / "configs" / ("%s.manifest_row.json" % row["exp_id"]))
        row["run_dir"] = str(BUNDLE_ROOT / "runs" / "M1_300k" / row["scale_group"] / row["exp_id"])
        row["pbs_output_log"] = str(BUNDLE_ROOT / "logs" / ("%s.pbs.out" % row["exp_id"]))
        rows.append(row)
    return rows


def materialize(rows):
    for sub in ("configs", "pbs", "logs", "runs", "summaries"):
        (BUNDLE_ROOT / sub).mkdir(parents=True, exist_ok=True)
    out = []
    for row in rows:
        run_dir = Path(row["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        row = dict(row)
        row["source_checkpoint"] = copy_seed_checkpoint(row["source_checkpoint"], row["run_dir"])
        Path(row["config_path"]).write_text(config_text(row))
        Path(row["manifest_row_path"]).write_text(json.dumps(row, indent=2, sort_keys=True))
        Path(row["pbs_path"]).write_text(pbs_text(row))
        out.append(row)
    write_csv(BUNDLE_ROOT / "summaries" / "m1_300k_master_registry.csv", out, FIELDS)
    return out


def submit_new(rows, dry_run=False):
    submitted_path = BUNDLE_ROOT / "summaries" / "m1_300k_submitted.csv"
    attempts_path = BUNDLE_ROOT / "summaries" / "m1_300k_scheduler_attempts.csv"
    existing = {r["exp_id"] for r in read_csv(submitted_path)}
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    submitted = []
    attempts = []
    failures = []
    for row in rows:
        if row["exp_id"] in existing:
            continue
        if dry_run:
            job_id = "DRYRUN-%s" % row["exp_id"]
            status = "dry_run"
        else:
            try:
                job_id = subprocess.check_output(
                    ["qsub", row["pbs_path"]],
                    universal_newlines=True,
                    stderr=subprocess.STDOUT,
                ).strip().split()[0]
                status = "submitted"
            except subprocess.CalledProcessError as exc:
                failures.append(
                    {
                        "phase": row["phase"],
                        "source_exp_id": row["source_exp_id"],
                        "exp_id": row["exp_id"],
                        "config_path": row["config_path"],
                        "pbs_path": row["pbs_path"],
                        "train_run_dir": row["run_dir"],
                        "pbs_output_log": row["pbs_output_log"],
                        "submitted_at": now,
                        "status": "qsub_failed",
                        "error": exc.output.strip(),
                    }
                )
                continue
        submitted.append(
            {
                "phase": row["phase"],
                "source_exp_id": row["source_exp_id"],
                "exp_id": row["exp_id"],
                "train_job_id": job_id,
                "config_path": row["config_path"],
                "pbs_path": row["pbs_path"],
                "train_run_dir": row["run_dir"],
                "pbs_output_log": row["pbs_output_log"],
                "submitted_at": now,
                "status": status,
            }
        )
        attempts.append(
            {
                "exp_id": row["exp_id"],
                "attempt_role": "initial",
                "job_id": job_id,
                "previous_job_id": "",
                "pbs_path": row["pbs_path"],
                "train_run_dir": row["run_dir"],
                "pbs_output_log": row["pbs_output_log"],
                "submitted_at": now,
                "status_hint": status,
            }
        )
    if not dry_run:
        append_csv(submitted_path, submitted, ["phase", "source_exp_id", "exp_id", "train_job_id", "config_path", "pbs_path", "train_run_dir", "pbs_output_log", "submitted_at", "status"])
        append_csv(attempts_path, attempts, ATTEMPT_FIELDS)
        append_csv(BUNDLE_ROOT / "summaries" / "m1_300k_submit_failures.csv", failures, ["phase", "source_exp_id", "exp_id", "config_path", "pbs_path", "train_run_dir", "pbs_output_log", "submitted_at", "status", "error"])
    for fail in failures:
        print("QSUB_FAILED %s error=%s" % (fail["exp_id"], fail["error"].replace("\n", " | ")))
    return submitted


def resume_existing(user, dry_run=False):
    registry = {r["exp_id"]: r for r in read_csv(BUNDLE_ROOT / "summaries" / "m1_300k_master_registry.csv")}
    submitted = {r["exp_id"]: r for r in read_csv(BUNDLE_ROOT / "summaries" / "m1_300k_submitted.csv")}
    attempts = read_csv(BUNDLE_ROOT / "summaries" / "m1_300k_scheduler_attempts.csv")
    latest = {}
    for row in attempts:
        latest[row["exp_id"]] = row
    states, qerr = qstat_states(user)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    if states is None:
        out = BUNDLE_ROOT / "summaries" / ("m1_300k_resume_submitted_%s.csv" % stamp)
        write_csv(out, [], RESUME_FIELDS)
        print("resume_skipped=qstat_unavailable")
        print("resume_needed_submitted=0")
        print("resume_csv=%s" % out)
        print("qstat_error=%s" % str(qerr).strip().replace("\n", " | "))
        return []
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rows = []
    new_attempts = []
    for exp_id in sorted(registry):
        if exp_id not in latest or exp_id not in submitted:
            continue
        attempt = latest[exp_id]
        jid = attempt["job_id"].split(".")[0]
        state = states.get(jid, "not_in_qstat")
        step = latest_step(submitted[exp_id]["train_run_dir"])
        if state in ("Q", "R") or step >= TARGET_STEPS - 1:
            continue
        if dry_run:
            new_job = "DRYRUN-%s" % exp_id
            status = "dry_run"
            err = ""
        else:
            try:
                new_job = subprocess.check_output(["qsub", submitted[exp_id]["pbs_path"]], universal_newlines=True, stderr=subprocess.STDOUT).strip().split()[0]
                status = "submitted"
                err = ""
            except subprocess.CalledProcessError as exc:
                new_job = ""
                status = "qsub_failed"
                err = exc.output.strip()
        row = {
            "old_job_id": attempt["job_id"],
            "new_job_id": new_job,
            "exp_id": exp_id,
            "latest_step_before_resume": str(step),
            "latest_checkpoint_before_resume": latest_checkpoint(submitted[exp_id]["train_run_dir"]),
            "pbs_path": submitted[exp_id]["pbs_path"],
            "train_run_dir": submitted[exp_id]["train_run_dir"],
            "pbs_output_log": submitted[exp_id]["pbs_output_log"],
            "submitted_at": now,
            "status": status,
            "error": err,
        }
        rows.append(row)
        if new_job and not dry_run:
            new_attempts.append(
                {
                    "exp_id": exp_id,
                    "attempt_role": "resume",
                    "job_id": new_job,
                    "previous_job_id": attempt["job_id"],
                    "pbs_path": submitted[exp_id]["pbs_path"],
                    "train_run_dir": submitted[exp_id]["train_run_dir"],
                    "pbs_output_log": submitted[exp_id]["pbs_output_log"],
                    "submitted_at": now,
                    "status_hint": status,
                }
            )
    out = BUNDLE_ROOT / "summaries" / ("m1_300k_resume_submitted_%s.csv" % stamp)
    write_csv(out, rows, RESUME_FIELDS)
    if not dry_run:
        append_csv(BUNDLE_ROOT / "summaries" / "m1_300k_resume_submitted.csv", rows, RESUME_FIELDS)
        append_csv(BUNDLE_ROOT / "summaries" / "m1_300k_scheduler_attempts.csv", new_attempts, ATTEMPT_FIELDS)
    print("resume_needed_submitted=%d" % len(rows))
    print("resume_csv=%s" % out)
    for row in rows:
        print("%s %s old=%s step=%s status=%s" % (row["new_job_id"] or "<none>", row["exp_id"], row["old_job_id"].split(".")[0], row["latest_step_before_resume"], row["status"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit-new", action="store_true")
    ap.add_argument("--resume-existing", action="store_true")
    ap.add_argument("--include-partial", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--user", default="kevienzzq")
    args = ap.parse_args()
    if args.resume_existing:
        resume_existing(user=args.user, dry_run=args.dry_run)
        return 0
    rows = materialize(build_rows(include_partial=args.include_partial))
    if args.submit_new:
        submitted = submit_new(rows, dry_run=args.dry_run)
        print("m1_rows=%d" % len(rows))
        print("submitted=%d" % len(submitted))
        for row in submitted:
            print("%s %s source=%s" % (row["train_job_id"], row["exp_id"], row["source_exp_id"]))
    else:
        print("m1_rows=%d" % len(rows))
        print("registry=%s" % (BUNDLE_ROOT / "summaries" / "m1_300k_master_registry.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

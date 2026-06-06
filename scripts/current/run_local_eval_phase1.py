#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
RUNTIME = PROJECT_DIR / "scripts" / "current" / "run_eval_ablation_case.sh"


def truthy(text):
    return str(text).strip().lower() in {"1", "true", "yes", "y"}


def load_rows(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path):
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def train_succeeded(run_dir):
    payload = load_json(Path(run_dir) / "job_info.json")
    return payload.get("train_status") == "succeeded"


def metrics_exist(eval_dir):
    return (Path(eval_dir) / "metrics_compare.json").exists()


def iter_candidates(rows, groups=None, exp_ids=None):
    groups = set(groups or [])
    exp_ids = set(exp_ids or [])
    for row in rows:
        if not truthy(row.get("submit", "1")):
            continue
        if groups and row.get("group") not in groups:
            continue
        if exp_ids and row.get("exp_id") not in exp_ids:
            continue
        run_dir = row.get("train_run_dir", "").strip()
        eval_dir = row.get("eval_run_dir", "").strip() or str(Path(run_dir) / "eval")
        if not run_dir:
            continue
        if not train_succeeded(run_dir):
            continue
        if metrics_exist(eval_dir):
            continue
        yield row, run_dir, eval_dir


def main():
    parser = argparse.ArgumentParser("Run local Phase 1 evals")
    parser.add_argument("--submitted-manifest", default=str(PROJECT_DIR / "experiments" / "ablation_manifest_phase1_submitted.csv"))
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--exp-id", action="append", default=[])
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.submitted_manifest)
    count = 0
    for row, run_dir, eval_dir in iter_candidates(rows, groups=args.group, exp_ids=args.exp_id):
        if args.max_cases and count >= args.max_cases:
            break
        count += 1
        print(f"{row['exp_id']}\t{run_dir}")
        if args.dry_run:
            continue
        env = os.environ.copy()
        for key in [
            "GROUP",
            "EXP_ID",
            "TRAIN_MODE",
            "RUN_DIR",
            "EVAL_DIR",
            "EVAL_IMAGES",
            "EVAL_BATCH",
            "K_EVAL",
            "STEP_SIZE",
            "NOISE_STD",
            "LANGEVIN_SIGN",
            "EVAL_SEED",
            "DATA_DIR",
        ]:
            value = row.get(key.lower(), "") or row.get(key, "")
            if value:
                env[key] = value
        env["PROJECT_DIR"] = str(PROJECT_DIR)
        env["RUN_DIR"] = run_dir
        env["EVAL_DIR"] = eval_dir
        if args.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        subprocess.check_call(["bash", str(RUNTIME)], env=env)


if __name__ == "__main__":
    sys.exit(main())

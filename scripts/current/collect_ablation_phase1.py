#!/usr/bin/env python3
"""Collect Phase 1 ablation outputs into unified CSV/JSON/Markdown summaries."""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

try:
    from ablation_common import (
        collect_training_tail_stats,
        load_json,
        normalize_metrics_payload,
        write_csv_rows,
    )
except Exception:
    from polaris_ebm.scripts.current.ablation_common import (
        collect_training_tail_stats,
        load_json,
        normalize_metrics_payload,
        write_csv_rows,
    )


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def path_exists(text: str) -> bool:
    return bool(text) and Path(text).exists()


def infer_status(row: Dict[str, str], metrics_path: str, job_info: Dict) -> str:
    if str(row.get("submit", "1")).strip() in ("0", "false", "False"):
        return "reference"
    if path_exists(metrics_path):
        return "done"
    eval_status = job_info.get("eval_status", "")
    train_status = job_info.get("train_status", "")
    if eval_status:
        return str(eval_status)
    if train_status:
        return str(train_status)
    return "pending"


def main() -> None:
    ap = argparse.ArgumentParser("Collect Phase 1 ablation results")
    ap.add_argument(
        "--submitted_manifest",
        default="/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/experiments/ablation_manifest_phase1_submitted.csv",
    )
    ap.add_argument("--out_dir", default="/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation")
    ap.add_argument("--summary-prefix", default="phase1_summary")
    ap.add_argument("--render-script", default="")
    args = ap.parse_args()

    submitted_rows = load_rows(Path(args.submitted_manifest))
    summary_rows: List[Dict[str, object]] = []

    for row in submitted_rows:
        run_dir = row.get("train_run_dir", "").strip() or row.get("reference_run_dir", "").strip()
        eval_dir = row.get("eval_run_dir", "").strip() or row.get("reference_eval_dir", "").strip()
        metrics_path = row.get("reference_metrics_json", "").strip()
        if not metrics_path and eval_dir:
            metrics_path = str(Path(eval_dir) / "metrics_compare.json")
        if not metrics_path and run_dir:
            metrics_path = str(Path(run_dir) / "metrics_compare.json")

        stats_path = row.get("reference_stats_json", "").strip()
        if not stats_path and eval_dir:
            stats_path = str(Path(eval_dir) / "stats.json")

        metrics = load_json(metrics_path, default={}) or {}
        stats = load_json(stats_path, default={}) or {}
        job_info = load_json(Path(run_dir) / "job_info.json", default={}) if run_dir else {}

        norm = normalize_metrics_payload(metrics) if metrics else {
            "fid_inception": None,
            "fid_feature": None,
            "unique_ratio": None,
            "distinct_images": None,
            "diversity_trace": None,
        }

        if isinstance(stats, dict) and "train_tail" in stats:
            train_tail = stats.get("train_tail") or {}
            walltime = stats.get("total_walltime_seconds")
        elif isinstance(stats, dict) and "max_f_neg_last_1k_mean" in stats:
            train_tail = dict(stats)
            if "fneg_stage2_over_stage3_last_1k_mean" not in train_tail and "fneg2_over_fneg3_last_1k_mean" in train_tail:
                train_tail["fneg_stage2_over_stage3_last_1k_mean"] = train_tail["fneg2_over_fneg3_last_1k_mean"]
            walltime = row.get("reference_walltime_seconds", "")
        else:
            train_tail = collect_training_tail_stats(run_dir=run_dir, train_mode=row["train_mode"]) if run_dir else {}
            walltime = job_info.get("total_walltime_seconds") or row.get("reference_walltime_seconds", "")

        if not walltime and job_info:
            walltime = job_info.get("total_walltime_seconds") or job_info.get("train_walltime_seconds")

        summary_rows.append(
            {
                "group": row["group"],
                "exp_id": row["exp_id"],
                "story": row.get("story", ""),
                "train_mode": row["train_mode"],
                "world_size": row["world_size"],
                "pipe_stages": row["pipe_stages"],
                "K": row["K"],
                "steps": row["steps"],
                "lr": row["lr"],
                "step_size": row["step_size"],
                "weight_mode": row["weight_mode"],
                "last2_beta": row.get("last2_beta", ""),
                "fid_inception": norm.get("fid_inception"),
                "fid_feature": norm.get("fid_feature"),
                "unique_ratio": norm.get("unique_ratio"),
                "distinct_images": norm.get("distinct_images"),
                "diversity_trace": norm.get("diversity_trace"),
                "max_f_neg_last_1k_mean": train_tail.get("max_f_neg_last_1k_mean"),
                "max_abs_chain_last_1k_mean": train_tail.get("max_abs_chain_last_1k_mean"),
                "fneg_stage2_over_stage3_last_1k_mean": train_tail.get("fneg_stage2_over_stage3_last_1k_mean"),
                "walltime": walltime,
                "run_dir": run_dir,
                "eval_dir": eval_dir,
                "status": infer_status(row=row, metrics_path=metrics_path, job_info=job_info or {}),
                "train_job_id": row.get("train_job_id", ""),
                "eval_job_id": row.get("eval_job_id", ""),
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_prefix = args.summary_prefix
    summary_csv = out_dir / f"{summary_prefix}.csv"
    summary_json = out_dir / f"{summary_prefix}.json"
    summary_md = out_dir / f"{summary_prefix}.md"

    fieldnames = [
        "group",
        "exp_id",
        "story",
        "train_mode",
        "world_size",
        "pipe_stages",
        "K",
        "steps",
        "lr",
        "step_size",
        "weight_mode",
        "last2_beta",
        "fid_inception",
        "fid_feature",
        "unique_ratio",
        "distinct_images",
        "diversity_trace",
        "max_f_neg_last_1k_mean",
        "max_abs_chain_last_1k_mean",
        "fneg_stage2_over_stage3_last_1k_mean",
        "walltime",
        "run_dir",
        "eval_dir",
        "status",
        "train_job_id",
        "eval_job_id",
    ]
    write_csv_rows(summary_csv, summary_rows, fieldnames=fieldnames)
    summary_json.write_text(json.dumps(summary_rows, indent=2))

    render_script = Path(args.render_script) if args.render_script else Path(__file__).resolve().parent / "render_ablation_phase1_report.py"
    subprocess.check_call(
        [
            sys.executable,
            str(render_script),
            "--summary_csv",
            str(summary_csv),
            "--out_md",
            str(summary_md),
        ]
    )

    print(f"[COLLECT] csv={summary_csv}")
    print(f"[COLLECT] json={summary_json}")
    print(f"[COLLECT] md={summary_md}")


if __name__ == "__main__":
    main()

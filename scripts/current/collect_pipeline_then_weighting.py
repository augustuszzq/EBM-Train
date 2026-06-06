#!/usr/bin/env python3
"""Collect pipeline-then-weighting experiment outputs into CSV/JSON/Markdown summaries."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

try:
    from ablation_common import (
        collect_training_tail_stats,
        load_json,
        normalize_metrics_payload,
        pretty_walltime,
        write_csv_rows,
    )
except Exception:
    from polaris_ebm.scripts.current.ablation_common import (
        collect_training_tail_stats,
        load_json,
        normalize_metrics_payload,
        pretty_walltime,
        write_csv_rows,
    )


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def path_exists(text: str) -> bool:
    return bool(text) and Path(text).exists()


def infer_status(row: Dict[str, str], metrics_path: str, job_info: Dict) -> str:
    if str(row.get("submit", "no")).strip().lower() not in {"1", "true", "yes", "y"}:
        return "prepared"
    if path_exists(metrics_path):
        return "done"
    eval_status = job_info.get("eval_status", "")
    train_status = job_info.get("train_status", "")
    if eval_status:
        return str(eval_status)
    if train_status:
        return str(train_status)
    return "pending"


def fmt_num(text, digits: int = 3) -> str:
    if text in ("", "None", "nan", "NaN", None):
        return ""
    return f"{float(text):.{digits}f}"


def render_table_e1(rows: List[Dict[str, object]]) -> List[str]:
    header = [
        "| P / GPUs | single emulation deep-only FID / time | multi-GPU pipeline deep-only FID / time |",
        "| ---: | ---: | ---: |",
    ]
    mapping = {2: {}, 4: {}, 8: {}, 16: {}}
    for row in rows:
        p = int(row["pipe_stages"])
        if row["train_mode"] == "single_pipe_emul":
            mapping[p]["single"] = row
        elif row["train_mode"] == "pipe_strict":
            mapping[p]["pipe"] = row
    lines = list(header)
    for p in [2, 4, 8, 16]:
        single = mapping[p].get("single", {})
        pipe = mapping[p].get("pipe", {})
        single_text = f"{fmt_num(single.get('fid_inception'))} / {pretty_walltime(single.get('walltime'))}" if single else ""
        pipe_text = f"{fmt_num(pipe.get('fid_inception'))} / {pretty_walltime(pipe.get('walltime'))}" if pipe else ""
        lines.append(f"| {p} | {single_text} | {pipe_text} |")
    return lines


def render_table_e2(rows: List[Dict[str, object]]) -> List[str]:
    header = [
        "| weighting | α | single emulation FID / time | multi-GPU pipeline FID / time |",
        "| --- | --- | ---: | ---: |",
    ]
    order = [
        ("uniform", "", "uniform"),
        ("deep-only", "", "deep_only"),
        ("last2 β=0.005", "0.005", "last2_beta"),
        ("last2 β=0.01", "0.01", "last2_beta"),
        ("last2 β=0.02", "0.02", "last2_beta"),
        ("last2 β=0.03", "0.03", "last2_beta"),
        ("last2 β=0.05", "0.05", "last2_beta"),
    ]
    lines = list(header)
    for label, beta, mode in order:
        single = None
        pipe = None
        for row in rows:
            if row["train_mode"] == "single_pipe_emul" and row["weight_mode"] == mode:
                if mode != "last2_beta" or str(row["last2_beta"]) == beta:
                    single = row
            if row["train_mode"] == "pipe_strict" and row["weight_mode"] == mode:
                if mode != "last2_beta" or str(row["last2_beta"]) == beta:
                    pipe = row
        if mode == "uniform":
            alpha = "(1/P,...,1/P)"
        elif mode == "deep_only":
            alpha = "deepest one-hot"
        else:
            alpha = f"last2({beta}, {1.0 - float(beta):.3f})"
        single_text = f"{fmt_num(single.get('fid_inception'))} / {pretty_walltime(single.get('walltime'))}" if single else ""
        pipe_text = f"{fmt_num(pipe.get('fid_inception'))} / {pretty_walltime(pipe.get('walltime'))}" if pipe else ""
        lines.append(f"| {label} | {alpha} | {single_text} | {pipe_text} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser("Collect pipeline-then-weighting results")
    ap.add_argument(
        "--submitted-manifest",
        default="/eagle/lc-mpi/Zhiqing/polaris_ebm/experiments/pipeline_then_weighting_manifest_submitted.csv",
    )
    ap.add_argument("--out-dir", default="/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_pipeline_then_weighting")
    args = ap.parse_args()

    submitted_rows = load_rows(Path(args.submitted_manifest))
    summary_rows: List[Dict[str, object]] = []

    for row in submitted_rows:
        run_dir = row.get("train_run_dir", "").strip()
        eval_dir = row.get("eval_run_dir", "").strip()
        metrics_path = str(Path(eval_dir) / "metrics_compare.json") if eval_dir else ""
        stats_path = str(Path(eval_dir) / "stats.json") if eval_dir else ""
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
        else:
            train_tail = collect_training_tail_stats(run_dir=run_dir, train_mode=row["train_mode"]) if run_dir else {}
            walltime = job_info.get("total_walltime_seconds") or job_info.get("train_walltime_seconds")

        summary_rows.append(
            {
                "phase": row["phase"],
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
    summary_csv = out_dir / "summary.csv"
    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"

    fieldnames = [
        "phase",
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
    write_csv_rows(summary_csv, summary_rows, fieldnames)
    summary_json.write_text(json.dumps(summary_rows, indent=2))

    e1_rows = [row for row in summary_rows if row["phase"] == "E1"]
    e2_rows = [row for row in summary_rows if row["phase"] == "E2"]
    out_lines = [
        "# Pipeline Then Weighting Summary",
        "",
        "## E1: Pipeline Factor First",
        *render_table_e1(e1_rows),
        "",
        "## E2: Weighted-Sum Factor",
        *render_table_e2(e2_rows),
        "",
    ]
    summary_md.write_text("\n".join(out_lines))

    print(f"[COLLECT] csv={summary_csv}")
    print(f"[COLLECT] json={summary_json}")
    print(f"[COLLECT] md={summary_md}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Collect objective-first follow-up experiment outputs into summary tables."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    from ablation_common import (
        collect_training_tail_stats,
        load_json,
        normalize_metrics_payload,
        pretty_walltime,
        write_csv_rows,
    )
except Exception:
    try:
        from scripts.current.ablation_common import (
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
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def path_exists(text: str) -> bool:
    return bool(text) and Path(text).exists()


def expected_trajectory_points(row: Dict[str, str], step_interval: int = 5000) -> int:
    steps = int(row.get("steps", "0") or "0")
    if steps <= 0 or step_interval <= 0:
        return 0
    return steps // step_interval


def infer_status(row: Dict[str, str], metrics_path: str, job_info: Dict, trajectory_points: int = 0) -> str:
    if str(row.get("submit", "no")).strip().lower() not in {"1", "true", "yes", "y"}:
        return "prepared"
    if path_exists(metrics_path):
        return "done"
    eval_status = job_info.get("eval_status", "")
    train_status = job_info.get("train_status", "")
    expected_points = expected_trajectory_points(row)
    if train_status == "succeeded" and expected_points > 0 and trajectory_points >= expected_points:
        return "done"
    if eval_status:
        return str(eval_status)
    if train_status:
        return str(train_status)
    return "pending"


def load_trajectory_points(run_dir: str) -> List[Dict[str, str]]:
    if not run_dir:
        return []
    path = Path(run_dir) / "trajectory" / "trajectory.csv"
    if not path.exists():
        return []
    return load_rows(path)


def float_or_none(value) -> Optional[float]:
    if value in ("", None, "None", "nan", "NaN"):
        return None
    return float(value)


def first_leq(points: List[Dict[str, str]], threshold: float) -> str:
    for row in points:
        fid = float_or_none(row.get("fid_inception"))
        if fid is not None and fid <= threshold:
            return row.get("ordinal_step", "") or row.get("step", "")
    return ""


def late_window_mean(points: List[Dict[str, str]], min_step: int) -> str:
    values = []
    for row in points:
        step = int(row.get("ordinal_step", row.get("step", "0")) or 0)
        fid = float_or_none(row.get("fid_inception"))
        if fid is not None and step >= min_step:
            values.append(fid)
    if not values:
        return ""
    return sum(values) / len(values)


def best_from_points(points: List[Dict[str, str]]) -> Dict[str, object]:
    if not points:
        return {
            "best_fid": "",
            "best_step": "",
            "final_traj_fid": "",
            "final_traj_step": "",
            "trajectory_points": 0,
            "first_leq_70": "",
            "first_leq_65": "",
            "first_leq_60": "",
            "late_window_mean_fid": "",
            "late_window_mean_fid_200k": "",
        }
    ordered = sorted(points, key=lambda row: int(row.get("ordinal_step", row.get("step", "0")) or 0))
    best = min(ordered, key=lambda row: float(row["fid_inception"]))
    final = ordered[-1]
    return {
        "best_fid": float(best["fid_inception"]),
        "best_step": int(best.get("ordinal_step", best.get("step", "0")) or 0),
        "final_traj_fid": float(final["fid_inception"]),
        "final_traj_step": int(final.get("ordinal_step", final.get("step", "0")) or 0),
        "trajectory_points": len(ordered),
        "first_leq_70": first_leq(ordered, 70.0),
        "first_leq_65": first_leq(ordered, 65.0),
        "first_leq_60": first_leq(ordered, 60.0),
        "late_window_mean_fid": late_window_mean(ordered, max(0, int(final.get("ordinal_step", "0") or 0) - 50000)),
        "late_window_mean_fid_200k": late_window_mean(ordered, 200000),
    }


def effective_metric(norm: Dict[str, object], traj_stats: Dict[str, object], key: str) -> object:
    value = norm.get(key)
    if value not in ("", None, "nan", "NaN"):
        return value
    if key == "fid_inception":
        return traj_stats.get("final_traj_fid", "")
    return value


def fmt_num(value, digits: int = 3) -> str:
    if value in ("", None, "nan", "NaN"):
        return ""
    return f"{float(value):.{digits}f}"


def render_phase_table(rows: List[Dict[str, object]], title: str) -> List[str]:
    lines = [f"## {title}", "", "| exp_id | mode | P | weighting | final | best @ step | <=70 | <=65 | <=60 | late-window | status |", "| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        lines.append(
            "| {exp_id} | {train_mode} | {pipe_stages} | {weight_mode} | {final} | {best} @ {best_step} | {t70} | {t65} | {t60} | {late} | {status} |".format(
                exp_id=row["exp_id"],
                train_mode=row["train_mode"],
                pipe_stages=row["pipe_stages"],
                weight_mode=row["weight_mode"],
                final=fmt_num(row.get("fid_inception")),
                best=fmt_num(row.get("best_fid")),
                best_step=row.get("best_step", ""),
                t70=row.get("first_leq_70", ""),
                t65=row.get("first_leq_65", ""),
                t60=row.get("first_leq_60", ""),
                late=fmt_num(row.get("late_window_mean_fid")),
                status=row["status"],
            )
        )
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser("Collect objective-first follow-up results")
    ap.add_argument(
        "--submitted-manifest",
        default="/eagle/lc-mpi/Zhiqing/polaris_ebm/experiments/objective_first_followup_manifest_submitted.csv",
    )
    ap.add_argument("--out-dir", default="/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_objective_first_followup")
    args = ap.parse_args()

    submitted_path = Path(args.submitted_manifest)
    if not submitted_path.exists():
        raise SystemExit("submitted manifest not found: %s" % submitted_path)

    submitted_rows = load_rows(submitted_path)
    summary_rows: List[Dict[str, object]] = []

    for row in submitted_rows:
        run_dir = row.get("train_run_dir", "").strip()
        eval_dir = row.get("eval_run_dir", "").strip() or str(Path(run_dir) / "eval") if run_dir else ""
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

        traj_stats = best_from_points(load_trajectory_points(run_dir))
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
                "seed": row.get("seed", ""),
                "paired_key": row.get("paired_key", ""),
                "target_role": row.get("target_role", ""),
                "budget_basis": row.get("budget_basis", ""),
                "legacy_context": row.get("legacy_context", ""),
                "fid_inception": effective_metric(norm, traj_stats, "fid_inception"),
                "fid_feature": effective_metric(norm, traj_stats, "fid_feature"),
                "unique_ratio": norm.get("unique_ratio"),
                "distinct_images": norm.get("distinct_images"),
                "diversity_trace": norm.get("diversity_trace"),
                "best_fid": traj_stats["best_fid"],
                "best_step": traj_stats["best_step"],
                "final_traj_fid": traj_stats["final_traj_fid"],
                "final_traj_step": traj_stats["final_traj_step"],
                "trajectory_points": traj_stats["trajectory_points"],
                "first_leq_70": traj_stats["first_leq_70"],
                "first_leq_65": traj_stats["first_leq_65"],
                "first_leq_60": traj_stats["first_leq_60"],
                "late_window_mean_fid": traj_stats["late_window_mean_fid"],
                "late_window_mean_fid_200k": traj_stats["late_window_mean_fid_200k"],
                "max_f_neg_last_1k_mean": train_tail.get("max_f_neg_last_1k_mean"),
                "max_abs_chain_last_1k_mean": train_tail.get("max_abs_chain_last_1k_mean"),
                "fneg_stage2_over_stage3_last_1k_mean": train_tail.get("fneg_stage2_over_stage3_last_1k_mean"),
                "walltime": walltime,
                "run_dir": run_dir,
                "eval_dir": eval_dir,
                "status": infer_status(
                    row=row,
                    metrics_path=metrics_path,
                    job_info=job_info or {},
                    trajectory_points=int(traj_stats["trajectory_points"] or 0),
                ),
                "train_job_id": row.get("train_job_id", ""),
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
        "seed",
        "paired_key",
        "target_role",
        "budget_basis",
        "legacy_context",
        "fid_inception",
        "fid_feature",
        "unique_ratio",
        "distinct_images",
        "diversity_trace",
        "best_fid",
        "best_step",
        "final_traj_fid",
        "final_traj_step",
        "trajectory_points",
        "first_leq_70",
        "first_leq_65",
        "first_leq_60",
        "late_window_mean_fid",
        "late_window_mean_fid_200k",
        "max_f_neg_last_1k_mean",
        "max_abs_chain_last_1k_mean",
        "fneg_stage2_over_stage3_last_1k_mean",
        "walltime",
        "run_dir",
        "eval_dir",
        "status",
        "train_job_id",
    ]
    write_csv_rows(summary_csv, summary_rows, fieldnames)
    summary_json.write_text(json.dumps(summary_rows, indent=2))

    phase_order = ["O1", "O2", "O3", "O4", "O5"]
    phase_titles = {
        "O1": "O1: Single-Side Objective Sweep",
        "O2": "O2: Paired Strict-Pipeline Realization",
        "O3": "O3: Seed Expansion",
        "O4": "O4: Long-Horizon Follow-Up",
        "O5": "O5: Compute-Matched Controls",
    }
    lines = ["# Objective-First Follow-Up Summary", ""]
    for phase in phase_order:
        phase_rows = [row for row in summary_rows if row["phase"] == phase]
        if phase_rows:
            lines.extend(render_phase_table(phase_rows, phase_titles[phase]))
    summary_md.write_text("\n".join(lines))

    print("[COLLECT] csv=%s" % summary_csv)
    print("[COLLECT] json=%s" % summary_json)
    print("[COLLECT] md=%s" % summary_md)


if __name__ == "__main__":
    main()

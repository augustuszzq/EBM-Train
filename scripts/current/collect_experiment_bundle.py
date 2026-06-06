#!/usr/bin/env python3
"""Collect the current experiment outputs into one refreshable bundle directory."""

import argparse
import csv
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional


PROJECT_DIR = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm")


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def qstat_status(job_id: str) -> str:
    if not job_id:
        return ""
    try:
        out = subprocess.check_output(["qstat", "-xf", job_id], universal_newlines=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        out = e.output
    state_m = re.search(r"job_state = (\w+)", out)
    exit_m = re.search(r"Exit_status = ([^\n]+)", out)
    if not state_m:
        return "unknown"
    state = state_m.group(1)
    if state == "F":
        return f"F:{exit_m.group(1).strip() if exit_m else '?'}"
    return state


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def latest_train_step(train_log: Path) -> Optional[int]:
    if not train_log.exists():
        return None
    step_pat = re.compile(r"step=(\d+)")
    last = None
    for line in train_log.read_text(errors="ignore").splitlines()[-500:]:
        m = step_pat.search(line)
        if m:
            last = int(m.group(1))
    return last


def trajectory_last_step(traj_csv: Path) -> Optional[int]:
    if not traj_csv.exists():
        return None
    rows = load_csv(traj_csv)
    if not rows:
        return None
    try:
        return max(int(r.get("step", 0)) for r in rows)
    except Exception:
        return None


def build_pipeline_then_weighting_summary() -> List[Dict[str, object]]:
    pbs_rows = {row["exp_id"]: row for row in load_csv(PROJECT_DIR / "experiments" / "pipeline_then_weighting_manifest_submitted.csv")}
    local_rows = {row["exp_id"]: row for row in load_csv(PROJECT_DIR / "experiments" / "pipeline_then_weighting_local_submitted.csv")}
    summary: List[Dict[str, object]] = []
    for exp_id in sorted(set(pbs_rows) | set(local_rows)):
        pbs = pbs_rows.get(exp_id, {})
        local = local_rows.get(exp_id, {})
        base = dict(pbs or local)
        run_dir = Path(local.get("train_run_dir") or pbs.get("train_run_dir") or "")
        train_log = run_dir / "train.log" if run_dir else Path()
        traj_csv = run_dir / "trajectory" / "trajectory.csv" if run_dir else Path()
        final_metrics = run_dir / "trajectory" / "step300000" / "metrics_compare.json" if run_dir else Path()
        metrics = load_json(final_metrics)
        local_done = bool(final_metrics.exists())
        pbs_status = qstat_status((pbs.get("train_job_id") or "").strip()) if pbs else ""
        if local_done:
            status = "done_local"
        elif metrics:
            status = "done_eval"
        elif pbs_status == "F:0":
            status = "done_train"
        elif pbs_status:
            status = pbs_status
        else:
            status = "unknown"
        summary.append(
            {
                "phase": base.get("phase", ""),
                "group": base.get("group", ""),
                "exp_id": exp_id,
                "train_mode": base.get("train_mode", ""),
                "pipe_stages": base.get("pipe_stages", ""),
                "weight_mode": base.get("weight_mode", ""),
                "last2_beta": base.get("last2_beta", ""),
                "steps": base.get("steps", ""),
                "status": status,
                "final_fid": metrics.get("fid_inception_baseline_vs_real", "") if metrics else "",
                "final_fid_feature": metrics.get("fid_feature_baseline_vs_real", "") if metrics else "",
                "final_unique_ratio": metrics.get("unique_ratio", "") if metrics else "",
                "last_train_step": latest_train_step(train_log) or "",
                "last_trajectory_step": trajectory_last_step(traj_csv) or "",
                "run_dir": str(run_dir) if run_dir else "",
                "trajectory_csv": str(traj_csv) if traj_csv.exists() else "",
                "final_metrics_path": str(final_metrics) if final_metrics.exists() else "",
                "pbs_job_id": pbs.get("train_job_id", ""),
                "local_run": "yes" if local else "no",
            }
        )
    return summary


def build_imagenet32_phasea_summary() -> List[Dict[str, object]]:
    rows = load_csv(PROJECT_DIR / "experiments" / "imagenet32_phasea_submitted.csv")
    summary: List[Dict[str, object]] = []
    for row in rows:
        run_dir = Path(row["train_run_dir"])
        fid_path = run_dir / "eval" / "conditional_fid.json"
        acc_path = run_dir / "eval" / "conditional_acc.json"
        fid_payload = load_json(fid_path)
        acc_payload = load_json(acc_path)
        summary.append(
            {
                "exp_id": row["exp_id"],
                "train_mode": row["train_mode"],
                "steps": row["steps"],
                "status": qstat_status((row.get("train_job_id") or "").strip()),
                "final_fid": fid_payload.get("final_fid", fid_payload.get("fid_inception", "")),
                "top1_conditional_acc": acc_payload.get("top1_conditional_acc", ""),
                "per_class_avg_acc": acc_payload.get("per_class_avg_acc", ""),
                "run_dir": str(run_dir),
                "conditional_fid_path": str(fid_path) if fid_path.exists() else "",
                "conditional_acc_path": str(acc_path) if acc_path.exists() else "",
                "train_job_id": row.get("train_job_id", ""),
            }
        )
    return summary


def write_markdown(out_dir: Path, ptw_rows: List[Dict[str, object]], imnet_rows: List[Dict[str, object]], copied: List[str]) -> None:
    done_ptw = sum(1 for r in ptw_rows if str(r["status"]).startswith("done"))
    running_ptw = sum(1 for r in ptw_rows if r["status"] == "R")
    queued_ptw = sum(1 for r in ptw_rows if r["status"] == "Q")
    failed_ptw = sum(1 for r in ptw_rows if str(r["status"]).startswith("F:"))
    imnet_done = sum(1 for r in imnet_rows if r["status"] == "F:0")
    lines = [
        "# Final Experiment Bundle",
        "",
        "This directory collects the current experiment summaries and status snapshots.",
        "",
        "## Snapshot",
        f"- `pipeline_then_weighting`: done={done_ptw}, running={running_ptw}, queued={queued_ptw}, failed_history={failed_ptw}",
        f"- `imagenet32_phasea`: completed={imnet_done}/{len(imnet_rows)}", 
        "",
        "## Included Files",
    ]
    for rel in copied:
        lines.append(f"- `{rel}`")
    lines += [
        "",
        "## Notes",
        "- `pipeline_then_weighting` local single runs are merged with PBS runs so completed local 300k jobs are not mistaken for old failed queue attempts.",
        "- `imagenet32_phasea` currently records train completion status; evaluation JSONs are included if present.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser("Collect current experiment bundle")
    ap.add_argument("--out-dir", default="/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_final_bundle")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied: List[str] = []
    static_files = [
        (PROJECT_DIR / "runs_ablation" / "phase1_summary.csv", "phase1_summary.csv"),
        (PROJECT_DIR / "runs_ablation" / "phase1_summary.json", "phase1_summary.json"),
        (PROJECT_DIR / "runs_ablation" / "phase1_summary.md", "phase1_summary.md"),
        (PROJECT_DIR / "runs_ablation_phase2" / "phase2_summary.csv", "phase2_summary.csv"),
        (PROJECT_DIR / "runs_ablation_phase2" / "phase2_summary.json", "phase2_summary.json"),
        (PROJECT_DIR / "runs_ablation_phase2" / "phase2_summary.md", "phase2_summary.md"),
        (PROJECT_DIR / "runs_ablation_500k" / "ablation_500k_summary.csv", "ablation_500k_summary.csv"),
        (PROJECT_DIR / "runs_ablation_500k" / "ablation_500k_summary.json", "ablation_500k_summary.json"),
        (PROJECT_DIR / "runs_ablation_500k" / "ablation_500k_summary.md", "ablation_500k_summary.md"),
        (PROJECT_DIR / "runs_ablation_replay" / "ablation_replay_points.csv", "ablation_replay_points.csv"),
        (PROJECT_DIR / "runs_ablation_replay" / "ablation_final_summary.csv", "ablation_final_summary.csv"),
        (PROJECT_DIR / "runs_ablation_replay" / "ablation_final_summary.json", "ablation_final_summary.json"),
        (PROJECT_DIR / "runs_analysis" / "seed_fid_trajectory_points.csv", "seed_fid_trajectory_points.csv"),
        (
            PROJECT_DIR / "runs_pipeline_then_weighting_replay" / "trajectory_points.csv",
            "pipeline_then_weighting_replay_points.csv",
        ),
        (
            PROJECT_DIR / "runs_pipeline_then_weighting_replay" / "trajectory_summary.csv",
            "pipeline_then_weighting_replay_summary.csv",
        ),
        (
            PROJECT_DIR / "runs_pipeline_then_weighting_replay" / "trajectory_summary.json",
            "pipeline_then_weighting_replay_summary.json",
        ),
    ]
    for src, rel in static_files:
        if copy_if_exists(src, out_dir / rel):
            copied.append(rel)

    ptw_rows = build_pipeline_then_weighting_summary()
    write_csv(out_dir / "pipeline_then_weighting_current.csv", ptw_rows, ptw_rows[0].keys())
    (out_dir / "pipeline_then_weighting_current.json").write_text(json.dumps(ptw_rows, indent=2))
    copied += ["pipeline_then_weighting_current.csv", "pipeline_then_weighting_current.json"]

    imnet_rows = build_imagenet32_phasea_summary()
    write_csv(out_dir / "imagenet32_phasea_current.csv", imnet_rows, imnet_rows[0].keys())
    (out_dir / "imagenet32_phasea_current.json").write_text(json.dumps(imnet_rows, indent=2))
    copied += ["imagenet32_phasea_current.csv", "imagenet32_phasea_current.json"]

    manifest_files = [
        (PROJECT_DIR / "experiments" / "pipeline_then_weighting_manifest.csv", "pipeline_then_weighting_manifest.csv"),
        (PROJECT_DIR / "experiments" / "pipeline_then_weighting_manifest_submitted.csv", "pipeline_then_weighting_manifest_submitted.csv"),
        (PROJECT_DIR / "experiments" / "pipeline_then_weighting_local_submitted.csv", "pipeline_then_weighting_local_submitted.csv"),
        (PROJECT_DIR / "experiments" / "pipeline_then_weighting_replay_manifest.csv", "pipeline_then_weighting_replay_manifest.csv"),
        (PROJECT_DIR / "experiments" / "imagenet32_phasea_manifest.csv", "imagenet32_phasea_manifest.csv"),
        (PROJECT_DIR / "experiments" / "imagenet32_phasea_submitted.csv", "imagenet32_phasea_submitted.csv"),
    ]
    for src, rel in manifest_files:
        if copy_if_exists(src, out_dir / rel):
            copied.append(rel)

    write_markdown(out_dir, ptw_rows, imnet_rows, copied)
    print(f"[DONE] bundle={out_dir}")


if __name__ == "__main__":
    main()

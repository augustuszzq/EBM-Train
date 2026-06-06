#!/usr/bin/env python3
"""Shared helpers for ablation replay manifest building and collection."""

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_REPLAY_ROOT = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation_replay/trajectories")


def existing_trajectory_overrides() -> Dict[str, str]:
    return {
        "A0_ddp_strict_300k_ref": "/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_analysis/seed1_fid_trajectory/ddp_seed1_local",
        "A0_pipeline_strict_300k_ref": "/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_analysis/seed1_fid_trajectory/pipeline_seed1_local",
        "A1_single_fullk_k100_s20k": "/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_analysis/ablation_fid_trajectory/A1_single_fullk_k100_s20k_local",
        "A1_single_pipe_uniform_p4_k100_s20k": "/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_analysis/ablation_fid_trajectory/A1_single_pipe_uniform_p4_k100_s20k_local",
        "ddp_fullk_K100_seed1_300k": "/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_analysis/seed1_fid_trajectory/ddp_seed1_local",
        "pipe_strict_P4_K100_beta001_lr1e4_seed1_300k": "/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_analysis/seed1_fid_trajectory/pipeline_seed1_local",
        "ddp_fullk_K100_seed2_300k": "/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_analysis/seed2_fid_trajectory/ddp_seed2_local",
        "pipe_strict_P4_K100_beta001_lr1e4_seed2_300k": "/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_analysis/seed2_fid_trajectory/pipeline_seed2_local",
        "ddp_fullk_K100_seed3_300k": "/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_analysis/seed3_fid_trajectory/ddp_seed3_local",
        "pipe_strict_P4_K100_beta001_lr1e4_seed3_300k": "/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_analysis/seed3_fid_trajectory/pipeline_seed3_local",
    }


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def checkpoint_dir_candidates(run_dir: Path) -> List[Path]:
    return [run_dir / "artifacts" / "checkpoints", run_dir / "checkpoints"]


def final_metrics_candidates(run_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    patterns = [
        "metrics_compare.json",
        "metrics_compare*.json",
        "eval/metrics_compare.json",
        "eval/metrics_compare*.json",
        "eval/*/metrics_compare.json",
        "eval/*/metrics_compare*.json",
    ]
    for pattern in patterns:
        candidates.extend(sorted(run_dir.glob(pattern)))
    return candidates


def find_final_metrics_path(run_dir: Path) -> Optional[Path]:
    candidates = final_metrics_candidates(run_dir)
    return candidates[-1] if candidates else None


def expected_trajectory_points(run_dir: Path, step_interval: int = 5000) -> int:
    for ckpt_dir in checkpoint_dir_candidates(run_dir):
        if ckpt_dir.exists():
            count = 0
            for ckpt in ckpt_dir.glob("ckpt_step*.pt"):
                try:
                    step = int(ckpt.stem.replace("ckpt_step", "")) + 1
                except Exception:
                    continue
                if step % step_interval == 0:
                    count += 1
            return count
    return 0


def replay_walltime_for_steps(steps: int) -> str:
    return "06:00:00" if steps >= 300000 else "02:00:00"


def build_replay_rows(
    source_rows: Iterable[Dict[str, str]],
    replay_root: Path = DEFAULT_REPLAY_ROOT,
    overrides: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    overrides = dict(overrides or {})
    rows: List[Dict] = []
    for src in source_rows:
        run_dir = (src.get("train_run_dir") or src.get("reference_run_dir") or "").strip()
        if not run_dir:
            continue
        run_dir_path = Path(run_dir)
        expected_points = expected_trajectory_points(run_dir_path)
        has_final_metrics = find_final_metrics_path(run_dir_path) is not None
        if expected_points == 0 and not has_final_metrics:
            continue
        exp_id = src["exp_id"]
        steps = int(src.get("steps", "0") or "0")
        override_dir = overrides.get(exp_id)
        replay_mode = "existing" if override_dir else ("qsub" if expected_points > 0 else "final_only")
        trajectory_dir = override_dir or (str(replay_root / exp_id) if expected_points > 0 else "")
        rows.append(
            {
                "group": src.get("group", ""),
                "exp_id": exp_id,
                "story": src.get("story", ""),
                "train_mode": src.get("train_mode", ""),
                "seed": int(src.get("seed", "0") or "0"),
                "K": src.get("K", ""),
                "steps": steps,
                "step_size": src.get("step_size", ""),
                "weight_mode": src.get("weight_mode", ""),
                "run_dir": str(run_dir_path),
                "trajectory_dir": trajectory_dir,
                "expected_points": expected_points,
                "replay_mode": replay_mode,
                "replay_walltime": replay_walltime_for_steps(steps) if expected_points > 0 else "",
                "replay_queue": "preemptable" if expected_points > 0 else "",
                "replay_job_id": "",
            }
        )
    rows.sort(key=lambda row: (row["group"], row["exp_id"]))
    return rows


def load_metrics(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def trajectory_progress_summary(trajectory_dir: Path, expected_points: int) -> Dict:
    metrics = sorted(trajectory_dir.glob("step*/metrics_compare.json"))
    available_points = len(metrics)
    if expected_points > 0 and available_points >= expected_points:
        status = "done"
    elif available_points > 0:
        status = "partial"
    else:
        status = "missing"
    final_step = None
    final_fid = None
    if metrics:
        final_step = int(metrics[-1].parent.name.replace("step", ""))
        final_fid = load_metrics(metrics[-1]).get("fid_inception_baseline_vs_real")
    return {
        "available_points": available_points,
        "status": status,
        "final_step": final_step,
        "final_fid": final_fid,
    }


def collect_replay_points(replay_rows: Iterable[Dict]) -> List[Dict]:
    points: List[Dict] = []
    for row in replay_rows:
        if not row.get("trajectory_dir"):
            continue
        trajectory_dir = Path(row["trajectory_dir"])
        for metrics_path in sorted(trajectory_dir.glob("step*/metrics_compare.json")):
            payload = load_metrics(metrics_path)
            points.append(
                {
                    "group": row.get("group", ""),
                    "exp_id": row["exp_id"],
                    "story": row.get("story", ""),
                    "train_mode": row.get("train_mode", ""),
                    "seed": row.get("seed", ""),
                    "step": int(metrics_path.parent.name.replace("step", "")),
                    "fid_inception": payload.get("fid_inception_baseline_vs_real"),
                    "fid_feature": payload.get("fid_feature_baseline_vs_real"),
                    "unique_ratio": payload.get("unique_ratio"),
                    "trajectory_dir": row["trajectory_dir"],
                    "metrics_path": str(metrics_path),
                }
            )
    points.sort(key=lambda row: (int(row["seed"]), row["exp_id"], int(row["step"])))
    return points


def summarize_from_final_metrics(run_dir: Path, replay_row: Dict) -> Optional[Dict]:
    metrics_path = find_final_metrics_path(run_dir)
    if metrics_path is None:
        return None
    payload = load_metrics(metrics_path)
    status = "final_only" if int(replay_row.get("expected_points", 0) or 0) == 0 else "missing_replay"
    return {
        "group": replay_row.get("group", ""),
        "exp_id": replay_row["exp_id"],
        "story": replay_row.get("story", ""),
        "train_mode": replay_row.get("train_mode", ""),
        "seed": replay_row.get("seed", ""),
        "steps": replay_row.get("steps", ""),
        "expected_points": replay_row.get("expected_points", ""),
        "available_points": 0,
        "status": status,
        "final_step": replay_row.get("steps", ""),
        "final_fid": payload.get("fid_inception_baseline_vs_real"),
        "fid_feature": payload.get("fid_feature_baseline_vs_real"),
        "unique_ratio": payload.get("unique_ratio"),
        "best_step": "",
        "best_fid": "",
        "final_minus_best": "",
        "trajectory_dir": replay_row.get("trajectory_dir", ""),
        "run_dir": replay_row.get("run_dir", ""),
    }


def summarize_replay_points(points: List[Dict], replay_rows: Iterable[Dict]) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for point in points:
        grouped.setdefault(point["exp_id"], []).append(point)
    row_by_exp = {row["exp_id"]: row for row in replay_rows}
    summaries: List[Dict] = []
    for exp_id, replay_row in sorted(row_by_exp.items()):
        pts = sorted(grouped.get(exp_id, []), key=lambda row: int(row["step"]))
        trajectory_dir = replay_row.get("trajectory_dir", "")
        progress = (
            trajectory_progress_summary(Path(trajectory_dir), int(replay_row["expected_points"]))
            if trajectory_dir
            else {"available_points": 0, "status": "missing", "final_step": None, "final_fid": None}
        )
        if pts:
            best = min(pts, key=lambda row: float(row["fid_inception"]))
            final = pts[-1]
            best_step = int(best["step"])
            best_fid = float(best["fid_inception"])
            final_step = int(final["step"])
            final_fid = float(final["fid_inception"])
            final_minus_best = final_fid - best_fid
            fid_feature = final.get("fid_feature")
            unique_ratio = final.get("unique_ratio")
        else:
            best_step = ""
            best_fid = ""
            final_step = ""
            final_fid = ""
            final_minus_best = ""
            fid_feature = ""
            unique_ratio = ""
        if pts:
            summaries.append(
                {
                    "group": replay_row.get("group", ""),
                    "exp_id": exp_id,
                    "story": replay_row.get("story", ""),
                    "train_mode": replay_row.get("train_mode", ""),
                    "seed": replay_row.get("seed", ""),
                    "steps": replay_row.get("steps", ""),
                    "expected_points": replay_row.get("expected_points", ""),
                    "available_points": progress["available_points"],
                    "status": progress["status"],
                    "final_step": final_step,
                    "final_fid": final_fid,
                    "fid_feature": fid_feature,
                    "unique_ratio": unique_ratio,
                    "best_step": best_step,
                    "best_fid": best_fid,
                    "final_minus_best": final_minus_best,
                    "trajectory_dir": replay_row.get("trajectory_dir", ""),
                    "run_dir": replay_row.get("run_dir", ""),
                }
            )
            continue
        fallback = summarize_from_final_metrics(Path(replay_row["run_dir"]), replay_row)
        if fallback is not None:
            summaries.append(fallback)
            continue
        summaries.append(
            {
                "group": replay_row.get("group", ""),
                "exp_id": exp_id,
                "story": replay_row.get("story", ""),
                "train_mode": replay_row.get("train_mode", ""),
                "seed": replay_row.get("seed", ""),
                "steps": replay_row.get("steps", ""),
                "expected_points": replay_row.get("expected_points", ""),
                "available_points": progress["available_points"],
                "status": progress["status"],
                "final_step": final_step,
                "final_fid": final_fid,
                "fid_feature": fid_feature,
                "unique_ratio": unique_ratio,
                "best_step": best_step,
                "best_fid": best_fid,
                "final_minus_best": final_minus_best,
                "trajectory_dir": replay_row.get("trajectory_dir", ""),
                "run_dir": replay_row.get("run_dir", ""),
            }
        )
    return summaries

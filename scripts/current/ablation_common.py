#!/usr/bin/env python3
"""Shared helpers for Phase 1 ablation training, evaluation, and collection."""

import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union


def compute_k_slices(k: int, pipe_stages: int) -> List[int]:
    if int(k) <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if int(pipe_stages) <= 0:
        raise ValueError(f"pipe_stages must be > 0, got {pipe_stages}")
    base = int(k) // int(pipe_stages)
    rem = int(k) % int(pipe_stages)
    return [base + (1 if i < rem else 0) for i in range(int(pipe_stages))]


def completion_ratios(k_slices: Sequence[int]) -> List[float]:
    total = float(sum(int(x) for x in k_slices))
    if total <= 0:
        raise ValueError("sum(k_slices) must be > 0")
    out: List[float] = []
    acc = 0.0
    for ks in k_slices:
        acc += float(int(ks))
        out.append(acc / total)
    return out


def active_stage_indices(step: int, pipe_stages: int) -> List[int]:
    if int(pipe_stages) <= 0:
        return []
    last = min(int(step), int(pipe_stages) - 1)
    return list(range(last + 1))


def compute_completion_aware_alpha(
    step: int,
    pipe_stages: int,
    weight_mode: str,
    last2_beta: float,
) -> List[float]:
    mode = str(weight_mode).strip().lower()
    active = active_stage_indices(step=step, pipe_stages=pipe_stages)
    alpha = [0.0 for _ in range(int(pipe_stages))]
    if not active:
        return alpha
    if mode == "uniform":
        w = 1.0 / float(len(active))
        for idx in active:
            alpha[idx] = w
        return alpha
    if mode == "deep_only":
        alpha[active[-1]] = 1.0
        return alpha
    if mode == "last2_beta":
        beta = float(last2_beta)
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"last2_beta must be in [0,1], got {last2_beta}")
        if len(active) == 1:
            alpha[active[-1]] = 1.0
            return alpha
        alpha[active[-2]] = beta
        alpha[active[-1]] = 1.0 - beta
        return alpha
    raise ValueError(f"unsupported completion-aware weight_mode={weight_mode}")


def canonical_weight_summary(
    pipe_stages: int,
    weight_mode: str,
    last2_beta: float,
) -> str:
    mode = str(weight_mode).strip().lower()
    if mode == "uniform":
        return f"uniform over active stages (P={pipe_stages})"
    if mode == "deep_only":
        return "deepest active stage only"
    if mode == "last2_beta":
        return f"last two active stages: beta={float(last2_beta):.4f}, 1-beta={1.0 - float(last2_beta):.4f}"
    return mode


def load_json(path: Union[os.PathLike, str], default=None):
    if path in (None, ""):
        return default
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r") as f:
        return json.load(f)


def write_json(path: Union[os.PathLike, str], payload: Dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def merge_json(path: Union[os.PathLike, str], updates: Dict) -> None:
    payload = load_json(path, default={}) or {}
    payload.update(updates)
    write_json(path, payload)


def read_csv_rows(path: Union[os.PathLike, str]) -> List[Dict[str, str]]:
    with Path(path).open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Union[os.PathLike, str], rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def find_latest_checkpoint(run_dir: Union[os.PathLike, str]) -> str:
    ckpts = sorted(Path(run_dir).joinpath("checkpoints").glob("ckpt_step*.pt"))
    if not ckpts:
        return ""
    return str(ckpts[-1])


def _tail_rows(path: Path, last_n: int) -> List[Dict[str, str]]:
    rows = read_csv_rows(path)
    if last_n > 0:
        return rows[-int(last_n):]
    return rows


def _mean_from_rows(rows: Sequence[Dict[str, str]], key: str) -> float:
    vals = []
    for row in rows:
        text = row.get(key, "")
        if text in ("", "nan", "NaN", "None", None):
            continue
        vals.append(float(text))
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def collect_training_tail_stats(run_dir: Union[os.PathLike, str], train_mode: str, last_n: int = 1000) -> Dict[str, float]:
    run_path = Path(run_dir)
    rank_files = sorted(run_path.glob("metrics_rank*.csv"))
    stats = {
        "max_f_neg_last_1k_mean": float("nan"),
        "max_abs_chain_last_1k_mean": float("nan"),
        "fneg_stage2_over_stage3_last_1k_mean": float("nan"),
    }
    if not rank_files:
        return stats

    base_rows = _tail_rows(rank_files[0], last_n=last_n)
    if base_rows and "max_f_neg" in base_rows[0]:
        stats["max_f_neg_last_1k_mean"] = _mean_from_rows(base_rows, "max_f_neg")
    if base_rows and "max_abs_chain" in base_rows[0]:
        stats["max_abs_chain_last_1k_mean"] = _mean_from_rows(base_rows, "max_abs_chain")

    mode = str(train_mode).strip().lower()
    if mode == "single_pipe_emul" and base_rows:
        if "f_neg_stage2" in base_rows[0] and "f_neg_stage3" in base_rows[0]:
            s2 = _mean_from_rows(base_rows, "f_neg_stage2")
            s3 = _mean_from_rows(base_rows, "f_neg_stage3")
            if math.isfinite(s2) and math.isfinite(s3) and abs(s3) > 1e-12:
                stats["fneg_stage2_over_stage3_last_1k_mean"] = float(s2 / s3)
        return stats

    if mode != "pipe_strict":
        return stats

    per_stage: Dict[int, List[float]] = {}
    for path in rank_files:
        rows = _tail_rows(path, last_n=last_n)
        if not rows or "stage" not in rows[0] or "f_neg" not in rows[0]:
            continue
        try:
            stage = int(float(rows[0]["stage"]))
        except Exception:
            continue
        vals = []
        for row in rows:
            text = row.get("f_neg", "")
            if text in ("", "nan", "NaN", "None", None):
                continue
            vals.append(float(text))
        if vals:
            per_stage[stage] = vals
    if 2 in per_stage and 3 in per_stage:
        mean2 = sum(per_stage[2]) / len(per_stage[2])
        mean3 = sum(per_stage[3]) / len(per_stage[3])
        if abs(mean3) > 1e-12:
            stats["fneg_stage2_over_stage3_last_1k_mean"] = float(mean2 / mean3)
    return stats


def normalize_metrics_payload(metrics: Dict) -> Dict[str, float]:
    baseline_distinct = metrics.get("baseline_distinct_images") or {}
    return {
        "fid_inception": metrics.get("fid_inception_baseline_vs_real"),
        "fid_feature": metrics.get("fid_feature_baseline_vs_real"),
        "unique_ratio": baseline_distinct.get("unique_ratio"),
        "distinct_images": baseline_distinct.get("unique_images"),
        "diversity_trace": metrics.get("diversity_trace_baseline"),
    }


def pretty_walltime(seconds: Optional[float]) -> str:
    if seconds in (None, ""):
        return ""
    total = int(round(float(seconds)))
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def flatten_manifest_row(row: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out

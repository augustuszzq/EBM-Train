#!/usr/bin/env python3
"""Collect cumulative wall-clock/GPU-hour accounting for long-K runs.

Resume fragments reuse the same logical run directory. The trainer's
``wallclock_sec`` counter resets on each fragment, so endpoint comparisons must
sum fragment runtime rather than reading only the final metrics row.
"""

import argparse
import csv
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = PROJECT_DIR / "runs_long_k_scaling"
WALLCLOCK_RE = re.compile(r"wallclock_sec=([0-9]+(?:\.[0-9]+)?)")


def existing_path(value: str) -> Optional[Path]:
    if not value:
        return None
    candidates = [Path(value)]
    if value.startswith("/eagle/"):
        candidates.append(Path("/lus") / value.lstrip("/"))
    if value.startswith("/lus/eagle/"):
        candidates.append(Path(value.replace("/lus/eagle/", "/eagle/", 1)))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def latest_metric_row(metrics_path: Path) -> Optional[Dict[str, str]]:
    if not metrics_path.exists():
        return None
    last = None
    with metrics_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last = row
    return last


def parse_float(value: Optional[str], default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def sum_wallclock_segments_from_log(log_path: Optional[Path]) -> Tuple[float, int]:
    """Sum trainer wallclock segments in an appended PBS log.

    Each trainer process starts its own ``wallclock_sec`` counter. A sharp
    decrease indicates a new resume fragment. The summed segment maxima are the
    cumulative actual training runtime visible in the log.
    """
    if log_path is None or not log_path.exists():
        return 0.0, 0

    total = 0.0
    segments = 0
    current_max = None
    previous = None
    try:
        awk_script = (
            r'/wallclock_sec=/ {'
            r's=$0; sub(/^.*wallclock_sec=/,"",s); sub(/[^0-9.].*$/,"",s); v=s+0; '
            r'if (seen && v + 60 < prev) { total += max; seg += 1; max = v; } '
            r'else if (!seen) { max = v; seen = 1; } '
            r'else if (v > max) { max = v; } '
            r'prev = v;'
            r'} '
            r'END { if (seen) { total += max; seg += 1; } printf "%.6f %d\n", total, seg; }'
        )
        out = subprocess.check_output(["awk", awk_script, str(log_path)], stderr=subprocess.DEVNULL)
        parts = out.decode("utf-8", "ignore").strip().split()
        if len(parts) >= 2:
            return float(parts[0]), int(float(parts[1]))
    except Exception:
        pass

    with log_path.open(errors="ignore") as f:
        for line in f:
            match = WALLCLOCK_RE.search(line)
            if not match:
                continue
            value = float(match.group(1))
            if previous is not None and current_max is not None:
                if value + 60.0 < previous:
                    total += current_max
                    segments += 1
                    current_max = value
                else:
                    current_max = max(current_max, value)
            else:
                current_max = value
            previous = value
    if current_max is not None:
        total += current_max
        segments += 1
    return total, segments


def sum_attempt_timing(run_dir: Path) -> Tuple[float, int]:
    path = run_dir / "attempt_timing.jsonl"
    if not path.exists():
        return 0.0, 0
    total = 0.0
    count = 0
    seen = set()  # type: Set[Tuple[str, str, str]]
    with path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "finish":
                continue
            key = (
                str(row.get("pbs_jobid", "")),
                str(row.get("started_at_epoch", "")),
                str(row.get("finished_at_epoch", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            total += parse_float(row.get("walltime_seconds"))
            count += 1
    return total, count


def job_info_wallclock(run_dir: Path) -> float:
    path = run_dir / "job_info.json"
    if not path.exists():
        return 0.0
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return 0.0
    return parse_float(payload.get("train_walltime_seconds"))


def infer_target_steps(exp_id: str, row: Dict[str, str]) -> int:
    for key in ("steps", "horizon_steps"):
        value = row.get(key, "")
        if value:
            try:
                return int(float(value))
            except ValueError:
                pass
    if "s500k" in exp_id:
        return 500000
    if "s300k" in exp_id:
        return 300000
    if "s20k" in exp_id:
        return 20000
    return 0


def iter_manifest_rows(bundle: Path) -> Iterable[Dict[str, str]]:
    seen = set()
    for name in (
        "long_k_master_registry.csv",
        "long_k_l1_l2_submitted.csv",
        "long_k_l0_submitted.csv",
    ):
        for row in read_csv(bundle / "summaries" / name):
            exp_id = row.get("exp_id", "")
            run_dir = row.get("run_dir") or row.get("train_run_dir")
            if not exp_id or not run_dir:
                continue
            key = f"{exp_id}|{run_dir}"
            if key in seen:
                continue
            seen.add(key)
            yield row


def collect(bundle: Path) -> List[Dict[str, str]]:
    rows = []
    for row in iter_manifest_rows(bundle):
        exp_id = row.get("exp_id", "")
        run_dir_raw = row.get("run_dir") or row.get("train_run_dir") or ""
        run_dir_path = existing_path(run_dir_raw)
        if run_dir_path is None:
            continue
        log_path = existing_path(row.get("pbs_output_log", ""))
        log_sec, log_segments = sum_wallclock_segments_from_log(log_path)
        attempt_sec, attempt_count = sum_attempt_timing(run_dir_path)
        info_sec = job_info_wallclock(run_dir_path)

        # Sources overlap. Use the largest observed source as the cumulative
        # runtime estimate instead of summing sources together.
        wallclock_sec = max(log_sec, attempt_sec, info_sec)
        source = "pbs_log_segments"
        if attempt_sec >= log_sec and attempt_sec >= info_sec and attempt_sec > 0:
            source = "attempt_timing_jsonl"
        elif info_sec >= log_sec and info_sec >= attempt_sec and info_sec > 0:
            source = "job_info_json"
        elif wallclock_sec <= 0:
            source = "missing"

        metrics = latest_metric_row(run_dir_path / "metrics_rank0.csv")
        latest_step = int(float(metrics["step"])) if metrics and metrics.get("step") else -1
        target_steps = infer_target_steps(exp_id, row)
        world_size = int(float(row.get("world_size") or row.get("WORLD_SIZE") or 1))
        gpu_hours = wallclock_sec * float(world_size) / 3600.0
        completed = bool(target_steps and latest_step >= target_steps - 1)
        step_time_sec = wallclock_sec / float(latest_step + 1) if latest_step >= 0 and wallclock_sec > 0 else 0.0
        throughput_steps_per_hour = 3600.0 / step_time_sec if step_time_sec > 0 else 0.0

        rows.append(
            {
                "phase": row.get("phase", ""),
                "exp_id": exp_id,
                "train_mode": row.get("train_mode", ""),
                "K": row.get("K", ""),
                "steps": str(target_steps),
                "seed": row.get("seed", ""),
                "world_size": str(world_size),
                "latest_step": str(latest_step),
                "completed": "yes" if completed else "no",
                "cumulative_wallclock_sec": f"{wallclock_sec:.3f}",
                "cumulative_gpu_hours": f"{gpu_hours:.6f}",
                "step_time_sec_cumulative": f"{step_time_sec:.6f}",
                "throughput_steps_per_hour": f"{throughput_steps_per_hour:.3f}",
                "runtime_source": source,
                "pbs_log_segments": str(log_segments),
                "pbs_log_wallclock_sec": f"{log_sec:.3f}",
                "attempt_timing_count": str(attempt_count),
                "attempt_timing_wallclock_sec": f"{attempt_sec:.3f}",
                "job_info_wallclock_sec": f"{info_sec:.3f}",
                "run_dir": str(run_dir_path),
                "pbs_output_log": str(log_path) if log_path else "",
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    fields = [
        "phase",
        "exp_id",
        "train_mode",
        "K",
        "steps",
        "seed",
        "world_size",
        "latest_step",
        "completed",
        "cumulative_wallclock_sec",
        "cumulative_gpu_hours",
        "step_time_sec_cumulative",
        "throughput_steps_per_hour",
        "runtime_source",
        "pbs_log_segments",
        "pbs_log_wallclock_sec",
        "attempt_timing_count",
        "attempt_timing_wallclock_sec",
        "job_info_wallclock_sec",
        "run_dir",
        "pbs_output_log",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    args = parser.parse_args()
    bundle = Path(args.bundle)
    rows = collect(bundle)
    out = bundle / "summaries" / "long_k_wallclock_cumulative.csv"
    write_csv(out, rows)
    # Keep the originally promised filename updated with cumulative semantics.
    write_csv(bundle / "summaries" / "long_k_wallclock_summary.csv", rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()

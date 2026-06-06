#!/usr/bin/env python3
"""Build a replay manifest for pipeline-then-weighting trajectory evaluation."""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

try:
    from ablation_replay_common import build_replay_rows, write_rows
except Exception:
    from polaris_ebm.scripts.current.ablation_replay_common import build_replay_rows, write_rows


PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
DEFAULT_OUT = PROJECT_DIR / "experiments" / "pipeline_then_weighting_replay_manifest.csv"


def load_rows(path: Path) -> List[Dict[str, str]]:
    import csv

    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def train_done(run_dir: Path) -> bool:
    log = run_dir / "train.log"
    return log.exists() and "[DONE]" in log.read_text(errors="ignore")


def merge_source_rows(
    pbs_rows: Iterable[Dict[str, str]],
    local_rows: Iterable[Dict[str, str]],
) -> List[Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {}
    for row in pbs_rows:
        run_dir = Path(row.get("train_run_dir", ""))
        if run_dir and train_done(run_dir):
            merged[row["exp_id"]] = dict(row)
    for row in local_rows:
        run_dir = Path(row.get("train_run_dir", ""))
        if run_dir and train_done(run_dir):
            merged[row["exp_id"]] = dict(row)
    return [merged[key] for key in sorted(merged)]


def build_rows(
    pbs_rows: Iterable[Dict[str, str]],
    local_rows: Iterable[Dict[str, str]],
) -> List[Dict]:
    source_rows = merge_source_rows(pbs_rows, local_rows)
    overrides = {
        row["exp_id"]: str(Path(row["train_run_dir"]) / "trajectory")
        for row in source_rows
    }
    replay_rows = build_replay_rows(source_rows, overrides=overrides)
    for row in replay_rows:
        row["replay_queue"] = "local"
        row["replay_walltime"] = ""
        row["replay_job_id"] = ""
    return replay_rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Build pipeline-then-weighting replay manifest")
    ap.add_argument(
        "--pbs-submitted",
        default=str(PROJECT_DIR / "experiments" / "pipeline_then_weighting_manifest_submitted.csv"),
    )
    ap.add_argument(
        "--local-submitted",
        default=str(PROJECT_DIR / "experiments" / "pipeline_then_weighting_local_submitted.csv"),
    )
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_rows(load_rows(Path(args.pbs_submitted)), load_rows(Path(args.local_submitted)))
    write_rows(Path(args.out), rows, list(rows[0].keys()) if rows else [])
    print(f"[DONE] wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

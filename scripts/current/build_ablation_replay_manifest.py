#!/usr/bin/env python3
"""Build a manifest of completed ablation runs that need replay evaluation."""

import argparse
from pathlib import Path

try:
    from ablation_replay_common import build_replay_rows, existing_trajectory_overrides, load_rows, write_rows
except Exception:
    from polaris_ebm.scripts.current.ablation_replay_common import build_replay_rows, existing_trajectory_overrides, load_rows, write_rows


PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
DEFAULT_MANIFEST = PROJECT_DIR / "experiments" / "ablation_replay_manifest.csv"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Build ablation replay manifest")
    ap.add_argument("--phase1-submitted", default=str(PROJECT_DIR / "experiments" / "ablation_manifest_phase1_submitted.csv"))
    ap.add_argument("--phase2-submitted", default=str(PROJECT_DIR / "experiments" / "ablation_manifest_phase2_submitted.csv"))
    ap.add_argument("--out", default=str(DEFAULT_MANIFEST))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    source_rows = load_rows(Path(args.phase1_submitted)) + load_rows(Path(args.phase2_submitted))
    replay_rows = build_replay_rows(source_rows, overrides=existing_trajectory_overrides())
    fieldnames = list(replay_rows[0].keys()) if replay_rows else []
    write_rows(Path(args.out), replay_rows, fieldnames)
    print(f"[DONE] wrote {len(replay_rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

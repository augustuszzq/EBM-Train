#!/usr/bin/env python3
"""Build a replay manifest for the submitted 500k runs."""

from pathlib import Path

try:
    from ablation_replay_common import build_replay_rows, load_rows, write_rows
except Exception:
    from polaris_ebm.scripts.current.ablation_replay_common import build_replay_rows, load_rows, write_rows


PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
SUBMITTED = PROJECT_DIR / "experiments" / "ablation_manifest_500k_submitted.csv"
OUT = PROJECT_DIR / "experiments" / "ablation_manifest_500k_replay.csv"
REPLAY_ROOT = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation_500k/trajectory_replay")


def main() -> int:
    rows = load_rows(SUBMITTED)
    replay_rows = build_replay_rows(rows, replay_root=REPLAY_ROOT, overrides={})
    fieldnames = list(replay_rows[0].keys()) if replay_rows else []
    write_rows(OUT, replay_rows, fieldnames)
    print(f"[DONE] wrote {len(replay_rows)} rows to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

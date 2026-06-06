#!/usr/bin/env python3
"""Collect 500k replay trajectories into point/summary/table artifacts."""

import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")


def main() -> int:
    cmd = [
        sys.executable,
        str(PROJECT_DIR / "scripts" / "current" / "collect_ablation_replay.py"),
        "--submitted_manifest",
        str(PROJECT_DIR / "experiments" / "ablation_manifest_500k_replay_submitted.csv"),
        "--out_dir",
        "/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation_500k",
    ]
    cmd.extend(sys.argv[1:])
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

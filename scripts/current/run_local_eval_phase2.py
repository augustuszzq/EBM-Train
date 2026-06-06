#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")


def main() -> int:
    cmd = [
        sys.executable,
        str(PROJECT_DIR / "scripts" / "current" / "run_local_eval_phase1.py"),
        "--submitted-manifest",
        str(PROJECT_DIR / "experiments" / "ablation_manifest_phase2_submitted.csv"),
    ]
    cmd.extend(sys.argv[1:])
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

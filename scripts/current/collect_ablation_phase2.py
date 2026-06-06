#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")


def main() -> int:
    cmd = [
        sys.executable,
        str(PROJECT_DIR / "scripts" / "current" / "collect_ablation_phase1.py"),
        "--submitted_manifest",
        str(PROJECT_DIR / "experiments" / "ablation_manifest_phase2_submitted.csv"),
        "--out_dir",
        "/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation_phase2",
        "--summary-prefix",
        "phase2_summary",
        "--render-script",
        str(PROJECT_DIR / "scripts" / "current" / "render_ablation_phase2_report.py"),
    ]
    cmd.extend(sys.argv[1:])
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

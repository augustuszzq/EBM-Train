#!/usr/bin/env python3
"""Run the current-CIFAR manual trajectory backfill with two concurrent lanes."""

import os
import subprocess
import sys
import time
from pathlib import Path


PYTHON = "/home/kevienzzq/.conda/envs/llm-env/bin/python"
PROJECT = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
EVAL = PROJECT / "scripts" / "current" / "eval_fid_trajectory.py"


TARGETS = [
    {
        "gpu": "0",
        "seed": "1",
        "label": "current_cifar_paper:cifar10:single_P8_equal:seed1:h300000",
        "run_dir": "/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_objective_first_followup/O1/O1_single_pipe_emul_P8_uniform_K100_seed1_300k_resume_20260421_223303/trajectory_merged_run",
        "out_dir": "/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_objective_first_followup/O1/O1_single_pipe_emul_P8_uniform_K100_seed1_300k_resume_20260421_223303/trajectory",
    },
    {
        "gpu": "1",
        "seed": "3",
        "label": "current_cifar_paper:cifar10:single_P1_terminal:seed3:h300000",
        "run_dir": "/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_objective_first_followup/O3/O3_single_fullk_K100_seed3_300k_resume_20260421_223303/trajectory_merged_run",
        "out_dir": "/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_objective_first_followup/O3/O3_single_fullk_K100_seed3_300k_resume_20260421_223303/trajectory",
    },
]


def launch(target):
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": target["gpu"],
            "PYTHONPATH": "/eagle/lc-mpi/Zhiqing" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    out_dir = Path(target["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "two_lane_backfill.log"
    cmd = [
        PYTHON,
        str(EVAL),
        "--run-dir",
        target["run_dir"],
        "--label",
        target["label"],
        "--out-dir",
        target["out_dir"],
        "--step-interval",
        "5000",
        "--eval-images",
        "5000",
        "--device",
        "cuda",
        "--eval-seed",
        target["seed"],
        "--skip-existing",
    ]
    log = log_path.open("ab")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
    return proc, log


def main():
    queue = list(TARGETS)
    active = {}
    while queue or active:
        for gpu in ["0", "1"]:
            if gpu in active or not queue:
                continue
            idx = next((i for i, target in enumerate(queue) if target["gpu"] == gpu), None)
            if idx is None:
                continue
            target = queue.pop(idx)
            proc, log = launch(target)
            active[gpu] = (proc, log, target)
            print(f"[LAUNCH] gpu={gpu} pid={proc.pid} label={target['label']}", flush=True)
        time.sleep(10)
        for gpu, (proc, log, target) in list(active.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log.close()
            print(f"[DONE] gpu={gpu} rc={rc} label={target['label']}", flush=True)
            if rc != 0:
                return rc
            del active[gpu]
    print("[DONE] two-lane backfill complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

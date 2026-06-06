#!/usr/bin/env bash
set -euo pipefail

cd /lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm

export PYTHONPATH="/eagle/lc-mpi/Zhiqing${PYTHONPATH:+:${PYTHONPATH}}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BACKFILL_GPUS="${BACKFILL_GPUS:-0,1}"

PY=/home/kevienzzq/.conda/envs/llm-env/bin/python

"${PY}" scripts/current/run_blocked_current_cifar_fid_backfill.py
"${PY}" scripts/current/build_master_experiment_registry.py
"${PY}" scripts/current/collect_current_cifar_fid_vs_step.py

#!/usr/bin/env bash
set -euo pipefail

PY=/home/kevienzzq/.conda/envs/llm-env/bin/python
RUNNER=/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/backfill_current_cifar_trajectories.py
STATE_BASE=/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_final_bundle

RANK="${PMI_RANK:?PMI_RANK is required}"
LOCAL_RANK="${PMI_LOCAL_RANK:?PMI_LOCAL_RANK is required}"
WORLD="${PMI_SIZE:?PMI_SIZE is required}"

exec "${PY}" -u "${RUNNER}" \
  --rank-index "${RANK}" \
  --rank-count "${WORLD}" \
  --gpus 0,1,2,3 \
  --state-json "${STATE_BASE}/current_cifar_backfill_rank${RANK}.json"

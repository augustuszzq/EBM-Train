#!/usr/bin/env bash
set -euo pipefail

PY=/home/kevienzzq/.conda/envs/llm-env/bin/python
RUNNER=/eagle/lc-mpi/Zhiqing/polaris_ebm/scripts/current/run_objective_first_followup_final_eval.py

TARGETS=(
  O3_ddp_fullk_K100_seed2_300k
  O3_ddp_fullk_K100_seed3_300k
  O3_ddp_fullk_K100_seed4_300k
  O3_ddp_fullk_K100_seed5_300k
  O3_pipe_strict_P4_uniform_K100_seed2_300k
  O3_pipe_strict_P4_uniform_K100_seed3_300k
  O3_pipe_strict_P2_uniform_K100_seed2_300k
  O3_pipe_strict_P2_uniform_K100_seed3_300k
)

RANK="${PMI_RANK:?PMI_RANK is required}"
LOCAL_RANK="${PMI_LOCAL_RANK:?PMI_LOCAL_RANK is required}"
TARGET="${TARGETS[$RANK]:-}"

if [[ -z "${TARGET}" ]]; then
  echo "[ERROR] no target assigned for PMI_RANK=${RANK}" >&2
  exit 2
fi

exec "${PY}" -u "${RUNNER}" --gpus "${LOCAL_RANK}" --targets "${TARGET}"

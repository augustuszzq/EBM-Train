#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm}"
INTERVAL_SEC="${INTERVAL_SEC:-900}"
MAX_NEW_PER_TICK="${MAX_NEW_PER_TICK:-4}"
MAX_TICKS="${MAX_TICKS:-0}"
LOG_PATH="${LOG_PATH:-${PROJECT_DIR}/runs_long_k_scaling/logs/long_k_successor_watch.log}"
PID_PATH="${PID_PATH:-${PROJECT_DIR}/runs_long_k_scaling/logs/long_k_successor_watch.pid}"

mkdir -p "$(dirname "${LOG_PATH}")"
cd "${PROJECT_DIR}"

echo "$$" > "${PID_PATH}"
trap 'rm -f "${PID_PATH}"' EXIT

tick=0
while true; do
  tick=$((tick + 1))
  {
    echo "========== $(date -u '+%Y-%m-%dT%H:%M:%SZ') tick=${tick} =========="
    qstat -u "$(whoami)" || true
    python3 scripts/current/submit_long_k_resume_successors.py --max-new "${MAX_NEW_PER_TICK}"
  } >> "${LOG_PATH}" 2>&1

  if [[ "${MAX_TICKS}" != "0" && "${tick}" -ge "${MAX_TICKS}" ]]; then
    echo "========== $(date -u '+%Y-%m-%dT%H:%M:%SZ') max ticks reached ==========" >> "${LOG_PATH}"
    break
  fi

  sleep "${INTERVAL_SEC}"
done

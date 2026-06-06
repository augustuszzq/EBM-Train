#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm}"
INTERVAL_SEC="${INTERVAL_SEC:-1800}"
MAX_NEW_PER_TICK="${MAX_NEW_PER_TICK:-4}"
MAX_TICKS="${MAX_TICKS:-0}"
LOG_ROOT="${LOG_ROOT:-/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_imagenet32_drl_prime/logs}"
LOG_PATH="${LOG_PATH:-${LOG_ROOT}/prime20k_successor_watch.log}"
PID_PATH="${PID_PATH:-${LOG_ROOT}/prime20k_successor_watch.pid}"

mkdir -p "${LOG_ROOT}"
cd "${PROJECT_DIR}"

if [[ -f "${PID_PATH}" ]]; then
  old_pid="$(cat "${PID_PATH}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "watcher already running pid=${old_pid}"
    exit 0
  fi
fi

echo "$$" > "${PID_PATH}"
trap 'rm -f "${PID_PATH}"' EXIT

tick=0
while true; do
  tick=$((tick + 1))
  {
    echo "========== $(date -u '+%Y-%m-%dT%H:%M:%SZ') tick=${tick} =========="
    qstat -u "$(whoami)" || true
    python3 scripts/current/submit_imagenet32_drl_prime20k_successors.py --max-new "${MAX_NEW_PER_TICK}"
  } >> "${LOG_PATH}" 2>&1

  if [[ "${MAX_TICKS}" != "0" && "${tick}" -ge "${MAX_TICKS}" ]]; then
    echo "========== $(date -u '+%Y-%m-%dT%H:%M:%SZ') max ticks reached ==========" >> "${LOG_PATH}"
    break
  fi

  sleep "${INTERVAL_SEC}"
done

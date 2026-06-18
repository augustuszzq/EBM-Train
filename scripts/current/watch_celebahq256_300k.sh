#!/bin/bash
# Watch and resume CelebA-HQ 256 300k DDP/pipeline validation runs.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs_celebahq256_preflight/smoke}"
POLL_SECONDS="${POLL_SECONDS:-1800}"
QUEUE="${QUEUE:-capacity}"
WALLTIME="${WALLTIME:-72:00:00}"
MAX_NEW="${MAX_NEW:-2}"
PY="${PY:-/home/kevienzzq/.conda/envs/llm-env/bin/python}"

LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}" "${RUN_ROOT}/summaries"
WATCH_LOG="${WATCH_LOG:-${LOG_DIR}/celebahq256_300k_watch.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/celebahq256_300k_watch.pid}"
LOCK_DIR="${LOCK_DIR:-${LOG_DIR}/celebahq256_300k_watch.lock}"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  if [[ -f "${PID_FILE}" ]]; then
    old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
      echo "[CELEBA WATCH] already running pid=${old_pid}" | tee -a "${WATCH_LOG}"
      exit 0
    fi
  fi
  echo "[CELEBA WATCH] stale lock detected; replacing ${LOCK_DIR}" | tee -a "${WATCH_LOG}"
  rm -rf "${LOCK_DIR}"
  mkdir "${LOCK_DIR}"
fi

cleanup() {
  rm -rf "${LOCK_DIR}"
}
trap cleanup EXIT

echo "$$" > "${PID_FILE}"
cd "${PROJECT_DIR}"

echo "[CELEBA WATCH] start pid=$$ queue=${QUEUE} walltime=${WALLTIME} poll=${POLL_SECONDS}s" | tee -a "${WATCH_LOG}"

while true; do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[CELEBA WATCH] tick ${ts}" | tee -a "${WATCH_LOG}"
  {
    echo "[CELEBA WATCH] qstat snapshot ${ts}"
    qstat -u "${USER}" || true
    echo "[CELEBA WATCH] resume pass ${ts}"
    "${PY}" "${PROJECT_DIR}/scripts/current/submit_celebahq256_300k_resumes.py" \
      --queue "${QUEUE}" \
      --walltime "${WALLTIME}" \
      --max-new "${MAX_NEW}"
  } >> "${WATCH_LOG}" 2>&1
  sleep "${POLL_SECONDS}"
done

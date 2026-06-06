#!/bin/bash
# Watch and resume interrupted 300k continuations for the multi-node long-K sweep.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm}"
BUNDLE_ROOT="${BUNDLE_ROOT:-/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_long_k_multinode_sweep}"
POLL_SECONDS="${POLL_SECONDS:-1800}"
USER_NAME="${USER_NAME:-kevienzzq}"
PY="${PY:-python3}"

LOG_DIR="${BUNDLE_ROOT}/logs"
mkdir -p "${LOG_DIR}"
WATCH_LOG="${WATCH_LOG:-${LOG_DIR}/m1_300k_watch.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/m1_300k_watch.pid}"
LOCK_DIR="${LOCK_DIR:-${LOG_DIR}/m1_300k_watch.lock}"

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  if [[ -f "${PID_FILE}" ]]; then
    old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
      echo "[M1 WATCH] already running pid=${old_pid}" | tee -a "${WATCH_LOG}"
      exit 0
    fi
  fi
  echo "[M1 WATCH] stale lock detected; replacing ${LOCK_DIR}" | tee -a "${WATCH_LOG}"
  rm -rf "${LOCK_DIR}"
  mkdir "${LOCK_DIR}"
fi

cleanup() {
  rm -rf "${LOCK_DIR}"
}
trap cleanup EXIT

echo "$$" > "${PID_FILE}"
cd "${PROJECT_DIR}"

echo "[M1 WATCH] start pid=$$ bundle=${BUNDLE_ROOT} poll=${POLL_SECONDS}s" | tee -a "${WATCH_LOG}"

while true; do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[M1 WATCH] tick ${ts}" | tee -a "${WATCH_LOG}"
  {
    echo "[M1 WATCH] qstat snapshot ${ts}"
    qstat -u "${USER_NAME}" || true
    echo "[M1 WATCH] resume sweep ${ts}"
    "${PY}" "${PROJECT_DIR}/scripts/current/long_k_multinode_300k_continuations.py" \
      --resume-existing \
      --user "${USER_NAME}"
  } >> "${WATCH_LOG}" 2>&1
  sleep "${POLL_SECONDS}"
done

#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm}"
INTERVAL_SEC="${INTERVAL_SEC:-1800}"
MAX_TICKS="${MAX_TICKS:-0}"
LOG="${LOG:-/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_long_k_overhead_attribution/logs/overhead_attribution_submit_watch.log}"
PY="${PY:-/home/kevienzzq/.conda/envs/llm-env/bin/python}"

mkdir -p "$(dirname "${LOG}")"
cd "${PROJECT_DIR}"

tick=0
while true; do
  tick=$((tick + 1))
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[${ts}] tick=${tick} submit attempt" | tee -a "${LOG}"
  "${PY}" scripts/current/long_k_overhead_attribution_manifest.py --submit --idempotent 2>&1 | tee -a "${LOG}" || true
  set +e
  "${PY}" - <<'PY' | tee -a "${LOG}"
import csv
from pathlib import Path

path = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_long_k_overhead_attribution/summaries/overhead_attribution_submitted.csv")
rows = list(csv.DictReader(path.open())) if path.exists() else []
submitted = [r for r in rows if r.get("train_job_id")]
failed = [r for r in rows if r.get("status") == "qsub_failed"]
print(f"[summary] rows={len(rows)} submitted_or_recorded={len(submitted)} qsub_failed={len(failed)}")
if rows and len(submitted) == len(rows):
    print("[summary] all rows have job ids; watcher can exit")
    raise SystemExit(0)
raise SystemExit(3)
PY
  rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" -eq 0 ]]; then
    exit 0
  fi
  if [[ "${MAX_TICKS}" -gt 0 && "${tick}" -ge "${MAX_TICKS}" ]]; then
    echo "[watcher] reached MAX_TICKS=${MAX_TICKS}; exiting" | tee -a "${LOG}"
    exit 0
  fi
  sleep "${INTERVAL_SEC}"
done

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python3 "${SCRIPT_DIR}/ablation_500k_manifest.py" --out "${PROJECT_DIR}/experiments/ablation_manifest_500k.csv"

exec bash "${SCRIPT_DIR}/submit_ablation_phase1.sh" \
  --manifest "${PROJECT_DIR}/experiments/ablation_manifest_500k.csv" \
  --submitted-manifest "${PROJECT_DIR}/experiments/ablation_manifest_500k_submitted.csv" \
  --runs-root "/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation_500k" \
  --payload-subdir "500k" \
  --eval-mode "local" \
  "$@"

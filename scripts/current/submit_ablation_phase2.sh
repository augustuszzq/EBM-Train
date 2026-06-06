#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

exec bash "${SCRIPT_DIR}/submit_ablation_phase1.sh" \
  --manifest "${PROJECT_DIR}/experiments/ablation_manifest_phase2.csv" \
  --submitted-manifest "${PROJECT_DIR}/experiments/ablation_manifest_phase2_submitted.csv" \
  --runs-root "/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation_phase2" \
  --payload-subdir "phase2" \
  --eval-mode "local" \
  "$@"

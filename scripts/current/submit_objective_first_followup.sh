#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/submit_pipeline_then_weighting.sh" \
  --manifest "/eagle/lc-mpi/Zhiqing/polaris_ebm/experiments/objective_first_followup_manifest.csv" \
  --submitted-manifest "/eagle/lc-mpi/Zhiqing/polaris_ebm/experiments/objective_first_followup_manifest_submitted.csv" \
  --runs-root "/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_objective_first_followup" \
  --payload-subdir "objective_first_followup" \
  --eval-mode "local" \
  --max-live-train-jobs "15" \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f "${PROJECT_DIR}/.env.drl" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_DIR}/.env.drl"
fi

PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data/imagenet32}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${PROJECT_DIR}/data/hf_cache}"
TRAIN_MAX_EXAMPLES="${TRAIN_MAX_EXAMPLES:-0}"
VAL_MAX_EXAMPLES="${VAL_MAX_EXAMPLES:-0}"
STREAMING="${STREAMING:-0}"

cd "${PROJECT_DIR}"
mkdir -p "${DATA_DIR}" "${HF_CACHE_DIR}"

streaming_arg=()
if [[ "${STREAMING}" == "1" || "${STREAMING}" == "true" ]]; then
  streaming_arg=(--streaming)
fi

echo "[DATA] materializing train split to ${DATA_DIR}"
PYTHONPATH=. "${PYTHON_BIN}" scripts/current/export_hf_imagenet32.py \
  --out-root "${DATA_DIR}" \
  --split train \
  --cache-dir "${HF_CACHE_DIR}" \
  --max-examples "${TRAIN_MAX_EXAMPLES}" \
  "${streaming_arg[@]}"

echo "[DATA] materializing val split to ${DATA_DIR}"
PYTHONPATH=. "${PYTHON_BIN}" scripts/current/export_hf_imagenet32.py \
  --out-root "${DATA_DIR}" \
  --split val \
  --cache-dir "${HF_CACHE_DIR}" \
  --max-examples "${VAL_MAX_EXAMPLES}" \
  "${streaming_arg[@]}"

echo "[DONE] ImageNet-32 materialized at ${DATA_DIR}"

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
RUN_DIR="${DRL_EVAL_RUN_DIR:-${1:-}}"
CKPT="${DRL_EVAL_CKPT:-${2:-}}"
NUM_IMAGES="${DRL_EVAL_IMAGES:-5000}"
IMAGES_PER_CLASS="${DRL_EVAL_IMAGES_PER_CLASS:-0}"
NUM_REAL="${DRL_EVAL_NUM_REAL:-5000}"
BATCH="${DRL_EVAL_BATCH:-128}"
K_EVAL="${DRL_EVAL_K:-100}"
STEP_SIZE="${DRL_EVAL_STEP_SIZE:-5e-4}"
NOISE_STD="${DRL_EVAL_NOISE_STD:-5e-4}"
LANGEVIN_SIGN="${DRL_EVAL_LANGEVIN_SIGN:--1.0}"
SEED="${DRL_EVAL_SEED:-1}"
SKIP_STANDARD_FID="${DRL_SKIP_STANDARD_FID:-1}"

cd "${PROJECT_DIR}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "[ERROR] usage: DRL_EVAL_RUN_DIR=/path/to/run bash scripts/current/eval_imagenet32_drl_local.sh"
  echo "        or: bash scripts/current/eval_imagenet32_drl_local.sh /path/to/run [checkpoint.pt]"
  exit 2
fi

if [[ -z "${CKPT}" ]]; then
  CKPT="$(find "${RUN_DIR}/checkpoints" -maxdepth 1 -type f -name '*.pt' | sort -V | tail -1)"
fi
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "[ERROR] checkpoint not found for run ${RUN_DIR}"
  exit 2
fi

CONFIG="${DRL_EVAL_CONFIG:-${RUN_DIR}/config_local.yaml}"
if [[ ! -f "${CONFIG}" ]]; then
  echo "[ERROR] config not found: ${CONFIG}"
  exit 2
fi

EVAL_DIR="${DRL_EVAL_DIR:-${RUN_DIR}/eval}"
mkdir -p "${EVAL_DIR}/samples"

echo "[EVAL] ckpt=${CKPT}"
echo "[EVAL] config=${CONFIG}"
echo "[EVAL] data_dir=${DATA_DIR}"
echo "[EVAL] num_images=${NUM_IMAGES} num_real=${NUM_REAL}"

PYTHONPATH=. "${PYTHON_BIN}" scripts/current/eval_generate_imagenet32_drl.py \
  --ckpt "${CKPT}" \
  --config "${CONFIG}" \
  --out_dir "${EVAL_DIR}/samples" \
  --num_images "${NUM_IMAGES}" \
  --images_per_class "${IMAGES_PER_CLASS}" \
  --batch "${BATCH}" \
  --K_eval "${K_EVAL}" \
  --langevin_sign "${LANGEVIN_SIGN}" \
  --step_size "${STEP_SIZE}" \
  --noise_std "${NOISE_STD}" \
  --seed "${SEED}"

fid_args=()
if [[ "${SKIP_STANDARD_FID}" == "1" || "${SKIP_STANDARD_FID}" == "true" ]]; then
  fid_args=(--skip_standard_fid)
fi

PYTHONPATH=. "${PYTHON_BIN}" scripts/current/eval_imagenet32_fid.py \
  --samples "${EVAL_DIR}/samples/samples.pt" \
  --config "${CONFIG}" \
  --data_dir "${DATA_DIR}" \
  --num_real "${NUM_REAL}" \
  --batch "${BATCH}" \
  --out_json "${EVAL_DIR}/metrics_compare.json" \
  "${fid_args[@]}"

echo "[DONE] eval outputs in ${EVAL_DIR}"

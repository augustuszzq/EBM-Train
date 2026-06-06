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
TORCHRUN_BIN="${TORCHRUN:-torchrun}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data/imagenet32}"
TRAIN_MODE="${DRL_TRAIN_MODE:-ddp_fullk}"
WORLD_SIZE="${WORLD_SIZE:-1}"
PIPE_STAGES="${DRL_PIPE_STAGES:-4}"
WEIGHT_MODE="${DRL_WEIGHT_MODE:-uniform}"
LAST2_BETA="${DRL_LAST2_BETA:-0.01}"
STEPS="${DRL_STEPS:-2000}"
BATCH_SIZE="${DRL_BATCH_SIZE:-128}"
K="${DRL_K:-100}"
LR="${DRL_LR:-1e-7}"
LR_WARMUP_STEPS="${DRL_LR_WARMUP_STEPS:-10000}"
STEP_SIZE="${DRL_STEP_SIZE:-5e-4}"
NOISE_STD="${DRL_NOISE_STD:-5e-4}"
GRAD_CLIP_NORM="${DRL_GRAD_CLIP_NORM:-1.0}"
SAMPLER_PRIME="${DRL_SAMPLER_PRIME:-0.0}"
SAMPLER_TEMPERATURE="${DRL_SAMPLER_TEMPERATURE:-1.0}"
ENERGY_LOSS_SCALE="${DRL_ENERGY_LOSS_SCALE:-1.0}"
SAVE_EVERY="${DRL_SAVE_EVERY:-1000}"
VIS_EVERY="${DRL_VIS_EVERY:-1000}"
LOG_EVERY="${DRL_LOG_EVERY:-50}"
SEED="${DRL_SEED:-1}"
CONFIG_TEMPLATE="${DRL_CONFIG_TEMPLATE:-configs/imagenet32_drl_port_sanity_ddp_2k.yaml}"
RUN_ROOT="${DRL_RUN_ROOT:-${PROJECT_DIR}/runs_imagenet32_drl_local}"
EXP_ID="${DRL_EXP_ID:-imagenet32_drl_${TRAIN_MODE}_smoke_seed${SEED}}"
DRY_RUN="${DRY_RUN:-0}"

cd "${PROJECT_DIR}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

if [[ ! -f "${DATA_DIR}/train/labels.csv" ]]; then
  echo "[ERROR] missing ${DATA_DIR}/train/labels.csv"
  echo "[HINT] set DATA_DIR in .env.drl or run: scripts/current/materialize_imagenet32_local.sh"
  exit 2
fi

timestamp="$(date -u +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_ROOT}/${EXP_ID}_${timestamp}"
mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/vis"
CONFIG_LOCAL="${RUN_DIR}/config_local.yaml"

PYTHONPATH=. DATA_DIR="${DATA_DIR}" CONFIG_TEMPLATE="${CONFIG_TEMPLATE}" CONFIG_LOCAL="${CONFIG_LOCAL}" \
  "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import yaml

src = Path(os.environ["CONFIG_TEMPLATE"])
dst = Path(os.environ["CONFIG_LOCAL"])
cfg = yaml.safe_load(src.read_text()) or {}
cfg.setdefault("benchmark", {})["data_root"] = os.environ["DATA_DIR"]
dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

common_args=(
  --config "${CONFIG_LOCAL}"
  --data_dir "${DATA_DIR}"
  --run_dir "${RUN_DIR}"
  --save_dir "${RUN_DIR}/checkpoints"
  --vis_dir "${RUN_DIR}/vis"
  --steps "${STEPS}"
  --save_every "${SAVE_EVERY}"
  --vis_every "${VIS_EVERY}"
  --K "${K}"
  --batch_size "${BATCH_SIZE}"
  --lr "${LR}"
  --lr_warmup_steps "${LR_WARMUP_STEPS}"
  --step_size "${STEP_SIZE}"
  --noise_std "${NOISE_STD}"
  --sampler_prime "${SAMPLER_PRIME}"
  --sampler_temperature "${SAMPLER_TEMPERATURE}"
  --energy_loss_scale "${ENERGY_LOSS_SCALE}"
  --grad_clip_norm "${GRAD_CLIP_NORM}"
  --seed "${SEED}"
  --fresh_init
  --sync_fresh_init
  --log_every "${LOG_EVERY}"
  --diagnostic_first_steps 200
  --diagnostic_log_every 50
)

case "${TRAIN_MODE}" in
  ddp_fullk)
    train_args=(--mode ddp_fullk "${common_args[@]}")
    ;;
  pipe_strict|pipeline)
    train_args=(
      --mode pipeline
      "${common_args[@]}"
      --pipe_group_scope world
      --pipe_stages "${PIPE_STAGES}"
      --weight_mode "${WEIGHT_MODE}"
      --last2_beta "${LAST2_BETA}"
      --skip_optimizer_until_full_diagonal
    )
    ;;
  *)
    echo "[ERROR] unsupported DRL_TRAIN_MODE=${TRAIN_MODE}; use ddp_fullk or pipe_strict"
    exit 2
    ;;
esac

echo "[RUN] PROJECT_DIR=${PROJECT_DIR}"
echo "[RUN] DATA_DIR=${DATA_DIR}"
echo "[RUN] RUN_DIR=${RUN_DIR}"
echo "[RUN] TRAIN_MODE=${TRAIN_MODE} WORLD_SIZE=${WORLD_SIZE}"

cmd=(
  "${TORCHRUN_BIN}"
  --standalone
  --nproc_per_node="${WORLD_SIZE}"
  "${PROJECT_DIR}/scripts/current/ebm_train_imagenet32_drl_sync.py"
  "${train_args[@]}"
)

printf '[TRAIN CMD] '
printf '%q ' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
  echo "[DRY_RUN] command generated but not executed"
  exit 0
fi

"${cmd[@]}" 2>&1 | tee "${RUN_DIR}/train.log"

echo "[DONE] ${RUN_DIR}"

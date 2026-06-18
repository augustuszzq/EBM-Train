#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm}"
cd "${PROJECT_DIR}"

STAMP="$(date -u +%Y%m%d_%H%M%S)"
ROOT="${PROJECT_DIR}/runs_celebahq256_preflight/fsdp_debug"
PBS_SCRIPT="${PROJECT_DIR}/scripts/current/pbs_run_ablation_case.pbs"
DATA_DIR="${DATA_DIR:-/eagle/lc-mpi/Zhiqing/polaris_ebm/data/celeba/celeba-lmdb}"

DDP_QUEUE="${DDP_QUEUE:-debug}"
PIPE_QUEUE="${PIPE_QUEUE:-debug-scaling}"
WALLTIME="${WALLTIME:-01:00:00}"
ONLY="${ONLY:-both}"
STEPS="${STEPS:-50}"
K="${K:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
N_F="${N_F:-128}"
LR="${LR:-5e-7}"
STEP_SIZE="${STEP_SIZE:-0.001}"
NOISE_STD="${NOISE_STD:-0.005}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-10.0}"
SAVE_EVERY="${SAVE_EVERY:-0}"
VIS_EVERY="${VIS_EVERY:-0}"
LOG_EVERY="${LOG_EVERY:-1}"

mkdir -p "${ROOT}"
if [[ "${ONLY}" != "both" && "${ONLY}" != "ddp" && "${ONLY}" != "pipe" ]]; then
  echo "[ERROR] ONLY must be one of: both, ddp, pipe" >&2
  exit 2
fi

SUBMITTED="${ROOT}/submitted_${STAMP}.csv"
printf 'exp_id,train_mode,job_id,run_dir,manifest_row_path,submitted_at,queue,walltime,n_f,parallel_backend\n' > "${SUBMITTED}"

submit_one() {
  local exp_id="$1"
  local train_mode="$2"
  local pipe_stages="$3"
  local weight_mode="$4"
  local config="$5"
  local queue="$6"
  local run_dir="${ROOT}/${exp_id}_${STAMP}"
  local manifest_row="${run_dir}/manifest_row_source.json"
  mkdir -p "${run_dir}"

  MANIFEST_ROW="${manifest_row}" RUN_DIR="${run_dir}" EXP_ID="${exp_id}" TRAIN_MODE="${train_mode}" \
  PIPE_STAGES="${pipe_stages}" WEIGHT_MODE="${weight_mode}" CONFIG="${config}" DATA_DIR="${DATA_DIR}" \
  STEPS="${STEPS}" K="${K}" BATCH_SIZE="${BATCH_SIZE}" N_F="${N_F}" LR="${LR}" STEP_SIZE="${STEP_SIZE}" \
  NOISE_STD="${NOISE_STD}" MAX_GRAD_NORM="${MAX_GRAD_NORM}" \
  /home/kevienzzq/.conda/envs/llm-env/bin/python - <<'PY'
import json
import os
import time

payload = {
    "benchmark": "celebahq256",
    "group": "celebahq256_fsdp_debug",
    "exp_id": os.environ["EXP_ID"],
    "story": "CelebA-HQ 256 larger-model FSDP debug; no paper metric claim",
    "train_mode": os.environ["TRAIN_MODE"],
    "world_size": 4,
    "pipe_stages": int(os.environ["PIPE_STAGES"]),
    "K": int(os.environ["K"]),
    "steps": int(os.environ["STEPS"]),
    "lr": float(os.environ["LR"]),
    "step_size": float(os.environ["STEP_SIZE"]),
    "noise_std": float(os.environ["NOISE_STD"]),
    "max_grad_norm": float(os.environ["MAX_GRAD_NORM"]),
    "weight_mode": os.environ["WEIGHT_MODE"],
    "last2_beta": 0.01,
    "seed": 1,
    "num_nodes": 1,
    "ppn": 4,
    "batch_size": int(os.environ["BATCH_SIZE"]),
    "n_f": int(os.environ["N_F"]),
    "parallel_backend": "fsdp",
    "save_every": 0,
    "vis_every": 0,
    "data_dir": os.environ["DATA_DIR"],
    "config": os.environ["CONFIG"],
    "submitted_at_epoch": int(time.time()),
}
with open(os.environ["MANIFEST_ROW"], "w") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
PY

  local qsub_vars
  qsub_vars=$(
    printf 'PROJECT_DIR=%s,GROUP=%s,EXP_ID=%s,STORY=%s,TRAIN_MODE=%s,WORLD_SIZE=%s,PIPE_STAGES=%s,K=%s,STEPS=%s,LR=%s,STEP_SIZE=%s,NOISE_STD=%s,WEIGHT_MODE=%s,LAST2_BETA=%s,SEED=%s,NUM_NODES=%s,PPN=%s,EVAL_IMAGES=%s,RUN_DIR=%s,MANIFEST_ROW_PATH=%s,SAVE_EVERY=%s,VIS_EVERY=%s,BATCH_SIZE=%s,N_F=%s,SIGMA_PD=%s,LANGEVIN_SIGN=%s,WEIGHT_DECAY=%s,MAX_GRAD_NORM=%s,PARALLEL_BACKEND=%s,DATA_DIR=%s,CONFIG=%s,SYNC_FRESH_INIT=%s,DEBUG_LEVEL=%s,LOG_EVERY=%s' \
      "${PROJECT_DIR}" \
      "celebahq256_fsdp_debug" \
      "${exp_id}" \
      "CelebA-HQ256 n_f=${N_F} FSDP debug" \
      "${train_mode}" \
      "4" \
      "${pipe_stages}" \
      "${K}" \
      "${STEPS}" \
      "${LR}" \
      "${STEP_SIZE}" \
      "${NOISE_STD}" \
      "${weight_mode}" \
      "0.01" \
      "1" \
      "1" \
      "4" \
      "0" \
      "${run_dir}" \
      "${manifest_row}" \
      "${SAVE_EVERY}" \
      "${VIS_EVERY}" \
      "${BATCH_SIZE}" \
      "${N_F}" \
      "0.0" \
      "-1.0" \
      "0.0" \
      "${MAX_GRAD_NORM}" \
      "fsdp" \
      "${DATA_DIR}" \
      "${config}" \
      "1" \
      "1" \
      "${LOG_EVERY}"
  )

  local job_id
  job_id="$(qsub -q "${queue}" -l select=1:system=polaris -l walltime="${WALLTIME}" -N "chqfsdp" -v "${qsub_vars}" "${PBS_SCRIPT}" | awk '{print $1}')"
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "${exp_id}" "${train_mode}" "${job_id}" "${run_dir}" "${manifest_row}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${queue}" "${WALLTIME}" "${N_F}" "fsdp" >> "${SUBMITTED}"
  printf '[SUBMITTED] %s %s %s %s\n' "${job_id}" "${train_mode}" "${exp_id}" "${run_dir}"
}

if [[ "${ONLY}" == "both" || "${ONLY}" == "ddp" ]]; then
  submit_one \
    "celebahq256_nf${N_F}_fsdp_ddp_k${K}_s${STEPS}_seed1" \
    "ddp_fullk" \
    "1" \
    "deep_only" \
    "${PROJECT_DIR}/configs/celebahq256_ddp_strict.yaml" \
    "${DDP_QUEUE}"
fi

if [[ "${ONLY}" == "both" || "${ONLY}" == "pipe" ]]; then
  submit_one \
    "celebahq256_nf${N_F}_fsdp_pipe_p4_k${K}_s${STEPS}_seed1" \
    "pipe_strict" \
    "4" \
    "uniform" \
    "${PROJECT_DIR}/configs/celebahq256_pipeline_strict.yaml" \
    "${PIPE_QUEUE}"
fi

printf '[SUBMITTED_CSV] %s\n' "${SUBMITTED}"

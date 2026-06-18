#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm}"
cd "${PROJECT_DIR}"

STAMP="$(date -u +%Y%m%d_%H%M%S)"
ROOT="${PROJECT_DIR}/runs_celebahq256_latent"
TRAIN_ROOT="${ROOT}/train"
EXPORT_ROOT="${ROOT}/export"
LATENT_ROOT="${LATENT_ROOT:-/eagle/lc-mpi/Zhiqing/polaris_ebm/data/celeba/celeba-latents/sd-vae-ft-ema-scale018215}"
PIXEL_DATA_ROOT="${PIXEL_DATA_ROOT:-/eagle/lc-mpi/Zhiqing/polaris_ebm/data/celeba/celeba-lmdb}"
PBS_TRAIN="${PROJECT_DIR}/scripts/current/pbs_run_ablation_case.pbs"
PBS_EXPORT="${PROJECT_DIR}/scripts/current/pbs_export_celebahq256_latents.pbs"

PBS_ACCOUNT="${PBS_ACCOUNT:-sbi-fair}"
PBS_FILESYSTEMS="${PBS_FILESYSTEMS:-home:eagle}"
EXPORT_QUEUE="${EXPORT_QUEUE:-preemptable}"
TRAIN_QUEUE="${TRAIN_QUEUE:-preemptable}"
EXPORT_WALLTIME="${EXPORT_WALLTIME:-12:00:00}"
TRAIN_WALLTIME="${TRAIN_WALLTIME:-72:00:00}"
STEPS="${STEPS:-300000}"
K="${K:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
N_F="${N_F:-64}"
LR="${LR:-1e-4}"
STEP_SIZE="${STEP_SIZE:-0.01}"
NOISE_STD="${NOISE_STD:-0.01}"
LANGEVIN_SIGN="${LANGEVIN_SIGN:-1.0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.0}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
VIS_EVERY="${VIS_EVERY:-0}"
LOG_EVERY="${LOG_EVERY:-10}"
SEED="${SEED:-1}"
EXPORT_BATCH="${EXPORT_BATCH:-64}"
EXPORT_LIMIT="${EXPORT_LIMIT:-0}"
EXPORT_FORCE="${EXPORT_FORCE:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

mkdir -p "${TRAIN_ROOT}" "${EXPORT_ROOT}"
SUBMITTED="${ROOT}/submitted_${STAMP}.csv"
printf 'kind,exp_id,train_mode,job_id,run_dir,manifest_row_path,submitted_at,queue,walltime,dependency\n' > "${SUBMITTED}"

needs_export=0
for split in train validation; do
  if [[ ! -f "${LATENT_ROOT}/${split}/latents.pt" || ! -f "${LATENT_ROOT}/${split}/_SUCCESS.json" ]]; then
    needs_export=1
  fi
done

export_job_id=""
if [[ "${needs_export}" == "1" || "${EXPORT_FORCE}" == "1" ]]; then
  export_dir="${EXPORT_ROOT}/export_${STAMP}"
  mkdir -p "${export_dir}"
  export_job_id="$(
    qsub -A "${PBS_ACCOUNT}" \
      -q "${EXPORT_QUEUE}" \
      -l select=1:system=polaris \
      -l filesystems="${PBS_FILESYSTEMS}" \
      -l walltime="${EXPORT_WALLTIME}" \
      -N "chqlatex" \
      -o "${export_dir}/export.o" \
      -v "PROJECT_DIR=${PROJECT_DIR},DATA_ROOT=${PIXEL_DATA_ROOT},OUT_ROOT=${LATENT_ROOT},BATCH=${EXPORT_BATCH},LIMIT=${EXPORT_LIMIT},FORCE=${EXPORT_FORCE},LOCAL_FILES_ONLY=${LOCAL_FILES_ONLY},SEED=${SEED}" \
      "${PBS_EXPORT}" | awk '{print $1}'
  )"
  printf 'export,celebahq256_latent_export,export,%s,%s,,%s,%s,%s,\n' \
    "${export_job_id}" "${export_dir}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${EXPORT_QUEUE}" "${EXPORT_WALLTIME}" >> "${SUBMITTED}"
  printf '[SUBMITTED] export %s %s\n' "${export_job_id}" "${export_dir}"
else
  printf '[SKIP_EXPORT] latent cache already exists at %s\n' "${LATENT_ROOT}"
fi

submit_train() {
  local exp_id="$1"
  local train_mode="$2"
  local pipe_stages="$3"
  local weight_mode="$4"
  local config="$5"
  local job_name="$6"
  local run_dir="${TRAIN_ROOT}/${exp_id}_${STAMP}"
  local manifest_row="${run_dir}/manifest_row_source.json"
  mkdir -p "${run_dir}"

  MANIFEST_ROW="${manifest_row}" RUN_DIR="${run_dir}" EXP_ID="${exp_id}" TRAIN_MODE="${train_mode}" \
  PIPE_STAGES="${pipe_stages}" WEIGHT_MODE="${weight_mode}" CONFIG="${config}" DATA_DIR="${LATENT_ROOT}" \
  STEPS="${STEPS}" K="${K}" BATCH_SIZE="${BATCH_SIZE}" N_F="${N_F}" LR="${LR}" STEP_SIZE="${STEP_SIZE}" \
  NOISE_STD="${NOISE_STD}" LANGEVIN_SIGN="${LANGEVIN_SIGN}" MAX_GRAD_NORM="${MAX_GRAD_NORM}" \
  SEED="${SEED}" SAVE_EVERY="${SAVE_EVERY}" VIS_EVERY="${VIS_EVERY}" \
  /home/kevienzzq/.conda/envs/llm-env/bin/python - <<'PY'
import json
import os
import time

payload = {
    "benchmark": "celebahq256_latent",
    "group": "celebahq256_latent_1node",
    "exp_id": os.environ["EXP_ID"],
    "story": "CelebA-HQ256 frozen-VAE latent EBM; ordinary completion-aware objective",
    "train_mode": os.environ["TRAIN_MODE"],
    "world_size": 4,
    "pipe_stages": int(os.environ["PIPE_STAGES"]),
    "K": int(os.environ["K"]),
    "steps": int(os.environ["STEPS"]),
    "lr": float(os.environ["LR"]),
    "step_size": float(os.environ["STEP_SIZE"]),
    "noise_std": float(os.environ["NOISE_STD"]),
    "langevin_sign": float(os.environ["LANGEVIN_SIGN"]),
    "max_grad_norm": float(os.environ["MAX_GRAD_NORM"]),
    "weight_mode": os.environ["WEIGHT_MODE"],
    "last2_beta": 0.01,
    "seed": int(os.environ["SEED"]),
    "num_nodes": 1,
    "ppn": 4,
    "batch_size": int(os.environ["BATCH_SIZE"]),
    "n_f": int(os.environ["N_F"]),
    "parallel_backend": "ddp",
    "save_every": int(os.environ["SAVE_EVERY"]),
    "vis_every": int(os.environ["VIS_EVERY"]),
    "data_dir": os.environ["DATA_DIR"],
    "config": os.environ["CONFIG"],
    "submitted_at_epoch": int(time.time()),
}
with open(os.environ["MANIFEST_ROW"], "w") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
PY

  qsub_vars=$(
    printf 'PROJECT_DIR=%s,GROUP=%s,EXP_ID=%s,STORY=%s,TRAIN_MODE=%s,WORLD_SIZE=%s,PIPE_STAGES=%s,K=%s,STEPS=%s,LR=%s,STEP_SIZE=%s,NOISE_STD=%s,WEIGHT_MODE=%s,LAST2_BETA=%s,SEED=%s,NUM_NODES=%s,PPN=%s,EVAL_IMAGES=%s,RUN_DIR=%s,MANIFEST_ROW_PATH=%s,SAVE_EVERY=%s,VIS_EVERY=%s,BATCH_SIZE=%s,N_F=%s,SIGMA_PD=%s,LANGEVIN_SIGN=%s,WEIGHT_DECAY=%s,MAX_GRAD_NORM=%s,PARALLEL_BACKEND=%s,DATA_DIR=%s,CONFIG=%s,SYNC_FRESH_INIT=%s,DEBUG_LEVEL=%s,LOG_EVERY=%s' \
      "${PROJECT_DIR}" \
      "celebahq256_latent_1node" \
      "${exp_id}" \
      "CelebA-HQ256 latent ${train_mode}" \
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
      "${SEED}" \
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
      "${LANGEVIN_SIGN}" \
      "0.0" \
      "${MAX_GRAD_NORM}" \
      "ddp" \
      "${LATENT_ROOT}" \
      "${config}" \
      "1" \
      "1" \
      "${LOG_EVERY}"
  )
  depend_args=()
  dependency=""
  if [[ -n "${export_job_id}" ]]; then
    dependency="${export_job_id}"
    depend_args=(-W "depend=afterok:${export_job_id}")
  fi
  local job_id
  job_id="$(qsub -A "${PBS_ACCOUNT}" -q "${TRAIN_QUEUE}" -l select=1:system=polaris -l filesystems="${PBS_FILESYSTEMS}" -l walltime="${TRAIN_WALLTIME}" -N "${job_name}" "${depend_args[@]}" -v "${qsub_vars}" "${PBS_TRAIN}" | awk '{print $1}')"
  printf 'train,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${exp_id}" "${train_mode}" "${job_id}" "${run_dir}" "${manifest_row}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${TRAIN_QUEUE}" "${TRAIN_WALLTIME}" "${dependency}" >> "${SUBMITTED}"
  printf '[SUBMITTED] train %s %s %s\n' "${job_id}" "${train_mode}" "${run_dir}"
}

submit_train \
  "celebahq256_latent_ddp_k${K}_s${STEPS}_seed${SEED}" \
  "ddp_fullk" \
  "1" \
  "deep_only" \
  "${PROJECT_DIR}/configs/celebahq256_latent_ddp_strict.yaml" \
  "chqlatd"

submit_train \
  "celebahq256_latent_pipe_p4_equal_k${K}_s${STEPS}_seed${SEED}" \
  "pipe_strict" \
  "4" \
  "uniform" \
  "${PROJECT_DIR}/configs/celebahq256_latent_pipeline_strict.yaml" \
  "chqlatp"

printf '[SUBMITTED_CSV] %s\n' "${SUBMITTED}"

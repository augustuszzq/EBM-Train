#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/eagle/lc-mpi/Zhiqing/polaris_ebm}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-/eagle/lc-mpi/Zhiqing:/lus/eagle/projects/lc-mpi/Zhiqing}"
CONDA_ENV="${CONDA_ENV:-/home/kevienzzq/.conda/envs/llm-env}"
TORCHRUN="${TORCHRUN:-${CONDA_ENV}/bin/torchrun}"
DATA_DIR="${DATA_DIR:-/eagle/lc-mpi/Zhiqing/polaris_ebm/data/imagenet32}"
RUN_ROOT="${RUN_ROOT:-/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_imagenet32_drl/local_phase1_two_node}"
TAG="${TAG:-$(date -u +%Y%m%d_%H%M%S)}"

DDP_STEPS="${DDP_STEPS:-500}"
PIPE_STEPS="${PIPE_STEPS:-500}"
K="${K:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SAVE_EVERY="${SAVE_EVERY:-5000}"
VIS_EVERY="${VIS_EVERY:-5000}"
SEED="${SEED:-1}"
LR="${LR:-5e-5}"
STEP_SIZE="${STEP_SIZE:-0.005}"
NOISE_STD="${NOISE_STD:-0.005}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.0}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-}"
SIGMA_PD="${SIGMA_PD:-0.03}"
LOG_EVERY="${LOG_EVERY:-50}"
DEBUG_LEVEL="${DEBUG_LEVEL:-1}"
DIAGNOSTIC_FIRST_STEPS="${DIAGNOSTIC_FIRST_STEPS:-200}"
DIAGNOSTIC_LOG_EVERY="${DIAGNOSTIC_LOG_EVERY:-50}"
CRASH_ABS_ENERGY="${CRASH_ABS_ENERGY:-1000.0}"
CRASH_MAX_ABS_CHAIN_UNCLAMPED="${CRASH_MAX_ABS_CHAIN_UNCLAMPED:-2.5}"
CRASH_GRAD_NORM="${CRASH_GRAD_NORM:-10000.0}"
SKIP_OPTIMIZER_UNTIL_FULL_DIAGONAL="${SKIP_OPTIMIZER_UNTIL_FULL_DIAGONAL:-1}"
DRY_RUN="${DRY_RUN:-0}"

extra_grad_clip_arg=""
if [[ -n "${GRAD_CLIP_NORM}" ]]; then
  extra_grad_clip_arg=' --grad_clip_norm "'"${GRAD_CLIP_NORM}"'"'
fi

if [[ -z "${PBS_NODEFILE:-}" || ! -f "${PBS_NODEFILE}" ]]; then
  echo "[ERROR] PBS_NODEFILE is required for two-node local launch" >&2
  exit 2
fi

mapfile -t NODES < <(sort -u "${PBS_NODEFILE}")
if [[ "${#NODES[@]}" -lt 2 ]]; then
  echo "[ERROR] need at least 2 allocated nodes, found ${#NODES[@]}" >&2
  exit 2
fi
NODE0="${NODES[0]}"
NODE1="${NODES[1]}"

DDP_RUN_DIR="${RUN_ROOT}/ddp_k100_s${DDP_STEPS}_seed${SEED}_${TAG}"
PIPE_RUN_DIR="${RUN_ROOT}/pipe_p4_equal_k100_s${PIPE_STEPS}_seed${SEED}_${TAG}"
mkdir -p "${DDP_RUN_DIR}/checkpoints" "${DDP_RUN_DIR}/vis" "${PIPE_RUN_DIR}/checkpoints" "${PIPE_RUN_DIR}/vis"

write_info() {
  local path="$1"
  {
    echo "tag=${TAG}"
    echo "node0=${NODE0}"
    echo "node1=${NODE1}"
    echo "ddp_run_dir=${DDP_RUN_DIR}"
    echo "pipe_run_dir=${PIPE_RUN_DIR}"
    echo "ddp_steps=${DDP_STEPS}"
    echo "pipe_steps=${PIPE_STEPS}"
    echo "k=${K}"
    echo "batch_size=${BATCH_SIZE}"
    echo "seed=${SEED}"
  } > "${path}"
}
write_info "${RUN_ROOT}/launcher_info_${TAG}.txt"

remote_prefix='set -euo pipefail
module use /soft/modulefiles
module load conda
conda activate "'"${CONDA_ENV}"'"
cd "'"${PROJECT_DIR}"'"
export PYTHONPATH="'"${PYTHONPATH_ROOT}"'"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
'

ddp_cmd='echo "[LOCAL-DDP] host=$(hostname) run_dir='"${DDP_RUN_DIR}"'"
"'"${TORCHRUN}"'" --standalone --nproc_per_node=4 "'"${PROJECT_DIR}"'/scripts/current/ebm_train_imagenet32_drl_sync.py" \
  --mode ddp_fullk \
  --run_dir "'"${DDP_RUN_DIR}"'" \
  --save_dir "'"${DDP_RUN_DIR}"'/checkpoints" \
  --vis_dir "'"${DDP_RUN_DIR}"'/vis" \
  --config "'"${PROJECT_DIR}"'/configs/imagenet32_drl_ddp_k100.yaml" \
  --data_dir "'"${DATA_DIR}"'" \
  --steps "'"${DDP_STEPS}"'" \
  --save_every "'"${SAVE_EVERY}"'" \
  --vis_every "'"${VIS_EVERY}"'" \
  --K "'"${K}"'" \
  --batch_size "'"${BATCH_SIZE}"'" \
  --lr "'"${LR}"'" \
  --weight_decay "'"${WEIGHT_DECAY}"'" \
  --max_grad_norm "'"${MAX_GRAD_NORM}"'" \
  --step_size "'"${STEP_SIZE}"'" \
  --noise_std "'"${NOISE_STD}"'" \
	  --langevin_sign 1.0 \
	  --fresh_init \
	  --sync_fresh_init \
	  --pos_noise_std "'"${SIGMA_PD}"'" \
	  --no_clamp_pos \
	  --seed "'"${SEED}"'" \
	  --debug_level "'"${DEBUG_LEVEL}"'" \
	  --log_every "'"${LOG_EVERY}"'" \
	  --diagnostic_first_steps "'"${DIAGNOSTIC_FIRST_STEPS}"'" \
	  --diagnostic_log_every "'"${DIAGNOSTIC_LOG_EVERY}"'" \
	  --crash_abs_energy "'"${CRASH_ABS_ENERGY}"'" \
	  --crash_max_abs_chain_unclamped "'"${CRASH_MAX_ABS_CHAIN_UNCLAMPED}"'" \
	  --crash_grad_norm "'"${CRASH_GRAD_NORM}"'"'"${extra_grad_clip_arg}"'
'

pipe_cmd='echo "[LOCAL-PIPE] host=$(hostname) run_dir='"${PIPE_RUN_DIR}"'"
"'"${TORCHRUN}"'" --standalone --nproc_per_node=4 "'"${PROJECT_DIR}"'/scripts/current/ebm_train_imagenet32_drl_sync.py" \
  --mode pipeline \
  --run_dir "'"${PIPE_RUN_DIR}"'" \
  --save_dir "'"${PIPE_RUN_DIR}"'/checkpoints" \
  --vis_dir "'"${PIPE_RUN_DIR}"'/vis" \
  --config "'"${PROJECT_DIR}"'/configs/imagenet32_drl_pipeline_p4_equal_k100.yaml" \
  --data_dir "'"${DATA_DIR}"'" \
  --steps "'"${PIPE_STEPS}"'" \
  --save_every "'"${SAVE_EVERY}"'" \
  --vis_every "'"${VIS_EVERY}"'" \
  --K "'"${K}"'" \
  --batch_size "'"${BATCH_SIZE}"'" \
  --lr "'"${LR}"'" \
  --weight_decay "'"${WEIGHT_DECAY}"'" \
  --max_grad_norm "'"${MAX_GRAD_NORM}"'" \
  --step_size "'"${STEP_SIZE}"'" \
  --noise_std "'"${NOISE_STD}"'" \
	  --langevin_sign 1.0 \
	  --fresh_init \
	  --sync_fresh_init \
	  --pos_noise_std "'"${SIGMA_PD}"'" \
	  --no_clamp_pos \
	  --seed "'"${SEED}"'" \
	  --debug_level "'"${DEBUG_LEVEL}"'" \
	  --log_every "'"${LOG_EVERY}"'" \
	  --diagnostic_first_steps "'"${DIAGNOSTIC_FIRST_STEPS}"'" \
	  --diagnostic_log_every "'"${DIAGNOSTIC_LOG_EVERY}"'" \
	  --crash_abs_energy "'"${CRASH_ABS_ENERGY}"'" \
	  --crash_max_abs_chain_unclamped "'"${CRASH_MAX_ABS_CHAIN_UNCLAMPED}"'" \
	  --crash_grad_norm "'"${CRASH_GRAD_NORM}"'"'"${extra_grad_clip_arg}"' \
	  --skip_optimizer_until_full_diagonal \
	  --pipe_group_scope world \
  --pipe_stages 4 \
  --weight_mode uniform \
  --last2_beta 0.01
'

echo "[LAUNCH] node0=${NODE0} DDP -> ${DDP_RUN_DIR}"
echo "[LAUNCH] node1=${NODE1} PIPE -> ${PIPE_RUN_DIR}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY_RUN] not launching"
  exit 0
fi

setsid nohup mpiexec -n 1 -ppn 1 --host "${NODE0}" bash -lc "${remote_prefix}${ddp_cmd}" > "${DDP_RUN_DIR}/local_launcher.log" 2>&1 < /dev/null &
DDP_PID=$!
setsid nohup mpiexec -n 1 -ppn 1 --host "${NODE1}" bash -lc "${remote_prefix}${pipe_cmd}" > "${PIPE_RUN_DIR}/local_launcher.log" 2>&1 < /dev/null &
PIPE_PID=$!

{
  echo "ddp_launcher_pid=${DDP_PID}"
  echo "pipe_launcher_pid=${PIPE_PID}"
} >> "${RUN_ROOT}/launcher_info_${TAG}.txt"

echo "[LAUNCHED] ddp_pid=${DDP_PID} pipe_pid=${PIPE_PID}"
echo "[INFO] ${RUN_ROOT}/launcher_info_${TAG}.txt"

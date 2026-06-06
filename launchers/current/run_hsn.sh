#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Mode:
#   MODE=baseline|pipeline
MODE="${MODE:-baseline}"

# Networking mode:
#   NET_MODE=auto|socket|ofi
#   USE_AWS_OFI=1 -> force OFI
#   USE_SOCKET=1  -> force Socket
NET_MODE="${NET_MODE:-auto}"
USE_AWS_OFI="${USE_AWS_OFI:-0}"
USE_SOCKET="${USE_SOCKET:-0}"
AUTO_IFACE="${AUTO_IFACE:-1}"

if [[ -z "${PBS_NODEFILE:-}" ]]; then
  echo "PBS_NODEFILE not set. Are you inside qsub -I?"
  exit 1
fi

HOSTFILE="$(mktemp)"
trap 'rm -f "$HOSTFILE"' EXIT
sort -u "$PBS_NODEFILE" > "$HOSTFILE"
mapfile -t NODES < "$HOSTFILE"

NNODES=${#NODES[@]}
MASTER_HOST="${NODES[0]}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
PPN="${PPN:-${GPUS_PER_NODE}}"
WORLD_SIZE="${WORLD_SIZE:-$((NNODES * PPN))}"
MASTER_PORT="${MASTER_PORT:-29500}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
DEBUG_LAUNCHER="${DEBUG_LAUNCHER:-0}"
CLEANUP="${CLEANUP:-0}"
GPU_CHECK="${GPU_CHECK:-0}"

if (( WORLD_SIZE != NNODES * PPN )); then
  echo "[ERROR] WORLD_SIZE (${WORLD_SIZE}) must equal NNODES*PPN ($((NNODES * PPN)))."
  exit 1
fi

detect_iface() {
  local iface
  for iface in hsn1 hsn0; do
    if ip link show "${iface}" >/dev/null 2>&1; then
      echo "${iface}"
      return 0
    fi
  done
  echo "hsn0"
}

if [[ -z "${HSN_IFNAME:-}" && "${AUTO_IFACE}" == "1" ]]; then
  HSN_IFNAME="$(detect_iface)"
fi
HSN_IFNAME="${HSN_IFNAME:-hsn0}"

strip_aws_ofi_from_ld_path() {
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH
    LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v 'aws-ofi-nccl' | paste -sd: -)"
  fi
}

if [[ "${NET_MODE}" == "auto" ]]; then
  if [[ "${USE_AWS_OFI}" == "1" ]]; then
    NET_MODE="ofi"
  elif [[ "${USE_SOCKET}" == "1" ]]; then
    NET_MODE="socket"
  elif [[ -f "/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib/libnccl-net.so" ]]; then
    NET_MODE="ofi"
  else
    NET_MODE="socket"
  fi
fi

if [[ "${NET_MODE}" == "ofi" ]]; then
  export NCCL_NET="AWS Libfabric"
  export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
  export NCCL_COLLNET_ENABLE="${NCCL_COLLNET_ENABLE:-1}"
  export FI_PROVIDER="${FI_PROVIDER:-cxi}"
  export FI_CXI_DISABLE_HOST_REGISTER="${FI_CXI_DISABLE_HOST_REGISTER:-1}"
  export FI_MR_CACHE_MONITOR="${FI_MR_CACHE_MONITOR:-userfaultfd}"
  export FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE:-software}"
  export FI_CXI_DEFAULT_CQ_SIZE="${FI_CXI_DEFAULT_CQ_SIZE:-131072}"
  export LD_LIBRARY_PATH="/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:/soft/libraries/hwloc/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
  export NCCL_NET="${NCCL_NET:-Socket}"
  unset NCCL_NET_GDR_LEVEL NCCL_CROSS_NIC NCCL_COLLNET_ENABLE
  unset FI_PROVIDER FI_CXI_DISABLE_HOST_REGISTER FI_CXI_RX_MATCH_MODE FI_CXI_DEFAULT_TCLASS FI_CXI_DEFAULT_CQ_SIZE FI_MR_CACHE_MONITOR
  strip_aws_ofi_from_ld_path
fi

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${HSN_IFNAME}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${HSN_IFNAME}}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_TIMEOUT="${NCCL_SOCKET_TIMEOUT:-300}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-300}"
export NCCL_SOCKET_PORT_RANGE="${NCCL_SOCKET_PORT_RANGE:-}"
export NCCL_DEBUG="${NCCL_DEBUG:-}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export TORCH_DISTRIBUTED_TIMEOUT="${TORCH_DISTRIBUTED_TIMEOUT:-3600}"
NCCL_NET_SHELL_ESCAPED="$(printf '%q' "${NCCL_NET}")"

MASTER_ADDR=$(
  mpiexec -n 1 -ppn 1 --host "${MASTER_HOST}" \
    bash -lc 'ip -o -4 addr show '"${HSN_IFNAME}"' | sed -E "s/ +/ /g" | cut -d" " -f4 | cut -d/ -f1'
)
if [[ -z "${MASTER_ADDR}" ]]; then
  echo "[ERROR] failed to resolve MASTER_ADDR on ${MASTER_HOST} (${HSN_IFNAME})"
  exit 1
fi

MODEL_SCALE="${MODEL_SCALE:-small}"
NF="${NF:-64}"
K="${K:-100}"
GLOBAL_BATCH="${GLOBAL_BATCH:-64}"
DATA_DIR="${DATA_DIR:-/eagle/lc-mpi/Zhiqing/ebm/data/cifar10}"
OUTPUT_DIR="${OUTPUT_DIR:-/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_dual}"
STEPS="${STEPS:-2000}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
STEP_SIZE="${STEP_SIZE:-0.2}"
NOISE_STD="${NOISE_STD:-1e-2}"
SAMPLE_DTYPE="${SAMPLE_DTYPE:-auto}"
POS_NOISE_STD="${POS_NOISE_STD:-0.0}"
DEBUG_LEVEL="${DEBUG_LEVEL:-1}"
LOG_EVERY="${LOG_EVERY:-10}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-1}"
DIST_TIMEOUT_SEC="${DIST_TIMEOUT_SEC:-3600}"
FRESH_INIT="${FRESH_INIT:-0}"
NO_CLAMP_X="${NO_CLAMP_X:-0}"
NO_CLAMP_POS="${NO_CLAMP_POS:-0}"

if [[ "${MODE}" == "baseline" ]]; then
  MODE_NAME="baseline"
  MODE_SAMPLE_DTYPE="${MODE_SAMPLE_DTYPE:-fp32}"
  DEFAULT_N_TRAINERS="${NNODES}"
elif [[ "${MODE}" == "pipeline" ]]; then
  MODE_NAME="pipeline"
  MODE_SAMPLE_DTYPE="${MODE_SAMPLE_DTYPE:-fp16}"
  DEFAULT_N_TRAINERS="$((WORLD_SIZE / 2))"
else
  echo "[ERROR] MODE must be baseline or pipeline, got ${MODE}"
  exit 1
fi

if [[ "${DEFAULT_N_TRAINERS}" -lt 1 ]]; then
  DEFAULT_N_TRAINERS=1
fi
if [[ "${DEFAULT_N_TRAINERS}" -ge "${WORLD_SIZE}" ]]; then
  DEFAULT_N_TRAINERS=$((WORLD_SIZE - 1))
fi

N_TRAINERS="${N_TRAINERS:-${DEFAULT_N_TRAINERS}}"
if [[ "${N_TRAINERS}" -le 0 || "${N_TRAINERS}" -ge "${WORLD_SIZE}" ]]; then
  echo "[ERROR] N_TRAINERS (${N_TRAINERS}) must satisfy 1 <= N_TRAINERS < WORLD_SIZE (${WORLD_SIZE})"
  exit 1
fi
if [[ "${GLOBAL_BATCH}" -le 0 ]]; then
  echo "[ERROR] GLOBAL_BATCH must be > 0"
  exit 1
fi
if (( GLOBAL_BATCH % N_TRAINERS != 0 )); then
  echo "[ERROR] GLOBAL_BATCH (${GLOBAL_BATCH}) must be divisible by N_TRAINERS (${N_TRAINERS})"
  exit 1
fi

if [[ -z "${SAMPLE_DTYPE:-}" || "${SAMPLE_DTYPE}" == "auto" ]]; then
  SAMPLE_DTYPE="${MODE_SAMPLE_DTYPE}"
fi

LOG_DIR="${LOG_DIR:-/eagle/lc-mpi/Zhiqing/polaris_ebm/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_hsn_${MODE_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "[RUN] MODE=${MODE_NAME} NNODES=${NNODES} PPN=${PPN} WORLD_SIZE=${WORLD_SIZE} N_TRAINERS=${N_TRAINERS}"
echo "[RUN] MASTER_HOST=${MASTER_HOST} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} HSN_IFNAME=${HSN_IFNAME}"
echo "[RUN] NET_MODE=${NET_MODE} NCCL_NET=${NCCL_NET} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "[RUN] sample_dtype=${SAMPLE_DTYPE} global_batch=${GLOBAL_BATCH} K=${K} steps=${STEPS}"
echo "[RUN] LOG_FILE=${LOG_FILE}"

if [[ "${CLEANUP}" == "1" || "${CLEANUP}" == "true" ]]; then
  echo "[RUN] CLEANUP enabled: killing stale ebm_train_sync_mode_a.py / torch.distributed.run on allocated nodes"
  mpiexec -n "${NNODES}" -ppn 1 --hostfile "${HOSTFILE}" \
    bash -lc 'pkill -f "[e]bm_train_sync_mode_a.py" 2>/dev/null || true; pkill -f "[t]orch.distributed.run.*[e]bm_train_sync_mode_a.py" 2>/dev/null || true'
  sleep 2
fi

if [[ "${GPU_CHECK}" == "1" || "${GPU_CHECK}" == "true" ]]; then
  echo "[RUN] GPU_CHECK enabled: dumping per-node GPU occupancy before launch"
  mpiexec -n "${NNODES}" -ppn 1 --hostfile "${HOSTFILE}" \
    bash -lc 'echo "[GPUCHK] host=$(hostname)"; nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader'
  mpiexec -n "${NNODES}" -ppn 1 --hostfile "${HOSTFILE}" \
    bash -lc 'echo "[GPUCHK-PROC] host=$(hostname)"; nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true'
fi

mpiexec -n "${WORLD_SIZE}" -ppn "${PPN}" --hostfile "${HOSTFILE}" --cpu-bind depth -d "${OMP_NUM_THREADS}" \
  bash -lc '
    set -euo pipefail
    module use /soft/modulefiles
    module load conda
    conda activate '"${CONDA_ENV:-/home/kevienzzq/.conda/envs/llm-env}"'
    cd "${PROJECT_DIR}"

    export MASTER_ADDR='"${MASTER_ADDR}"'
    export MASTER_PORT='"${MASTER_PORT}"'
    export NCCL_NET='"${NCCL_NET_SHELL_ESCAPED}"'
    export NCCL_SOCKET_IFNAME='"${NCCL_SOCKET_IFNAME}"'
    export GLOO_SOCKET_IFNAME='"${GLOO_SOCKET_IFNAME}"'
    export NCCL_IB_DISABLE='"${NCCL_IB_DISABLE}"'
    export NCCL_SOCKET_TIMEOUT='"${NCCL_SOCKET_TIMEOUT}"'
    export NCCL_TIMEOUT='"${NCCL_TIMEOUT}"'
    export NCCL_SOCKET_PORT_RANGE='"${NCCL_SOCKET_PORT_RANGE}"'
    export NCCL_DEBUG='"${NCCL_DEBUG}"'
    export NCCL_DEBUG_SUBSYS='"${NCCL_DEBUG_SUBSYS}"'
    export TORCH_DISTRIBUTED_DEBUG='"${TORCH_DISTRIBUTED_DEBUG}"'
    export TORCH_DISTRIBUTED_TIMEOUT='"${TORCH_DISTRIBUTED_TIMEOUT}"'
    export DEBUG_LAUNCHER='"${DEBUG_LAUNCHER}"'
    export OMP_NUM_THREADS='"${OMP_NUM_THREADS}"'
    export FRESH_INIT='"${FRESH_INIT}"'
    export NO_CLAMP_X='"${NO_CLAMP_X}"'
    export NO_CLAMP_POS='"${NO_CLAMP_POS}"'
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export RANK=${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-${PMI_RANK:-${PALS_RANKID:-${MPICH_RANK:-0}}}}}
    export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE:-${PMIX_SIZE:-${PMI_SIZE:-${PALS_NTASKS:-${MPICH_NTASKS:-1}}}}}
    if [[ -n "${OMPI_COMM_WORLD_LOCAL_RANK:-}" ]]; then
      export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}
    elif [[ -n "${PMIX_LOCAL_RANK:-}" ]]; then
      export LOCAL_RANK=${PMIX_LOCAL_RANK}
    elif [[ -n "${MPI_LOCALRANKID:-}" ]]; then
      export LOCAL_RANK=${MPI_LOCALRANKID}
    elif [[ -n "${PMI_LOCAL_RANK:-}" ]]; then
      export LOCAL_RANK=${PMI_LOCAL_RANK}
    elif [[ -n "${MPICH_LOCAL_RANK:-}" ]]; then
      export LOCAL_RANK=${MPICH_LOCAL_RANK}
    elif [[ -n "${PALS_LOCAL_RANKID:-}" ]]; then
      export LOCAL_RANK=${PALS_LOCAL_RANKID}
    else
      export LOCAL_RANK=$((RANK % '"${PPN}"'))
    fi
    if [[ -n "${OMPI_COMM_WORLD_LOCAL_SIZE:-}" ]]; then
      export LOCAL_WORLD_SIZE=${OMPI_COMM_WORLD_LOCAL_SIZE}
    elif [[ -n "${PMIX_LOCAL_SIZE:-}" ]]; then
      export LOCAL_WORLD_SIZE=${PMIX_LOCAL_SIZE}
    elif [[ -n "${MPI_LOCALNRANKS:-}" ]]; then
      export LOCAL_WORLD_SIZE=${MPI_LOCALNRANKS}
    elif [[ -n "${PMI_LOCAL_SIZE:-}" ]]; then
      export LOCAL_WORLD_SIZE=${PMI_LOCAL_SIZE}
    elif [[ -n "${MPICH_LOCAL_SIZE:-}" ]]; then
      export LOCAL_WORLD_SIZE=${MPICH_LOCAL_SIZE}
    elif [[ -n "${PALS_LOCAL_SIZE:-}" ]]; then
      export LOCAL_WORLD_SIZE=${PALS_LOCAL_SIZE}
    else
      export LOCAL_WORLD_SIZE='"${PPN}"'
    fi
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $(('"${PPN}"' - 1)))"
    fi
    if [[ "${DEBUG_LAUNCHER}" -ge 1 ]]; then
      echo "[RANKMAP] host=$(hostname) rank=${RANK} local_rank=${LOCAL_RANK} world_size=${WORLD_SIZE} local_world_size=${LOCAL_WORLD_SIZE}" >&2
      echo "[RANKMAP] PMIX_RANK=${PMIX_RANK:-unset} PMI_RANK=${PMI_RANK:-unset} PALS_RANKID=${PALS_RANKID:-unset}" >&2
      echo "[RANKMAP] PMIX_LOCAL_RANK=${PMIX_LOCAL_RANK:-unset} PMI_LOCAL_RANK=${PMI_LOCAL_RANK:-unset} PALS_LOCAL_RANKID=${PALS_LOCAL_RANKID:-unset}" >&2
      echo "[RANKMAP] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
    fi

    CONDA_PY="${CONDA_PY:-/home/kevienzzq/.conda/envs/llm-env/bin/python}"
    EXTRA_ARGS=()
    if [[ "${FRESH_INIT}" == "1" ]]; then
      EXTRA_ARGS+=(--fresh_init)
    fi
    if [[ "${NO_CLAMP_X}" == "1" ]]; then
      EXTRA_ARGS+=(--no_clamp_x)
    fi
    if [[ "${NO_CLAMP_POS}" == "1" ]]; then
      EXTRA_ARGS+=(--no_clamp_pos)
    fi
    "${CONDA_PY}" -u scripts/current/ebm_train_sync_mode_a.py \
      --mode '"${MODE_NAME}"' \
      --model_scale '"${MODEL_SCALE}"' \
      --n_f '"${NF}"' \
      --K '"${K}"' \
      --batch_size '"${GLOBAL_BATCH}"' \
      --data_dir '"${DATA_DIR}"' \
      --output_dir '"${OUTPUT_DIR}"' \
      --steps '"${STEPS}"' \
      --lr '"${LR}"' \
      --weight_decay '"${WEIGHT_DECAY}"' \
      --max_grad_norm '"${MAX_GRAD_NORM}"' \
      --step_size '"${STEP_SIZE}"' \
      --noise_std '"${NOISE_STD}"' \
      --sample_dtype '"${SAMPLE_DTYPE}"' \
      --pos_noise_std '"${POS_NOISE_STD}"' \
      --n_trainers '"${N_TRAINERS}"' \
      --num_workers '"${NUM_WORKERS}"' \
      --seed '"${SEED}"' \
      --dist_timeout_sec '"${DIST_TIMEOUT_SEC}"' \
      --debug_level '"${DEBUG_LEVEL}"' \
      --log_every '"${LOG_EVERY}"' \
      "${EXTRA_ARGS[@]}"
  ' |& tee "${LOG_FILE}"

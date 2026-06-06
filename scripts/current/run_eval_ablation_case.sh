#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PROJECT_DIR:-}" ]]; then
  PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
elif [[ -n "${PBS_O_WORKDIR:-}" && -f "${PBS_O_WORKDIR}/scripts/current/eval_generate.py" ]]; then
  PROJECT_DIR="${PBS_O_WORKDIR}"
else
  PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi
cd "${PROJECT_DIR}"

module use /soft/modulefiles
module load conda
conda activate /home/kevienzzq/.conda/envs/llm-env

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PROJECT_PARENT="$(cd "${PROJECT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_PARENT}${PYTHONPATH:+:$PYTHONPATH}"

PY=/home/kevienzzq/.conda/envs/llm-env/bin/python

GROUP="${GROUP:?GROUP is required}"
EXP_ID="${EXP_ID:?EXP_ID is required}"
TRAIN_MODE="${TRAIN_MODE:?TRAIN_MODE is required}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
EVAL_DIR="${EVAL_DIR:-${RUN_DIR}/eval}"
EVAL_IMAGES="${EVAL_IMAGES:-5000}"
EVAL_BATCH="${EVAL_BATCH:-256}"
K_EVAL="${K_EVAL:-100}"
STEP_SIZE="${STEP_SIZE:-1.0}"
NOISE_STD="${NOISE_STD:-0.01}"
LANGEVIN_SIGN="${LANGEVIN_SIGN:-1.0}"
EVAL_SEED="${EVAL_SEED:-1}"
DATA_DIR="${DATA_DIR:-/eagle/lc-mpi/Zhiqing/ebm/data/cifar10}"
CKPT_PATH="${CKPT_PATH:-}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${EVAL_DIR}"
EVAL_LOG="${EVAL_DIR}/eval.log"
JOB_INFO_JSON="${RUN_DIR}/job_info.json"
export PROJECT_DIR GROUP EXP_ID TRAIN_MODE RUN_DIR EVAL_DIR EVAL_IMAGES EVAL_BATCH K_EVAL STEP_SIZE
export NOISE_STD LANGEVIN_SIGN EVAL_SEED DATA_DIR CKPT_PATH DRY_RUN JOB_INFO_JSON

python3 - <<PY
import json, os, time
path = os.environ["JOB_INFO_JSON"]
payload = {}
if os.path.exists(path):
    with open(path) as f:
        payload = json.load(f)
payload["eval_job_id"] = os.environ.get("PBS_JOBID", "")
payload["eval_status"] = "running"
payload["eval_started_at_epoch"] = int(time.time())
payload["eval_started_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
with open(path, "w") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
PY

if [[ -z "${CKPT_PATH}" ]]; then
  CKPT_PATH="$(ls -1 "${RUN_DIR}"/checkpoints/ckpt_step*.pt 2>/dev/null | sort -V | tail -n 1 || true)"
fi
if [[ -z "${CKPT_PATH}" || ! -f "${CKPT_PATH}" ]]; then
  echo "[ERROR] no checkpoint found for ${RUN_DIR}" | tee -a "${EVAL_LOG}"
  exit 3
fi
export CKPT_PATH

TMP_GEN="${EVAL_DIR}/_tmp_generate_$(date +%Y%m%d_%H%M%S)"
METRICS_JSON="${EVAL_DIR}/metrics_compare.json"
GRID_PNG="${EVAL_DIR}/grid.png"
STATS_JSON="${EVAL_DIR}/stats.json"
SAMPLES_PT="${EVAL_DIR}/samples.pt"
export TMP_GEN METRICS_JSON GRID_PNG STATS_JSON SAMPLES_PT

echo "[EVAL] exp_id=${EXP_ID} run_dir=${RUN_DIR} ckpt=${CKPT_PATH}" | tee -a "${EVAL_LOG}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY RUN] skipping eval_generate/eval_metrics" | tee -a "${EVAL_LOG}"
  export EVAL_RC=0
else
  set +e
  "${PY}" -u "${PROJECT_DIR}/scripts/current/eval_generate.py" \
    --ckpt "${CKPT_PATH}" \
    --out_dir "${TMP_GEN}" \
    --num_images "${EVAL_IMAGES}" \
    --batch "${EVAL_BATCH}" \
    --K_eval "${K_EVAL}" \
    --langevin_sign "${LANGEVIN_SIGN}" \
    --step_size "${STEP_SIZE}" \
    --noise_std "${NOISE_STD}" \
    --seed "${EVAL_SEED}" \
    --device cuda \
    --no_clamp_x 2>&1 | tee -a "${EVAL_LOG}"
  gen_rc=${PIPESTATUS[0]}
  if [[ "${gen_rc}" -ne 0 ]]; then
    set -e
    export EVAL_RC="${gen_rc}"
  else
    cp "${TMP_GEN}/grid.png" "${GRID_PNG}"
    cp "${TMP_GEN}/samples.pt" "${SAMPLES_PT}"
    "${PY}" -u "${PROJECT_DIR}/scripts/current/eval_metrics.py" \
      --baseline_samples "${TMP_GEN}/samples.pt" \
      --pipeline_samples "${TMP_GEN}/samples.pt" \
      --data_dir "${DATA_DIR}" \
      --num_real "${EVAL_IMAGES}" \
      --batch "${EVAL_BATCH}" \
      --out_json "${METRICS_JSON}" \
      --device cuda 2>&1 | tee -a "${EVAL_LOG}"
    metrics_rc=${PIPESTATUS[0]}
    export EVAL_RC="${metrics_rc}"
  fi
  set -e
fi

python3 - <<PY
import json, os, time
from pathlib import Path

try:
    from ablation_common import collect_training_tail_stats, load_json, normalize_metrics_payload
except Exception:
    from polaris_ebm.scripts.current.ablation_common import collect_training_tail_stats, load_json, normalize_metrics_payload

job_path = Path(os.environ["JOB_INFO_JSON"])
payload = load_json(job_path, default={}) or {}
end_ts = int(time.time())
start_ts = int(payload.get("eval_started_at_epoch", end_ts))
eval_rc = int(os.environ.get("EVAL_RC", "0"))
payload["eval_finished_at_epoch"] = end_ts
payload["eval_finished_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_ts))
payload["eval_walltime_seconds"] = max(0, end_ts - start_ts)
payload["eval_status"] = "dry_run" if os.environ.get("DRY_RUN", "0") == "1" else ("succeeded" if eval_rc == 0 else "failed")
payload["total_walltime_seconds"] = float(payload.get("train_walltime_seconds", 0.0)) + float(payload.get("eval_walltime_seconds", 0.0))
with job_path.open("w") as f:
    json.dump(payload, f, indent=2, sort_keys=True)

if eval_rc != 0 or os.environ.get("DRY_RUN", "0") == "1":
    raise SystemExit(eval_rc)

metrics = load_json(os.environ["METRICS_JSON"], default={}) or {}
gen_stats = load_json(Path(os.environ["TMP_GEN"]) / "stats.json", default={}) or {}
train_tail = collect_training_tail_stats(run_dir=os.environ["RUN_DIR"], train_mode=os.environ["TRAIN_MODE"])
summary = {
    "group": os.environ["GROUP"],
    "exp_id": os.environ["EXP_ID"],
    "train_mode": os.environ["TRAIN_MODE"],
    "run_dir": os.environ["RUN_DIR"],
    "eval_dir": os.environ["EVAL_DIR"],
    "ckpt_path": os.environ["CKPT_PATH"],
    "grid_path": os.environ["GRID_PNG"],
    "samples_path": os.environ["SAMPLES_PT"],
    "metrics_path": os.environ["METRICS_JSON"],
    "normalized_metrics": normalize_metrics_payload(metrics),
    "train_tail": train_tail,
    "generate_stats": gen_stats,
    "eval_walltime_seconds": payload.get("eval_walltime_seconds"),
    "total_walltime_seconds": payload.get("total_walltime_seconds"),
}
with open(os.environ["STATS_JSON"], "w") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
PY

echo "[DONE] eval_dir=${EVAL_DIR}" | tee -a "${EVAL_LOG}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_NAME="ebm-train"

cd "${PROJECT_DIR}"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[SETUP] updating conda env ${ENV_NAME} from environment.yml"
    conda env update -n "${ENV_NAME}" -f environment.yml --prune
  else
    echo "[SETUP] creating conda env ${ENV_NAME} from environment.yml"
    conda env create -f environment.yml
  fi
  echo "[DONE] activate with: conda activate ${ENV_NAME}"
else
  PYTHON_BIN="${PYTHON:-python3}"
  echo "[SETUP] conda not found; creating .venv with ${PYTHON_BIN}"
  "${PYTHON_BIN}" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip wheel setuptools
  .venv/bin/python -m pip install -r requirements-drl.txt
  echo "[DONE] activate with: source .venv/bin/activate"
fi

echo "[NEXT] copy .env.drl.example to .env.drl and set DATA_DIR"

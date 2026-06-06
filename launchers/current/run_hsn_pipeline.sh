#!/bin/bash
set -euo pipefail

export MODE="${MODE:-pipeline}"
export SAMPLE_DTYPE="${SAMPLE_DTYPE:-fp16}"

exec bash "$(dirname "$0")/run_hsn.sh"

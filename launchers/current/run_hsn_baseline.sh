#!/bin/bash
set -euo pipefail

export MODE="${MODE:-baseline}"
export SAMPLE_DTYPE="${SAMPLE_DTYPE:-fp32}"

exec bash "$(dirname "$0")/run_hsn.sh"

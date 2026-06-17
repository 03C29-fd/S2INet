#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

RUN_NAME="${RUN_NAME:-rgc_linear_lag${RESPONSE_LAG:-3}_h${HISTORY_BINS:-11}_k${TRAIN_REPEATS:-8}_$(date +%Y%m%d_%H%M%S)}" \
DECODER="${DECODER:-linear}" \
bash run_rgc_frame_decoder.sh

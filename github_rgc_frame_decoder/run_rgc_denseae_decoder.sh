#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

RUN_NAME="${RUN_NAME:-rgc_denseae_lag${RESPONSE_LAG:-3}_h${HISTORY_BINS:-11}_k${TRAIN_REPEATS:-8}_$(date +%Y%m%d_%H%M%S)}" \
DECODER="${DECODER:-dense_ae}" \
LATENT_DIM="${LATENT_DIM:-384}" \
BASE_CHANNELS="${BASE_CHANNELS:-96}" \
DROPOUT="${DROPOUT:-0.30}" \
bash run_rgc_frame_decoder.sh

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export DECODER="${DECODER:-lowfreq_aux}"
export LOSS_MODE="${LOSS_MODE:-rf_ssim_wlmse_gabor}"
export LATENT_DIM="${LATENT_DIM:-384}"
export TEMPORAL_CHANNELS="${TEMPORAL_CHANNELS:-128}"
export BASE_CHANNELS="${BASE_CHANNELS:-96}"
export DROPOUT="${DROPOUT:-0.25}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-3}"
export LR="${LR:-3e-4}"
export GABOR_LOSS_WEIGHT="${GABOR_LOSS_WEIGHT:-0.02}"
export LOW_AUX_SIGMA="${LOW_AUX_SIGMA:-0}"
export LOW_AUX_WEIGHT="${LOW_AUX_WEIGHT:-0.10}"
export LOW_AUX_SSIM_WEIGHT="${LOW_AUX_SSIM_WEIGHT:-0.0}"
export RESIDUAL_GATE="${RESIDUAL_GATE:-rf_scalar}"
export RESIDUAL_WEIGHT="${RESIDUAL_WEIGHT:-0.10}"
export GATE_L1_WEIGHT="${GATE_L1_WEIGHT:-0.02}"
export CONSISTENCY_WEIGHT="${CONSISTENCY_WEIGHT:-0.02}"
export SINGLE_REPEAT_LOSS_WEIGHT="${SINGLE_REPEAT_LOSS_WEIGHT:-0.02}"
export RUN_NAME="${RUN_NAME:-rgc_lowfreqaux_lag${RESPONSE_LAG:-3}_h${HISTORY_BINS:-11}_seed${SEED:-1}}"

bash run_rgc_frame_decoder.sh

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi

if command -v conda >/dev/null 2>&1 && [ -n "${CONDA_ENV:-}" ]; then
  conda activate "$CONDA_ENV"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_NAME="${RUN_NAME:-rgc_pure_linear_lag${RESPONSE_LAG:-3}_h${HISTORY_BINS:-11}_k${TRAIN_REPEATS:-8}_$(date +%Y%m%d_%H%M%S)}"

EXTRA_ARGS=()
if [ -n "${MAX_SAMPLES:-}" ]; then
  EXTRA_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

"$PYTHON_BIN" train_rgc_pure_linear_baseline.py \
  --data-dir "${DATA_DIR:-data}" \
  --spikes-mat "${SPIKES_MAT:-data/movieBinnedSpiking.mat}" \
  --output-dir "${OUTPUT_DIR:-runs_rgc_frame}" \
  --run-name "$RUN_NAME" \
  --epochs "${EPOCHS:-60}" \
  --batch-size "${BATCH_SIZE:-96}" \
  --lr "${LR:-3e-4}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --num-workers "${NUM_WORKERS:-0}" \
  --seed "${SEED:-42}" \
  --device "${DEVICE:-auto}" \
  --image-size "${IMAGE_SIZE:-64}" \
  --split "${SPLIT:-scene}" \
  --val-movies "${VAL_MOVIES:-5}" \
  --val-ratio "${VAL_RATIO:-0.2}" \
  --embargo "${EMBARGO:-30}" \
  --scene-length "${SCENE_LENGTH:-120}" \
  --response-lag "${RESPONSE_LAG:-3}" \
  --history-bins "${HISTORY_BINS:-11}" \
  --train-repeats "${TRAIN_REPEATS:-8}" \
  --eval-repeats "${EVAL_REPEATS:-0}" \
  --loss-mode "${LOSS_MODE:-mse_ssim}" \
  --ssim-loss-weight "${SSIM_LOSS_WEIGHT:-0.05}" \
  --single-repeat-loss-weight "${SINGLE_REPEAT_LOSS_WEIGHT:-0.0}" \
  --patience "${PATIENCE:-12}" \
  --viz-samples "${VIZ_SAMPLES:-8}" \
  --save-last \
  "${EXTRA_ARGS[@]}"

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

echo "Python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
$PYTHON_BIN - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
PY

RUN_NAME="${RUN_NAME:-rgc_scene_lag3_h11_k8_rf_gabor_$(date +%Y%m%d_%H%M%S)}"

EXTRA_ARGS=()
if [ -n "${MAX_SAMPLES:-}" ]; then
  EXTRA_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

"$PYTHON_BIN" train_rgc_frame_decoder.py \
  --data-dir "${DATA_DIR:-data}" \
  --spikes-mat "${SPIKES_MAT:-data/movieBinnedSpiking.mat}" \
  --output-dir "${OUTPUT_DIR:-runs_rgc_frame}" \
  --run-name "$RUN_NAME" \
  --epochs "${EPOCHS:-80}" \
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
  --repeat-sampling "${REPEAT_SAMPLING:-random}" \
  --encoder "${ENCODER:-multiscale}" \
  --decoder "${DECODER:-attention}" \
  --repeat-pool "${REPEAT_POOL:-attention}" \
  --latent-dim "${LATENT_DIM:-384}" \
  --temporal-channels "${TEMPORAL_CHANNELS:-96}" \
  --base-channels "${BASE_CHANNELS:-96}" \
  --attention-heads "${ATTENTION_HEADS:-4}" \
  --attention-layers "${ATTENTION_LAYERS:-1}" \
  --dropout "${DROPOUT:-0.30}" \
  --loss-mode "${LOSS_MODE:-rf_ssim_wlmse_gabor}" \
  --loss-mu "${LOSS_MU:-0.1}" \
  --gabor-loss-weight "${GABOR_LOSS_WEIGHT:-0.02}" \
  --ssim-loss-weight "${SSIM_LOSS_WEIGHT:-0.05}" \
  --gradient-loss-weight "${GRADIENT_LOSS_WEIGHT:-0.1}" \
  --adv-weight "${ADV_WEIGHT:-0.01}" \
  --rf-sigma "${RF_SIGMA:-3.0}" \
  --freq-blur-sigma "${FREQ_BLUR_SIGMA:-2.0}" \
  --freq-ridge-alpha "${FREQ_RIDGE_ALPHA:-1.0}" \
  --consistency-weight "${CONSISTENCY_WEIGHT:-0.02}" \
  --single-repeat-loss-weight "${SINGLE_REPEAT_LOSS_WEIGHT:-0.03}" \
  --patience "${PATIENCE:-12}" \
  --estimate-batches "${ESTIMATE_BATCHES:-10}" \
  --viz-samples "${VIZ_SAMPLES:-8}" \
  --save-last \
  "${EXTRA_ARGS[@]}"

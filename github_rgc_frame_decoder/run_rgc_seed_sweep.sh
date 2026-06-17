#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SEEDS="${SEEDS:-1 2 3}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-96}"
RESPONSE_LAG="${RESPONSE_LAG:-3}"
BASE_RUN_PREFIX="${BASE_RUN_PREFIX:-rgc_seed_sweep_lag${RESPONSE_LAG}}"

for SEED_VALUE in $SEEDS; do
  RUN_NAME="${BASE_RUN_PREFIX}_seed${SEED_VALUE}_h${HISTORY_BINS:-11}_k${TRAIN_REPEATS:-8}"
  echo "=== Running seed=${SEED_VALUE}, lag=${RESPONSE_LAG} -> ${RUN_NAME} ==="
  RUN_NAME="$RUN_NAME" \
  SEED="$SEED_VALUE" \
  RESPONSE_LAG="$RESPONSE_LAG" \
  EPOCHS="$EPOCHS" \
  BATCH_SIZE="$BATCH_SIZE" \
  bash run_rgc_frame_decoder.sh
done

python summarize_rgc_frame_runs.py "${OUTPUT_DIR:-runs_rgc_frame}" --prefix "$BASE_RUN_PREFIX"

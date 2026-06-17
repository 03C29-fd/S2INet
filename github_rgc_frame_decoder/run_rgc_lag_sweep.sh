#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LAGS="${LAGS:-0 1 2 3 4 5 6}"
EPOCHS="${EPOCHS:-60}"
BATCH_SIZE="${BATCH_SIZE:-96}"
BASE_RUN_PREFIX="${BASE_RUN_PREFIX:-rgc_lag_sweep}"

for LAG in $LAGS; do
  RUN_NAME="${BASE_RUN_PREFIX}_lag${LAG}_h${HISTORY_BINS:-11}_k${TRAIN_REPEATS:-8}"
  echo "=== Running lag=${LAG} -> ${RUN_NAME} ==="
  RUN_NAME="$RUN_NAME" \
  RESPONSE_LAG="$LAG" \
  EPOCHS="$EPOCHS" \
  BATCH_SIZE="$BATCH_SIZE" \
  bash run_rgc_frame_decoder.sh
done

python summarize_rgc_frame_runs.py "${OUTPUT_DIR:-runs_rgc_frame}" --prefix "$BASE_RUN_PREFIX"

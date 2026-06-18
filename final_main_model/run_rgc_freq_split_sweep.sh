#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

LOW_MODES="${LOW_MODES:-ridge rf_template hybrid}"
BLURS="${BLURS:-3.0 5.0 7.0}"
RADII="${RADII:-0 4 6 8 10}"

for MODE in $LOW_MODES; do
  for BS in $BLURS; do
    for R in $RADII; do
      RUN_NAME="${BASE_RUN_PREFIX:-fs}_mode${MODE}_blur${BS}_radius${R}" \
      FREQ_LOW_MODE="$MODE" \
      FREQ_BLUR_SIGMA="$BS" \
      FREQ_HIGH_RF_RADIUS="$R" \
      bash run_rgc_freq_split_decoder.sh
    done
  done
done

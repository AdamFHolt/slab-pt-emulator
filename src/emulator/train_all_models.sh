#!/bin/bash

set -euo pipefail

# Where preprocessed datasets live and where to store trained models
DATA_ROOT="./data"
OUT_ROOT="./models"

# GP config
KERNEL="rbf"
RESTARTS=20
LS_LOW=1e-3
LS_HIGH=1e3
NOISE_INIT=3e-3
NOISE_LOW=1e-6
NOISE_HIGH=1     # relaxed upper bound from 0.1 -> 1

DEPTHS=(25 50 75)
FEATURE_SETS=(
  "dTdt"
  "dTdt_thermalParam"
  "dTdt_etaRatio"               
  "dTdt_thermalParam_etaRatio"
)

for d in "${DEPTHS[@]}"; do
  for fs in "${FEATURE_SETS[@]}"; do
    DATA_NAME="${d}km_${fs}"
    echo "------"
    echo "Training: depth=${d}km, variant=${fs}, kernel=${KERNEL}"
    python train_emulator.py \
      --data-root "${DATA_ROOT}" \
      --data-name "${DATA_NAME}" \
      --model gp \
      --kernel "${KERNEL}" \
      --gp-restarts "${RESTARTS}" \
      --ls-bounds "${LS_LOW}" "${LS_HIGH}" \
      --noise-init "${NOISE_INIT}" \
      --noise-bounds "${NOISE_LOW}" "${NOISE_HIGH}" \
      --out "${OUT_ROOT}"
  done
done

echo "All trainings complete."

#!/bin/bash
set -euo pipefail

# Path to your master CSVs (adjust if needed)
MASTER_BASE="../../subd-model-runs/run-outputs/analysis"

# Common args
TARGET="dTdt_C_per_Myr"
MAX_DTD_T=-20

DEPTHS=(25 50 75)
FEATURE_COMBOS=(
  ""                        # baseline
  "--add-thermal-param"
  "--add-eta-ratio"
  "--add-thermal-param --add-eta-ratio"
)

for d in "${DEPTHS[@]}"; do
  for combo in "${FEATURE_COMBOS[@]}"; do
    echo "------"
    echo "Processing: ${d} km (${combo:-no extra features})"
    python preprocess_training.py \
      --master "${MASTER_BASE}/master_${d}km_DT1-6.csv" \
      --targets "${TARGET}" \
      --max-dTdt "${MAX_DTD_T}" \
      ${combo}
  done
done

echo "All preprocessing complete."

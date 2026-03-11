#!/usr/bin/env bash
set -euo pipefail

# Legacy helper retained for reference.
# Prefer the documented Makefile/config-driven workflow for routine use.

SUITE="${1:-const-vc}"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MASTER_BASE="$SCRIPT_DIR/../../../subd-model-runs/$SUITE/analysis"
MASTER="${MASTER_BASE}/master_DT1-10.csv"
PARAMS="$SCRIPT_DIR/../../../data/params/params-list.$SUITE.csv"
ODIR="$SCRIPT_DIR/../data/$SUITE"

DEPTHS=(10 20 30 40 50 60 70 80)
FEATURE_COMBOS=(
  ""                       
  "--add-thermal-param"
  "--add-thermal-param --add-eta-ratio"
)

for d in "${DEPTHS[@]}"; do
  for combo in "${FEATURE_COMBOS[@]}"; do
    echo "Preprocessing for depth ${d} km with combo: ${combo}"
    python "$SCRIPT_DIR/../single_depth/preprocess_single_depth_training.py" \
      --master "${MASTER}" \
      --params "${PARAMS}" \
      --outdir "${ODIR}" \
      --depth-km "${d}" \
      --targets "dTdt_C_per_Myr" \
      --suite "${SUITE}" \
      ${combo}
  done
done

echo "All preprocessing complete."

#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./make_science_emulator_plots.sh const-vc
# Optional env overrides:
#   DEPTHS="10 40 70" VARIANT=dTdt MODEL_TAG=gp_m25

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SCIENCE_DIR="${SCRIPT_DIR}/science"

SUITE="${1:-${SUITE:-const-vc}}"
DEPTHS="${DEPTHS:-10 40 70}"
VARIANT="${VARIANT:-dTdt}"
MODEL_TAG="${MODEL_TAG:-gp_m25}"

echo "== Science emulator plots =="
echo "suite    : ${SUITE}"
echo "depths   : ${DEPTHS}"
echo "variant  : ${VARIANT}"
echo "model    : ${MODEL_TAG}"
echo "out root : ${REPO_ROOT}/plots/science-emulator"
echo

echo "[1/2] all-depths signed sensitivity summary"
python3 "${SCIENCE_DIR}/plot_single_depth_sensitivity_all_depths.py" \
  --suite "${SUITE}" \
  --variant "${VARIANT}" \
  --model-tag "${MODEL_TAG}"

echo "  [2/2] representative single-depth partial-dependence summaries"
for depth in ${DEPTHS}; do
  data_name="${depth}km_${VARIANT}"
  echo "    - ${data_name}"
  python3 "${SCIENCE_DIR}/plot_single_depth_sensitivity.py" \
    --suite "${SUITE}" \
    --data-name "${data_name}" \
    --model-tag "${MODEL_TAG}"
done

echo "== Done =="

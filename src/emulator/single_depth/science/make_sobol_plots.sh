#!/usr/bin/env bash
set -euo pipefail

# Compute Sobol sensitivity indices for the single-depth dTdt emulators across
# the full depth grid, then render per-emulator and vs-depth figures.
#
# Usage:
#   ./make_sobol_plots.sh                 # both suites
#   ./make_sobol_plots.sh const-vc        # one suite
# Optional env overrides:
#   SUITES="const-vc ramped-vc"
#   DEPTHS="5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80"
#   VARIANT=dTdt MODEL_TAG=gp_m25 N_BASE=1024

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -ge 1 ]; then
  SUITES="$1"
else
  SUITES="${SUITES:-const-vc ramped-vc}"
fi
DEPTHS="${DEPTHS:-5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80}"
VARIANT="${VARIANT:-dTdt}"
MODEL_TAG="${MODEL_TAG:-gp_m25}"
N_BASE="${N_BASE:-1024}"

echo "== Sobol sensitivity plots =="
echo "suites   : ${SUITES}"
echo "depths   : ${DEPTHS}"
echo "variant  : ${VARIANT}"
echo "model    : ${MODEL_TAG}"
echo "n_base   : ${N_BASE}"
echo

for suite in ${SUITES}; do
  echo "## suite: ${suite}"
  for depth in ${DEPTHS}; do
    data_name="${depth}km_${VARIANT}"
    echo "  - compute ${data_name}"
    python3 "${SCRIPT_DIR}/compute_sobol_sensitivity.py" \
      --suite "${suite}" \
      --data-name "${data_name}" \
      --model-tag "${MODEL_TAG}" \
      --n-base "${N_BASE}"
    python3 "${SCRIPT_DIR}/plot_sobol_sensitivity.py" \
      --suite "${suite}" \
      --data-name "${data_name}" \
      --model-tag "${MODEL_TAG}"
  done
  echo "  - vs-depth summary (${suite})"
  python3 "${SCRIPT_DIR}/plot_sobol_vs_depth.py" \
    --suite "${suite}" \
    --data-suffix "${VARIANT}" \
    --model-tag "${MODEL_TAG}"
done

echo "== Done =="

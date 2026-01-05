#!/usr/bin/env bash
set -euo pipefail


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Command-line SUITE -----------------------------------------------------
SUITE="${1:-const-vc}"   # const-vc or ramped-vc

# --- Configuration ----------------------------------------------------------

OUT_DIR="../../plots/qc-numerical-mods"
mkdir -p "${OUT_DIR}"
mkdir -p "${OUT_DIR}/${SUITE}"


# =========================================================================== #
# 8) Vertical T profiles (all runs; times 0, 2.5, 5 Myr)
# =========================================================================== #
echo
echo "=== QC: Vertical T profiles (all runs) ==="

TPROF_GLOB="../../subd-model-runs/${SUITE}/analysis/run_*/Tprof_*.csv"
OUT_PREFIX_TPROF="${OUT_DIR}/${SUITE}/Tprofiles_t0t2.5t5"

python qc_T_profiles_allmodels_3times.debug.py \
  --glob "${TPROF_GLOB}" \
  --times 0.5 2.5 5 \
  --tol-myr 0.1 \
  --out "${OUT_PREFIX_TPROF}" \

# =========================================================================== #

echo
echo "All QC plots done. Saved under: ${OUT_DIR}/${SUITE}/"


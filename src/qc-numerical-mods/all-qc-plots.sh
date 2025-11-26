#!/usr/bin/env bash
set -euo pipefail

# all-qc-plots.sh
#
# Generate QC plots:
#   - single-depth:  5,10,...,80 km
#   - three-depth:   20, 50, 80 km
#
# Assumes this script is in: src/qc-numerical-mods
# and the Python scripts are:
#   cooling-rates_all-mods_1depth.py
#   cooling-rates_all-mods_3depths.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Configuration ----------------------------------------------------------
SUITE="const-vc"
TSTEP1=1
TSTEP2=10
YVAR="dTdt_C_per_Myr"

PARAMS="../../data/params/params-list.${SUITE}.csv"
MASTER="../../subd-model-runs/${SUITE}/analysis/master_DT${TSTEP1}-${TSTEP2}.csv"
OUT_DIR="../../plots/qc-numerical-mods"

mkdir -p "${OUT_DIR}"

# --- 1) Single-depth plots: 5,10,...,80 km --------------------------------
echo "=== Single-depth QC plots (5-80 km) ==="
for depth in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80; do
  OUT_PREFIX="${OUT_DIR}/${SUITE}_DT${TSTEP1}-${TSTEP2}_cooling-vs-params_${depth}km"
  echo "  [1depth] depth = ${depth} km → ${OUT_PREFIX}.png"

  python cooling-rates_all-mods_1depth.py \
    --params "${PARAMS}" \
    --master "${MASTER}" \
    --depth-km "${depth}" \
    --y "${YVAR}" \
    --out "${OUT_PREFIX}"
done

# --- 2) Three-depth plot: 20,50,80 km --------------------------------------
echo
echo "=== Three-depth QC plot (20, 50, 80 km) ==="
OUT_PREFIX_3="${OUT_DIR}/${SUITE}_DT${TSTEP1}-${TSTEP2}_cooling-vs-params_20km50km80km"

python cooling-rates_all-mods_3depths.py \
  --params "${PARAMS}" \
  --master "${MASTER}" \
  --depths 20 50 80 \
  --y "${YVAR}" \
  --out "${OUT_PREFIX_3}"

echo
echo "All QC plots done. Saved under: ${OUT_DIR}"

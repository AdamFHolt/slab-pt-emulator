#!/usr/bin/env bash
set -euo pipefail

# qc_make_all_plots.sh
#
# Generate QC plots:
#   - single-depth (5,10,...,80 km)
#   - three-depth (20, 50, 80 km)
#   - dt/dt vs depth panels
#   - histograms for 20,50,80 km
#
# Assumes this script lives in: src/qc-numerical-mods
# and Python scripts in the same folder:
#   qc_dTdt_vs_params_1depth.py
#   qc_dTdt_vs_params_3depths.py
#   qc_dTdt_vs_depth_3depths.py
#   qc_hist_dTdt_3depths.py

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

# =========================================================================== #
# 1) Single-depth QC plots: 5,10,...,80 km
# =========================================================================== #
echo "=== Single-depth QC plots (5–80 km) ==="

for depth in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80; do

  OUT_PREFIX="${OUT_DIR}/${SUITE}_DT${TSTEP1}-${TSTEP2}_dTdt-vs-params_${depth}km"

  echo "  [1depth] depth = ${depth} km → ${OUT_PREFIX}.png"

  python qc_dTdt_vs_params_1depth.py \
    --params "${PARAMS}" \
    --master "${MASTER}" \
    --depth-km "${depth}" \
    --y "${YVAR}" \
    --out "${OUT_PREFIX}"
done

# =========================================================================== #
# 2) Multi-depth QC: 20, 50, 80 km — dT/dt vs params
# =========================================================================== #
echo
echo "=== Three-depth QC plot (20, 50, 80 km) ==="

OUT_PREFIX_3="${OUT_DIR}/${SUITE}_DT${TSTEP1}-${TSTEP2}_dTdt-vs-params_20km50km80km"

python qc_dTdt_vs_params_3depths.py \
  --params "${PARAMS}" \
  --master "${MASTER}" \
  --depths 20 50 80 \
  --y "${YVAR}" \
  --out "${OUT_PREFIX_3}"

# =========================================================================== #
# 3) Three-depth dT/dt vs depth scatter panel
# =========================================================================== #
echo
echo "=== QC: dT/dt vs depth panels (20, 50, 80 km) ==="

OUT_PREFIX_DTDEPTH="${OUT_DIR}/${SUITE}_DT${TSTEP1}-${TSTEP2}_dTdt-vs-depth_20km50km80km"

python qc_dTdt_vs_depth_3depths.py \
  --master "${MASTER}" \
  --depths 20 50 80 \
  --y "${YVAR}" \
  --out "${OUT_PREFIX_DTDEPTH}"

# =========================================================================== #
# 4) Histograms for 20, 50, 80 km
# =========================================================================== #
echo
echo "=== QC: Histograms of dT/dt (20, 50, 80 km) ==="

OUT_PREFIX_HIST="${OUT_DIR}/${SUITE}_DT${TSTEP1}-${TSTEP2}_hist-dTdt_20km50km80km"

python qc_hist_dTdt_3depths.py \
  --params "${PARAMS}" \
  --master "${MASTER}" \
  --depths 20 50 80 \
  --y "${YVAR}" \
  --out "${OUT_PREFIX_HIST}"

# =========================================================================== #
# 5) LHS parameter pairplot
# =========================================================================== #
echo
echo "=== QC: LHS parameter pairplot (${SUITE}) ==="

OUT_PREFIX_PAIR="${OUT_DIR}/${SUITE}_params-pairplot"

python qc_pairplot_params.py \
  --params "${PARAMS}" \
  --out "${OUT_PREFIX_PAIR}" \
  --mode compute-log10


# =========================================================================== #



echo
echo "All QC plots done. Saved under: ${OUT_DIR}"


#!/usr/bin/env bash
set -euo pipefail

# qc_make_all_plots.sh
#
# Core QC plots (default):
#   - dT/dt vs params (single depths)
#   - dT/dt vs params (multi-depth panels)
#   - dT/dt vs depth panel
#   - dT/dt histograms
#   - vertical T profiles (all runs, selected times)
#   - master DT health check
#   - depth x parameter correlation heatmap
#
# Additional diagnostics are included below as commented optional blocks.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- Command-line SUITE -----------------------------------------------------
SUITE="${1:-const-vc}"   # const-vc or ramped-vc

# --- Configuration ----------------------------------------------------------
TSTEP1=1
TSTEP2=10
YVAR="dTdt_C_per_Myr"

PARAMS="../../data/params/params-list.${SUITE}.csv"
MASTER="../../subd-model-runs/${SUITE}/analysis/master_DT${TSTEP1}-${TSTEP2}.csv"

OUT_DIR="../../plots/qc-numerical-mods"
mkdir -p "${OUT_DIR}"
mkdir -p "${OUT_DIR}/${SUITE}"

# =========================================================================== #
# 1) Single-depth QC plots: 5,10,...,80 km
# =========================================================================== #
echo "=== Single-depth QC plots (5-80 km) ==="

for depth in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80; do

  OUT_PREFIX="${OUT_DIR}/${SUITE}/DT${TSTEP1}-${TSTEP2}_dTdt-vs-params_${depth}km"

  echo "  [1depth] depth = ${depth} km -> ${OUT_PREFIX}.png"

  python qc_dTdt_vs_params_1depth.py \
    --params "${PARAMS}" \
    --master "${MASTER}" \
    --depth-km "${depth}" \
    --y "${YVAR}" \
    --out "${OUT_PREFIX}"
done

# =========================================================================== #
# 2) Multi-depth QC: 20, 50, 80 km - dT/dt vs params
# =========================================================================== #
echo
echo "=== Three-depth QC plot (20, 50, 80 km) ==="

OUT_PREFIX_3="${OUT_DIR}/${SUITE}/DT${TSTEP1}-${TSTEP2}_dTdt-vs-params_20km50km80km"

python qc_dTdt_vs_params_mult-depths.py \
  --params "${PARAMS}" \
  --master "${MASTER}" \
  --depths 20 50 80 \
  --y "${YVAR}" \
  --out "${OUT_PREFIX_3}"

# =========================================================================== #
# 3) Multi-depth QC: 20, 40, 60, 80 km - dT/dt vs params
# =========================================================================== #
echo
echo "=== Four-depth QC plot (20, 40, 60, 80 km) ==="

OUT_PREFIX_4="${OUT_DIR}/${SUITE}/DT${TSTEP1}-${TSTEP2}_dTdt-vs-params_20km40km60km80km"

python qc_dTdt_vs_params_mult-depths.py \
  --params "${PARAMS}" \
  --master "${MASTER}" \
  --depths 20 40 60 80 \
  --y "${YVAR}" \
  --out "${OUT_PREFIX_4}"

# =========================================================================== #
# 4) Three-depth dT/dt vs depth scatter panel
# =========================================================================== #
echo
echo "=== QC: dT/dt vs depth panels (20, 50, 80 km) ==="

OUT_PREFIX_DTDEPTH="${OUT_DIR}/${SUITE}/DT${TSTEP1}-${TSTEP2}_dTdt-vs-depth_20km50km80km"

python qc_dTdt_vs_depth_mult-depths.py \
  --master "${MASTER}" \
  --depths 20 50 80 \
  --y "${YVAR}" \
  --out "${OUT_PREFIX_DTDEPTH}"

# =========================================================================== #
# 5) Histograms for 20, 50, 80 km
# =========================================================================== #
echo
echo "=== QC: Histograms of dT/dt (20, 50, 80 km) ==="

OUT_PREFIX_HIST="${OUT_DIR}/${SUITE}/DT${TSTEP1}-${TSTEP2}_hist-dTdt_20km50km80km"

python qc_hist_dTdt_mult-depths.py \
  --params "${PARAMS}" \
  --master "${MASTER}" \
  --depths 20 50 80 \
  --y "${YVAR}" \
  --out "${OUT_PREFIX_HIST}"

# =========================================================================== #
# 6) Vertical T profiles (all runs; times 0.5, 2.5, 5 Myr)
# =========================================================================== #
echo
echo "=== QC: Vertical T profiles (all runs) ==="

TPROF_GLOB="../../subd-model-runs/${SUITE}/analysis/run_*/Tprof_*.csv"
OUT_PREFIX_TPROF="${OUT_DIR}/${SUITE}/Tprofiles_t0t2.5t5"

python qc_T_profiles_allmodels_3times.py \
  --glob "${TPROF_GLOB}" \
  --times 0.5 2.5 5 \
  --tol-myr 0.1 \
  --out "${OUT_PREFIX_TPROF}"

# =========================================================================== #
# 7) Master DT health checks before ML preprocessing
# =========================================================================== #
echo
echo "=== QC: master DT health checks (${SUITE}) ==="

OUT_PREFIX_HEALTH="${OUT_DIR}/${SUITE}/master_DT${TSTEP1}-${TSTEP2}_healthcheck"

python qc_master_dTdt_healthcheck.py \
  --master "${MASTER}" \
  --y "${YVAR}" \
  --out "${OUT_PREFIX_HEALTH}"

# =========================================================================== #
# 8) Depth x parameter correlation heatmap (core pre-ML diagnostic)
# =========================================================================== #
echo
echo "=== QC: depth x parameter correlation heatmap (${SUITE}) ==="

OUT_PREFIX_CORR="${OUT_DIR}/${SUITE}/param_depth_corr_dTdt_pearson"
python qc_param_depth_correlation_heatmap.py \
  --params "${PARAMS}" \
  --master "${MASTER}" \
  --y "${YVAR}" \
  --method pearson \
  --out "${OUT_PREFIX_CORR}"

# =========================================================================== #
# 9) Optional: LHS parameter pairplot
# =========================================================================== #
#
# OUT_PREFIX_PAIR="${OUT_DIR}/${SUITE}/params-pairplot"
# python qc_pairplot_params.py \
#   --params "${PARAMS}" \
#   --out "${OUT_PREFIX_PAIR}" \
#   --mode compute-log10

# =========================================================================== #
# 10) Optional: LHS parameter pairplot colored by dT/dt (10, 50, 70 km)
# =========================================================================== #
#
# for depth in 10 50 70; do
#   OUT_PREFIX_PAIR_COLOR="${OUT_DIR}/${SUITE}/params-pairplot_colored_${depth}km"
#
#   python qc_pairplot_params_colored.py \
#     --params "${PARAMS}" \
#     --master "${MASTER}" \
#     --depth-km "${depth}" \
#     --y "${YVAR}" \
#     --out "${OUT_PREFIX_PAIR_COLOR}" \
#     --mode compute-log10
# done

# =========================================================================== #
# 11) Optional: per-run depth-time heatmaps from Tprof_*.csv
# =========================================================================== #
#
# OUT_DIR_TPROF_HEAT="${OUT_DIR}/${SUITE}/Tprof_heatmaps"
# python qc_Tprof_depth_time_heatmap.py \
#   --glob "../../subd-model-runs/${SUITE}/analysis/run_*/Tprof_*.csv" \
#   --out-dir "${OUT_DIR_TPROF_HEAT}"
#
# Quick test on first few runs only:
# python qc_Tprof_depth_time_heatmap.py \
#   --glob "../../subd-model-runs/${SUITE}/analysis/run_*/Tprof_*.csv" \
#   --out-dir "${OUT_DIR_TPROF_HEAT}" \
#   --max-runs 10

# =========================================================================== #
# 12) Optional: cluster map of runs from full depth profiles
# =========================================================================== #
#
# OUT_PREFIX_CLUST="${OUT_DIR}/${SUITE}/run_clusters_dTdt_k4"
# python qc_run_clustering_map.py \
#   --params "${PARAMS}" \
#   --master "${MASTER}" \
#   --y "${YVAR}" \
#   --n-clusters 4 \
#   --out "${OUT_PREFIX_CLUST}"
# # Add --save-csv if you also want run_id -> cluster assignments.

# =========================================================================== #

echo
echo "All QC plots done. Saved under: ${OUT_DIR}/${SUITE}/"

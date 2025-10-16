#!/usr/bin/env bash
set -euo pipefail

# ---------------------------
# all-qc-plots.sh
# Run all QC plots from anywhere.
# ---------------------------

usage() {
  cat <<EOF
Usage: $(basename "$0") [D1] [D2] [T1] [T2]

  D1, D2     Depths in km (default: 25 50)
  T1, T2     Timestep indices used for ΔT (default: 1 6)

Environment:
  SLABPT_ROOT   If set, use as repo root; otherwise auto-detect two levels up from this script.

Examples:
  $(basename "$0")
  $(basename "$0") 25 75 1 6
EOF
}

# ---- parse args (positionals with defaults) ----
D1="${1:-25}"
D2="${2:-50}"
D3="${3:-75}"
T1="${4:-1}"
T2="${5:-6}"
YVAR="dTdt_C_per_Myr"
YVAR2="dT_C"

# ---- locate repo root & important dirs ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLABPT_ROOT:-"$(cd "$SCRIPT_DIR/../.." && pwd)"}"   # src/QC-model-runs/ -> repo root
DATA_DIR="$REPO_ROOT/data"
ANALYSIS_DIR="$REPO_ROOT/subd-model-runs/run-outputs/analysis"
PLOTS_DIR="$REPO_ROOT/plots/numerical-models"
QC_DIR="$SCRIPT_DIR"   # where the qc_*.py live

# ---- inputs & outputs ----
PARAMS="$DATA_DIR/params-list.csv"
MASTER1="$ANALYSIS_DIR/master_${D1}km_DT${T1}-${T2}.csv"
MASTER2="$ANALYSIS_DIR/master_${D2}km_DT${T1}-${T2}.csv"
MASTER3="$ANALYSIS_DIR/master_${D3}km_DT${T1}-${T2}.csv"

mkdir -p "$PLOTS_DIR" 

# ---- sanity checks ----
[[ -f "$PARAMS" ]] || { echo "ERROR: params CSV not found: $PARAMS"; exit 2; }
[[ -f "$MASTER1" ]] || { echo "ERROR: master CSV not found: $MASTER1"; exit 2; }
[[ -f "$MASTER2" ]] || { echo "ERROR: master CSV not found: $MASTER2"; exit 2; }

# ---- echo config ----
echo "== QC config =="
echo "Repo root:   $REPO_ROOT"
echo "Params:      $PARAMS"
echo "Masters:     $MASTER1"
echo "             $MASTER2"
echo "Y variable:  $YVAR"
echo "Outputs:     $PLOTS_DIR"
echo "==============="
echo

# ---- 1) Pairwise parameter correlations (corner plot) ----
python3 "$QC_DIR/qc_pairplot_params.py" \
  --params "$PARAMS" \
  --out "$PLOTS_DIR/qc_pairplot_params" \

# ---- 2) Histograms (params + responses) for D1 ----
python3 "$QC_DIR/qc_histograms.py" \
  --params "$PARAMS" \
  --master "$MASTER1" \
  --out "$PLOTS_DIR/qc_histograms_${D1}km" \

# ---- 3) ΔT or ΔT/Δt vs depth panels ----
python3 "$QC_DIR/qc_dt_vs_depth.py" \
  --masters "$MASTER1" "$MASTER2" \
  --y "$YVAR" \
  --out "$PLOTS_DIR/qc_DT_vs_depth_${D1}-${D2}km" \
  --dpi 220

# ---- 4) ΔT or ΔT/Δt vs parameters (single depth D1) ----
python3 "$QC_DIR/qc_cooling-rates_all-mods.py" \
  --params "$PARAMS" \
  --master "$MASTER1" \
  --y "$YVAR" \
  --out "$PLOTS_DIR/DT_vs_params_${D1}km" \

python3 "$QC_DIR/qc_cooling-rates_all-mods.py" \
  --params "$PARAMS" \
  --master "$MASTER2" \
  --y "$YVAR" \
  --out "$PLOTS_DIR/DT_vs_params_${D2}km" \

python3 "$QC_DIR/qc_cooling-rates_all-mods.py" \
  --params "$PARAMS" \
  --master "$MASTER3" \
  --y "$YVAR" \
  --out "$PLOTS_DIR/DT_vs_params_${D3}km" \

# ---- 5) ΔT or ΔT/Δt vs parameters (two depths overlay) ----
python3 "$QC_DIR/qc_cooling-rates_all-mods_2-depths.py" \
  --params "$PARAMS" \
  --master1 "$MASTER1" \
  --master2 "$MASTER2" \
  --y "$YVAR" \
  --out "$PLOTS_DIR/DT_vs_params_${D1}-${D2}km" \

python3 "$QC_DIR/qc_cooling-rates_all-mods_3-depths.py" \
  --params "$PARAMS" \
  --masters "$MASTER1" "$MASTER2" "$MASTER3"  \
  --y "$YVAR" \
  --out "$PLOTS_DIR/DT_vs_params_${D1}-${D2}-${D3}km" \


echo "Done. Plots in: $PLOTS_DIR"

#!/usr/bin/env bash
set -uo pipefail
# Build every 0.5-10 Myr emulator product one suite is missing, in the order the
# const-vc set was built (2026-09-17), so the two suites end up on the same footing:
#   1. profile-PCA series at steps 1..20 (run_10myr_series.sh: preprocess, train, quality)
#      + the by-time RMSE table (summarize_10myr_quality.py)
#   2. single-depth dTdt windows 5-10 Myr (dt10-20) and 0.5-10 Myr (dt1-20):
#      preprocess, train, Sobol (run_window_sobol.sh) + sobol_windows_summary.csv
#   3. the 3-panel validation + Sobol summary figure (plot_emulator_validation_sobol.py)
#
# Usage:  ./build_10myr_products.sh SUITE [NPROC]     (from anywhere; logs to stdout)
# Needs subd-model-runs/<suite>/analysis/logs-extend/worklist_full_depth.txt (fixed run set:
# runs with Tprof_1..20 present and finite on 0-80 km) and master_DT{1-20,10-20}.csv.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY="${ROOT}/env/bin/python"
SUITE="${1:?usage: build_10myr_products.sh SUITE [NPROC]}"
NPROC="${2:-10}"
RUNS_FILE="${ROOT}/subd-model-runs/${SUITE}/analysis/logs-extend/worklist_full_depth.txt"
cd "$ROOT"
[[ -f "$RUNS_FILE" ]] || { echo "missing $RUNS_FILE" >&2; exit 1; }
t0=$(date +%s); stamp() { echo "[$(date +%H:%M:%S) +$(( ($(date +%s)-t0)/60 ))min] $*"; }

stamp "1/3 profile-PCA 0.5-10 Myr series for ${SUITE} ($(wc -l < "$RUNS_FILE") runs, NPROC=${NPROC})"
SUITE="$SUITE" src/emulator/profile_pca/run_10myr_series.sh "$RUNS_FILE" all "$NPROC" || echo "[FAIL] 10myr series"
mkdir -p plots/qc-emulator/profile-pca/10myr
"$PY" src/emulator/profile_pca/summarize_10myr_quality.py \
  --models-root "src/emulator/models/profile_pca_10myr/${SUITE}/runs" \
  --out "plots/qc-emulator/profile-pca/10myr/${SUITE}_profile_rmse_by_time.csv" || echo "[FAIL] 10myr summary"

for w in "dt10-20 10 20" "dt1-20 1 20"; do
  set -- $w
  stamp "2/3 single-depth window $1 for ${SUITE}"
  SUITE="$SUITE" src/emulator/single_depth/run_window_sobol.sh "$1" "$2" "$3" all || echo "[FAIL] window $1"
done
SD="plots/science-emulator/single_depth/${SUITE}"
"$PY" src/emulator/single_depth/science/summarize_sobol_vs_depth.py \
  --window "0.5-5 Myr=${SD}/sobol" --window "0.5-10 Myr=${SD}/sobol_dt1-20" \
  --window "5-10 Myr=${SD}/sobol_dt10-20" --crossover age_OP,v_conv \
  --out "${SD}/sobol_windows_summary.csv" || echo "[FAIL] sobol summary"

stamp "3/3 summary figure"
"$PY" src/emulator/science/plot_emulator_validation_sobol.py "$SUITE" || echo "[FAIL] figure"
stamp "done"

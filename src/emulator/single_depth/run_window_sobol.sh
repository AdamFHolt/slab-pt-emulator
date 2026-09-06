#!/usr/bin/env bash
set -euo pipefail

# Preprocess -> train -> Sobol for one const-vc cooling-rate window.
#
# The window is identified by the master table it comes from, e.g.
#   master_DT1-20.csv   -> tag dt1-20   (0.5-10 Myr mean cooling rate)
#   master_DT10-20.csv  -> tag dt10-20  (5-10 Myr mean cooling rate)
#
# Datasets/models/plots are kept in tag-specific roots so the existing
# 0.5-5 Myr (master_DT1-10.csv) products are never overwritten.
#
# Usage:
#   ./run_window_sobol.sh TAG T1 T2 [STAGE]
#     MASTER=<path> overrides the master table the tag is built from.
#     STAGE = all | preprocess | train | sobol   (default: all)
#
# Example:
#   ./run_window_sobol.sh dt1-20 1 20

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PY="${ROOT}/env/bin/python"
[[ -x "$PY" ]] || PY="python3"

TAG="${1:?usage: run_window_sobol.sh TAG T1 T2 [STAGE]}"
T1="${2:?}"
T2="${3:?}"
STAGE="${4:-all}"

SUITE=const-vc
DEPTHS="${DEPTHS:-5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80}"
N_BASE="${N_BASE:-1024}"
MODEL_TAG=gp_m25

MASTER="${MASTER:-${ROOT}/subd-model-runs/${SUITE}/analysis/master_DT${T1}-${T2}.csv}"
PARAMS="${ROOT}/data/params/params-list.${SUITE}.csv"
DATA_ROOT="${ROOT}/src/emulator/data/single_depth_${TAG}"
MODEL_ROOT="${ROOT}/src/emulator/models/single_depth_${TAG}"
SOBOL_OUT="${ROOT}/plots/science-emulator/single_depth/${SUITE}/sobol_${TAG}"
CFG="configs/gp.${SUITE}.${TAG}.yaml"

cd "$ROOT"

if [[ "$STAGE" == "all" || "$STAGE" == "preprocess" ]]; then
  [[ -f "$MASTER" ]] || { echo "missing master table: $MASTER" >&2; exit 1; }
  for d in ${DEPTHS}; do
    echo "[preprocess] ${TAG} depth=${d} km"
    "$PY" src/emulator/single_depth/core/preprocess_single_depth_training.py \
      --master "$MASTER" \
      --params "$PARAMS" \
      --outdir "${DATA_ROOT}/${SUITE}/runs" \
      --depth-km "$d" \
      --targets "dTdt_C_per_Myr" \
      --suite "$SUITE"
  done
fi

if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
  "$PY" train.py --config "$CFG"
fi

if [[ "$STAGE" == "all" || "$STAGE" == "sobol" ]]; then
  mkdir -p "$SOBOL_OUT"
  for d in ${DEPTHS}; do
    echo "[sobol] ${TAG} depth=${d} km"
    "$PY" src/emulator/single_depth/science/compute_sobol_sensitivity.py \
      --suite "$SUITE" \
      --data-name "${d}km_dTdt" \
      --model-tag "$MODEL_TAG" \
      --data-root "${DATA_ROOT}" \
      --models-root "${MODEL_ROOT}" \
      --outdir "$SOBOL_OUT" \
      --n-base "$N_BASE"
  done
fi

echo "[OK] ${TAG} stage=${STAGE} complete"

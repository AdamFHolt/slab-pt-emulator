#!/usr/bin/env bash
set -euo pipefail

# Build, train and score the const-vc profile-PCA emulators at every output
# step of the extended 0.5-10 Myr window (steps 1..20, 0.5 Myr apart).
#
# All 20 times share one fixed run set (--runs-file), so the 85/15 split by run
# is identical at every time and the per-time quality numbers are comparable.
#
# Usage:
#   ./run_10myr_series.sh RUNS_FILE [STAGE]
#     STAGE = all | preprocess | train | quality   (default: all)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PY="${ROOT}/env/bin/python"
[[ -x "$PY" ]] || PY="python3"

RUNS_FILE="${1:?usage: run_10myr_series.sh RUNS_FILE [STAGE]}"
STAGE="${2:-all}"

SUITE="const-vc"
K=10
SCORE_SPACE=whitened
CFG="configs/gp.${SUITE}.profile-pca.10myr.yaml"
DATA_ROOT="${ROOT}/src/emulator/data/profile_pca_10myr/${SUITE}/runs"
MODEL_ROOT="${ROOT}/src/emulator/models/profile_pca_10myr/${SUITE}/runs"

# steps 1..20 -> 0.5 .. 10.0 Myr
TIMES=()
LABELS=()
for k in $(seq 1 20); do
  t=$(awk -v k="$k" 'BEGIN{printf "%g", k*0.5}')
  TIMES+=("$t")
  LABELS+=("$(echo "$t" | sed 's/\./p/g')")
done

cd "$ROOT"

if [[ "$STAGE" == "all" || "$STAGE" == "preprocess" ]]; then
  for i in "${!TIMES[@]}"; do
    t="${TIMES[$i]}"; lab="${LABELS[$i]}"
    dname="profileT_pca_t${lab}Myr_k${K}"
    echo "[preprocess] t=${t} Myr -> ${dname}"
    "$PY" src/emulator/profile_pca/core/preprocess_profile_pca.py \
      --suite "$SUITE" \
      --target-time-myr "$t" \
      --k "$K" \
      --score-space "$SCORE_SPACE" \
      --runs-file "$RUNS_FILE" \
      --dataset-name "$dname" \
      --outdir "$DATA_ROOT"
  done
fi

if [[ "$STAGE" == "all" || "$STAGE" == "train" ]]; then
  for lab in "${LABELS[@]}"; do
    dname="profileT_pca_t${lab}Myr_k${K}"
    echo "[train] ${dname}"
    "$PY" train.py --config "$CFG" --datasets "$dname"
  done
fi

if [[ "$STAGE" == "all" || "$STAGE" == "quality" ]]; then
  for lab in "${LABELS[@]}"; do
    dname="profileT_pca_t${lab}Myr_k${K}"
    echo "[quality] ${dname}"
    "$PY" src/emulator/profile_pca/core/evaluate_profile_pca_quality.py \
      --dataset-dir "${DATA_ROOT}/${dname}" \
      --model-dir "${MODEL_ROOT}/${dname}/gp_m25" \
      --json-out "${MODEL_ROOT}/${dname}/gp_m25/profile_pca_quality.json"
  done
fi

echo "[OK] stage=${STAGE} complete"

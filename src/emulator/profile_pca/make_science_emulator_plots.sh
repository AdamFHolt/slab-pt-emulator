#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./make_science_emulator_plots.sh const-vc
# Optional env overrides:
#   PROFILE_TIMES="0.5 1 2 3 4 5" PROFILE_K=10 MODEL_TAG=gp_m25
#   SENS_TIMES="0.5 3 5" PROFILE_FEATURES="v_conv age_SP age_OP"

SUITE="${1:-${SUITE:-const-vc}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SCIENCE_DIR="${SCRIPT_DIR}/science"

PROFILE_TIMES="${PROFILE_TIMES:-0.5 1 2 3 4 5}"
PROFILE_K="${PROFILE_K:-10}"
MODEL_TAG="${MODEL_TAG:-gp_m25}"
SENS_TIMES="${SENS_TIMES:-0.5 3 5}"
PROFILE_FEATURES="${PROFILE_FEATURES:-v_conv age_SP age_OP}"

echo "== Profile-PCA science plots =="
echo "suite        : ${SUITE}"
echo "times        : ${PROFILE_TIMES}"
echo "k            : ${PROFILE_K}"
echo "model        : ${MODEL_TAG}"
echo "3-panel times: ${SENS_TIMES}"
echo "family vars  : ${PROFILE_FEATURES}"
echo "out root     : ${REPO_ROOT}/plots/science-emulator/profile-pca"
echo

echo "[1/3] time summary"
python3 "${SCIENCE_DIR}/plot_profile_pca_time_summary.py" \
  --suite "${SUITE}" \
  --times "${PROFILE_TIMES}"

echo "[2/3] 3-time sensitivity"
python3 "${SCIENCE_DIR}/plot_profile_pca_sensitivity.py" \
  --suite "${SUITE}" \
  --times "${SENS_TIMES}" \
  --k "${PROFILE_K}" \
  --model-tag "${MODEL_TAG}"

echo "[3/3] profile families"
for feature in ${PROFILE_FEATURES}; do
  echo "    - ${feature}"
  python3 "${SCIENCE_DIR}/plot_profile_pca_profile_family.py" \
    --suite "${SUITE}" \
    --times "${SENS_TIMES}" \
    --feature "${feature}" \
    --k "${PROFILE_K}" \
    --model-tag "${MODEL_TAG}"
done

echo
echo "== Done =="

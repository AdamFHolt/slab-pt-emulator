#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./make_qc_emulator_plots.sh const-vc
# Optional env overrides:
#   DATA_ROOT=... MODELS_ROOT=... PLOTS_ROOT=... SPLIT=val DPI=220 N_SAMPLES=20 SEED=0 PROFILE_TIMES="0.5 1 2 3 4 5" PROFILE_K=10 MODEL_TAG=gp_m25

SUITE="${1:-const-vc}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
QC_DIR="${SCRIPT_DIR}/qc"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/src/emulator/data/profile_pca}"
MODELS_ROOT="${MODELS_ROOT:-${REPO_ROOT}/src/emulator/models/profile_pca}"
PLOTS_ROOT="${PLOTS_ROOT:-${REPO_ROOT}/plots/qc-emulator/profile-pca}"

SPLIT="${SPLIT:-val}"
DPI="${DPI:-220}"
N_SAMPLES="${N_SAMPLES:-20}"
SEED="${SEED:-0}"
PROFILE_TIMES="${PROFILE_TIMES:-0.5 1 2 3 4 5}"
PROFILE_K="${PROFILE_K:-10}"
MODEL_TAG="${MODEL_TAG:-gp_m25}"

RUNS_ROOT="${PLOTS_ROOT}/runs/${SUITE}"
mkdir -p "${RUNS_ROOT}/reconstruction" "${RUNS_ROOT}/emulator-reconstruction" "${RUNS_ROOT}/score-diagnostics"

echo "== Profile-PCA QC plots =="
echo "suite    : ${SUITE}"
echo "split    : ${SPLIT}"
echo "times    : ${PROFILE_TIMES}"
echo "k        : ${PROFILE_K}"
echo "model    : ${MODEL_TAG}"
echo "data     : ${DATA_ROOT}"
echo "models   : ${MODELS_ROOT}"
echo "plots    : ${PLOTS_ROOT}"
echo "samples  : ${N_SAMPLES}"
echo

for t in ${PROFILE_TIMES}; do
  tlabel="$(echo "${t}" | sed 's/\./p/g')"
  dname="profileT_pca_t${tlabel}Myr_k${PROFILE_K}"
  ds="${DATA_ROOT}/${SUITE}/runs/${dname}"
  md="${MODELS_ROOT}/${SUITE}/runs/${dname}/${MODEL_TAG}"

  if [[ ! -d "${ds}" ]]; then
    echo "[WARN] skip missing dataset ${ds}"
    continue
  fi
  if [[ ! -d "${md}" ]]; then
    echo "[WARN] skip missing model ${md}"
    continue
  fi

  echo "[RUN] suite=${SUITE} dataset=${dname} split=${SPLIT}"
  python3 "${QC_DIR}/plot_profile_pca_reconstruction.py" \
    --dataset-dir "${ds}" \
    --split "${SPLIT}" \
    --n-samples "${N_SAMPLES}" \
    --seed "${SEED}" \
    --out "${RUNS_ROOT}/reconstruction/${dname}_true-vs-recon.png"

  python3 "${QC_DIR}/plot_profile_pca_emulator_reconstruction.py" \
    --dataset-dir "${ds}" \
    --model-dir "${md}" \
    --split "${SPLIT}" \
    --n-samples "${N_SAMPLES}" \
    --seed "${SEED}" \
    --out "${RUNS_ROOT}/emulator-reconstruction/${dname}_raw-vs-pca-vs-emu.png"

  python3 "${QC_DIR}/plot_profile_pca_score_diagnostics.py" \
    --dataset-dir "${ds}" \
    --model-dir "${md}" \
    --split "${SPLIT}" \
    --out-prefix "${RUNS_ROOT}/score-diagnostics/${dname}"
done

echo
echo "== Done =="

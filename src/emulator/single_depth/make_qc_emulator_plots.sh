#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./make_qc_emulator_plots.sh const-vc dTdt_thermalParam gp_rbf
# Optional env overrides:
#   DATA_ROOT=... MODELS_ROOT=... PLOTS_ROOT=... DPI=... LABEL_THRESH=... RESIDUAL_COVERAGE=0

SUITE="${1:-const-vc}"
VARIANT="${2:-dTdt_thermalParam}"
ALGO="${3:-gp_m25}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
QC_DIR="${SCRIPT_DIR}/qc"

# Roots default relative to the repo layout, so the script works from repo root
# and from ad hoc shells without depending on the caller's cwd.
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/src/emulator/data/single_depth}"
MODELS_ROOT="${MODELS_ROOT:-${REPO_ROOT}/src/emulator/models/single_depth}"
PLOTS_ROOT="${PLOTS_ROOT:-${REPO_ROOT}/plots/qc-emulator/single_depth}"

DPI="${DPI:-220}"
LABEL_THRESH="${LABEL_THRESH:-8.0}"

# Make residual-colored coverage plots too? (0/1)
RESIDUAL_COVERAGE="${RESIDUAL_COVERAGE:-1}"

echo "== QC emulator plots =="
echo "suite   : ${SUITE}"
echo "variant : ${VARIANT}"
echo "algo    : ${ALGO}"
echo "data    : ${DATA_ROOT}"
echo "models  : ${MODELS_ROOT}"
echo "plots   : ${PLOTS_ROOT}"
echo "dpi     : ${DPI}"
echo "label_th: ${LABEL_THRESH}"
echo

# -------------------------
# 1) Pred vs true depth grid (single figure across all depths)
# -------------------------
echo "[1/3] pred-vs-true depth grid"
python "${QC_DIR}/plot_emulator_vs_true_depthgrid.py" \
  --data-root "${DATA_ROOT}" \
  --models-root "${MODELS_ROOT}" \
  --suite "${SUITE}" \
  --variant "${VARIANT}" \
  --algo "${ALGO}" \
  --outdir "${PLOTS_ROOT}" \
  --dpi "${DPI}"

echo

# -------------------------
# 2) Misfit vs features (one figure per depth)
# -------------------------
echo "[2/3] misfit-vs-features (per depth)"
python "${QC_DIR}/plot_emulator_misfit-vs-params.py" \
  --data-root "${DATA_ROOT}" \
  --models-root "${MODELS_ROOT}" \
  --suite "${SUITE}" \
  --variant "${VARIANT}" \
  --algo "${ALGO}" \
  --outdir "${PLOTS_ROOT}" \
  --label-thresh "${LABEL_THRESH}" \
  --dpi "${DPI}"

echo

# -------------------------
# 3) Parameter coverage (one figure per depth)
# -------------------------
echo "[3/3] param coverage (per depth)"
python "${QC_DIR}/plot_emulator_param-coverage.py" \
  --data-root "${DATA_ROOT}" \
  --models-root "${MODELS_ROOT}" \
  --suite "${SUITE}" \
  --variant "${VARIANT}" \
  --algo "${ALGO}" \
  --color-by none \
  --outdir "${PLOTS_ROOT}" \
  --dpi "${DPI}"

if [[ "${RESIDUAL_COVERAGE}" == "1" ]]; then
  echo
  echo "[3b/3] param coverage colored by |residual| (per depth)"
  python "${QC_DIR}/plot_emulator_param-coverage.py" \
    --data-root "${DATA_ROOT}" \
    --models-root "${MODELS_ROOT}" \
    --suite "${SUITE}" \
    --variant "${VARIANT}" \
    --algo "${ALGO}" \
    --color-by residual \
    --outdir "${PLOTS_ROOT}" \
    --dpi "${DPI}"
fi

echo
echo "== Done =="

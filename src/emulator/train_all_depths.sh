#!/usr/bin/env bash
set -euo pipefail

SUITE="${1:-const-vc}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="$SCRIPT_DIR/data/$SUITE"
OUT_ROOT="$SCRIPT_DIR/models/$SUITE"

# GP config
KERNEL="matern25"
RESTARTS=25
LS_LOW=1e-3
LS_HIGH=1e3
NOISE_INIT=3e-3
NOISE_LOW=1e-6
NOISE_HIGH=1

echo "[INFO] SUITE=$SUITE"
echo "[INFO] DATA_ROOT=$DATA_ROOT"
echo "[INFO] OUT_ROOT=$OUT_ROOT"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERROR] DATA_ROOT does not exist: $DATA_ROOT" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

# Find dataset folders (one level deep)
mapfile -t DATASETS < <(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "[ERROR] No dataset folders found under: $DATA_ROOT" >&2
  exit 1
fi

echo "[INFO] Found ${#DATASETS[@]} datasets:"
printf "  - %s\n" "${DATASETS[@]}"

for data_name in "${DATASETS[@]}"; do
  echo "------"
  echo "Training: suite=${SUITE}, data=${data_name}, kernel=${KERNEL}"
  python3 train_emulator.py \
    --data-root "${DATA_ROOT}" \
    --data-name "${data_name}" \
    --model gp \
    --kernel "${KERNEL}" \
    --gp-restarts "${RESTARTS}" \
    --ls-bounds "${LS_LOW}" "${LS_HIGH}" \
    --noise-init "${NOISE_INIT}" \
    --noise-bounds "${NOISE_LOW}" "${NOISE_HIGH}" \
    --out "${OUT_ROOT}"
done

echo "All trainings complete."

#!/bin/bash
set -euo pipefail

DATA_ROOT="./data"
MODEL_ROOT="./models"
OUT_DIR="./plots"

mkdir -p "$OUT_DIR"

# Define the depth sets and feature variants
DEPTHS=("25km" "50km" "75km")
VARIANTS=("dTdt" "dTdt_thermalParam" "dTdt_etaRatio" "dTdt_thermalParam_etaRatio")
ALGOS=("gp_m25" "gp_m15" "gp_rbf")  

# Loop through each feature variant and plot across all depths
for ALGO in "${ALGOS[@]}"; do
    for VAR in "${VARIANTS[@]}"; do
        echo "------"
        echo "Plotting ${ALGO} emulator performance for variant: ${VAR}"
        python plot_emulator_fit_vs_val.py \
            --data-root "$DATA_ROOT" \
            --models "$MODEL_ROOT" \
            --algo "$ALGO" \
            --names "${DEPTHS[0]}_${VAR}" "${DEPTHS[1]}_${VAR}" "${DEPTHS[2]}_${VAR}" \
            --yidx 0 \
            --out "${OUT_DIR}/emulator_fit_vs_val__${VAR}.${ALGO}.png"
    done
done

echo "------"
echo "All plots complete. Saved to ${OUT_DIR}/"

#!/usr/bin/env bash
set -euo pipefail

# Regenerate the standard single-depth param-sweep comparison plots.
# This wraps plot_param_sweep_compare.py for the datasets we commonly compare in
# notes, docs, and QC review.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PLOT_SCRIPT="$SCRIPT_DIR/plot_param_sweep_compare.py"
OUT_ROOT="$REPO_ROOT/plots/qc-emulator/single_depth"
SWEEP_ROOT="$REPO_ROOT/src/emulator/models/single_depth"

run_one() {
  local suite="$1"
  local dataset="$2"
  python3 "$PLOT_SCRIPT" \
    --suite "$suite" \
    --data-name "$dataset" \
    --sweep-root "$SWEEP_ROOT" \
    --outdir "$OUT_ROOT"
}

for suite in const-vc ramped-vc; do
  run_one "$suite" "40km_dTdt"
  run_one "$suite" "10km_dTdt_thermalParam"
  run_one "$suite" "40km_dTdt_thermalParam"
  run_one "$suite" "70km_dTdt_thermalParam"
done

echo "[OK] Param-sweep comparison plots refreshed under: $OUT_ROOT"

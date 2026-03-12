#!/usr/bin/env bash
set -euo pipefail

# Legacy helper retained for reference.
# Prefer the documented Makefile/config-driven workflow for routine use.

# Train a *single depth* across multiple emulator types/hyperparams,
# then copy each trained model directory into a unique param-sweep folder.
#
# Usage:
#   ./train_one_depth_param_sweep.sh [suite] [depth_km] [variant] [algo_list]
#
# Examples:
#   ./train_one_depth_param_sweep.sh const-vc 40 dTdt_thermalParam
#   ./train_one_depth_param_sweep.sh ramped-vc 40 dTdt gp_rbf,gp_m15,gp_m25,rf
#
# Notes:
# - Assumes you already preprocessed data into: ./data/single_depth/<suite>/<depth>km_<variant>/
# - Assumes train_emulator.py creates: <out>/<data-name>/<model_dir_name>/model.joblib
#   where model_dir_name is e.g. gp_rbf, gp_m15, gp_m25, rf (per your code).

SUITE="${1:-const-vc}"
DEPTH_KM="${2:-40}"
VARIANT="${3:-dTdt}"
ALGO_LIST_CSV="${4:-gp_rbf,gp_m15,gp_m25,rf}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="$SCRIPT_DIR/../data/single_depth/${SUITE}/runs"
DATA_NAME="${DEPTH_KM}km_${VARIANT}"

TRAIN_PY="$SCRIPT_DIR/../train_emulator.py"

# Where train_emulator.py will write its normal structure:
BASE_OUT_ROOT="$SCRIPT_DIR/../models/single_depth/${SUITE}/runs"

# Where *unique copies* of each run will go:
SWEEP_ROOT="$SCRIPT_DIR/../models/single_depth/${SUITE}/param_sweep"


# Make sure dataset exists
if [[ ! -d "$DATA_ROOT/$DATA_NAME" ]]; then
  echo "[ERR] Dataset folder not found: $DATA_ROOT/$DATA_NAME"
  echo "      (expected preprocessed outputs like X_std.npy, Y_std.npy, metadata.json)"
  exit 1
fi

mkdir -p "$BASE_OUT_ROOT"
mkdir -p "$SWEEP_ROOT"
mkdir -p "$SWEEP_ROOT/$DATA_NAME"

# -------------------------
# Define your sweep grid
#
# Edit these arrays however you want. The TAG naming below will prevent collisions.
# -------------------------

# GP common
GP_RESTARTS_LIST=(5 15 25)
LS_INIT_LIST=(1.0)
LS_LOW_LIST=(1e-3)
LS_HIGH_LIST=(1e3)
NOISE_INIT_LIST=(1e-3 3e-3)
NOISE_LOW_LIST=(1e-6)
NOISE_HIGH_LIST=(1.0)
ALPHA_LIST=(1e-6)

# RF common
RF_TREES_LIST=(300 600)
RF_MAX_DEPTH_LIST=("None" 20 40)  # "None" -> unlimited
RF_JOBS=-1

# -------------------------
# Helpers
# -------------------------

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }

# Turn algo dir into model args
# gp_rbf -> --model gp --kernel rbf
# gp_m15 -> --model gp --kernel matern15
# gp_m25 -> --model gp --kernel matern25
# rf     -> --model rf
algo_to_args() {
  local algo="$1"
  case "$algo" in
    gp_rbf) echo "--model gp --kernel rbf" ;;
    gp_m15) echo "--model gp --kernel matern15" ;;
    gp_m25) echo "--model gp --kernel matern25" ;;
    rf)     echo "--model rf" ;;
    *)
      echo "[ERR] Unknown algo '$algo' (expected gp_rbf, gp_m15, gp_m25, rf)" >&2
      exit 1
      ;;
  esac
}

# Make a safe tag string (no spaces)
mk_tag_gp() {
  local kernel="$1" restarts="$2" lsinit="$3" lslow="$4" lshigh="$5" ninit="$6" nlow="$7" nhigh="$8" alpha="$9"
  echo "gp_${kernel}_r${restarts}_ls${lsinit}_b${lslow}-${lshigh}_n${ninit}_nb${nlow}-${nhigh}_a${alpha}"
}
mk_tag_rf() {
  local trees="$1" maxd="$2"
  echo "rf_t${trees}_d${maxd}"
}

# Copy model output dir to unique sweep dir
copy_unique() {
  local src_dir="$1" tag="$2"
  local dst_dir="$SWEEP_ROOT/$DATA_NAME/$tag"
  mkdir -p "$dst_dir"
  rm -rf "$dst_dir"  # overwrite if rerun
  cp -a "$src_dir/." "$dst_dir/"
  log "Copied -> $dst_dir"
}

# Parse CSV list into array
IFS=',' read -r -a ALGOS <<< "$ALGO_LIST_CSV"

log "Suite=$SUITE  Depth=${DEPTH_KM}km  Variant=$VARIANT"
log "Dataset=$DATA_NAME"
log "Algos=${ALGOS[*]}"
log "Base out=$BASE_OUT_ROOT"
log "Sweep out=$SWEEP_ROOT/$DATA_NAME"

# -------------------------
# Run sweep
# -------------------------

for algo in "${ALGOS[@]}"; do
  algo="$(echo "$algo" | xargs)"  # trim
  model_args="$(algo_to_args "$algo")"

  if [[ "$algo" == rf ]]; then
    for trees in "${RF_TREES_LIST[@]}"; do
      for maxd in "${RF_MAX_DEPTH_LIST[@]}"; do
        tag="$(mk_tag_rf "$trees" "$maxd")"
        log "RUN $tag"

        # Convert "None" to actual CLI argument omission
        rf_depth_args=()
        if [[ "$maxd" != "None" ]]; then
          rf_depth_args=(--rf-max-depth "$maxd")
        fi

        python "$TRAIN_PY" \
          --data-root "$DATA_ROOT" \
          --data-name "$DATA_NAME" \
          $model_args \
          --rf-trees "$trees" \
          "${rf_depth_args[@]}" \
          --rf-jobs "$RF_JOBS" \
          --seed 42 \
          --out "$BASE_OUT_ROOT"

        # train_emulator writes to: <out>/<data-name>/<model_dir_name>/
        # model_dir_name for rf is "rf"
        src="$BASE_OUT_ROOT/$DATA_NAME/rf"
        if [[ ! -f "$src/model.joblib" ]]; then
          echo "[ERR] Expected model not found at: $src/model.joblib"
          exit 1
        fi
        copy_unique "$src" "$tag"
      done
    done

  else
    # GP kernel name from algo
    kernel="rbf"
    [[ "$algo" == gp_m15 ]] && kernel="matern15"
    [[ "$algo" == gp_m25 ]] && kernel="matern25"

    for restarts in "${GP_RESTARTS_LIST[@]}"; do
      for lsinit in "${LS_INIT_LIST[@]}"; do
        for lslow in "${LS_LOW_LIST[@]}"; do
          for lshigh in "${LS_HIGH_LIST[@]}"; do
            for ninit in "${NOISE_INIT_LIST[@]}"; do
              for nlow in "${NOISE_LOW_LIST[@]}"; do
                for nhigh in "${NOISE_HIGH_LIST[@]}"; do
                  for alpha in "${ALPHA_LIST[@]}"; do

                    tag="$(mk_tag_gp "$kernel" "$restarts" "$lsinit" "$lslow" "$lshigh" "$ninit" "$nlow" "$nhigh" "$alpha")"
                    log "RUN $tag"

                    python "$TRAIN_PY" \
                      --data-root "$DATA_ROOT" \
                      --data-name "$DATA_NAME" \
                      $model_args \
                      --gp-restarts "$restarts" \
                      --ls-init "$lsinit" \
                      --ls-bounds "$lslow" "$lshigh" \
                      --noise-init "$ninit" \
                      --noise-bounds "$nlow" "$nhigh" \
                      --alpha "$alpha" \
                      --seed 42 \
                      --out "$BASE_OUT_ROOT"

                    # train_emulator writes to: <out>/<data-name>/<model_dir_name>/
                    # model_dir_name for gp is "gp_rbf" OR "gp_m15" OR "gp_m25" per your code
                    src="$BASE_OUT_ROOT/$DATA_NAME/$algo"
                    if [[ ! -f "$src/model.joblib" ]]; then
                      echo "[ERR] Expected model not found at: $src/model.joblib"
                      exit 1
                    fi
                    copy_unique "$src" "$tag"

                  done
                done
              done
            done
          done
        done
      done
    done
  fi
done

log "DONE. Param sweep outputs in: $SWEEP_ROOT/$DATA_NAME"

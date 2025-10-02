#!/usr/bin/env bash
set -euo pipefail

# Usage check
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 TIMESTEP1 TIMESTEP2 [DEPTHS]"
  echo "Example: $0 1 6 25,50,75"
  exit 1
fi

TIMESTEP1=$1              # model times [Myrs]
TIMESTEP2=$2
DEPTHS=${3:-"25,50,75"}   # depths [km] 
IFS=',' read -r depth1 depth2 depth3 <<< "$DEPTHS"

# where this script and extract_cooling_rates_one-mod.py live
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRACT="${SCRIPT_DIR}/extract_cooling_rates_one-mod.py"  
ROOT="${SCRIPT_DIR}/../subd-model-runs/run-outputs"
OUTDT_DIR="$ROOT/analysis"
mkdir -p "$OUTDT_DIR"

# master CSV files (for ML training)
MASTER_DIR="$ROOT/analysis"
MASTER1="$MASTER_DIR/master_${depth1}km_DT${TIMESTEP1}-${TIMESTEP2}.csv"
MASTER2="$MASTER_DIR/master_${depth2}km_DT${TIMESTEP1}-${TIMESTEP2}.csv"
MASTER3="$MASTER_DIR/master_${depth3}km_DT${TIMESTEP1}-${TIMESTEP2}.csv"
# Initialize with headers (overwrite if exist)
echo "run_id,T1_C,T2_C,dT_C,dt_Myr,dTdt_C_per_Myr" > "$MASTER1"
echo "run_id,T1_C,T2_C,dT_C,dt_Myr,dTdt_C_per_Myr" > "$MASTER2"
echo "run_id,T1_C,T2_C,dT_C,dt_Myr,dTdt_C_per_Myr" > "$MASTER3"

SKIP_IF_DONE=true     # skip if OUTDT already exists

echo "Scanning: $ROOT"
shopt -s nullglob

for RUN_DIR in "$ROOT"/run_*; do

  echo "***********************************"
  MOD_NAME="$(basename "$RUN_DIR")"        # e.g., run_123
  # echo $MOD_NAME
  RUN_NUM="${MOD_NAME#run_}" 
  INNER="$RUN_DIR/outputs/$MOD_NAME/solution/solution-$(printf '%05d' "$TIMESTEP2").pvtu"

  if [[ -f "$INNER" ]]; then

    # DT output file name
    OUTDT="$OUTDT_DIR/${MOD_NAME}.DT_${TIMESTEP1}_${TIMESTEP2}.csv"
    
    # Run extractor only if ofile doesn't exist
    if [[ ! -f "$OUTDT" ]]; then
      echo "[RUN ] $MOD_NAME (t1=$TIMESTEP1, t2=$TIMESTEP2, depths=$DEPTHS)"
      "$EXTRACT" "$RUN_NUM" "$TIMESTEP1" "$TIMESTEP2" "$DEPTHS" || {
        echo "[FAIL] $MOD_NAME"; continue;
      }
    else
      echo "[SKIP] $MOD_NAME (found $OUTDT)"
    fi

    # Append to master files for specific depths
    for depth in $depth1 $depth2 $depth3; do
      line=$(awk -F, -v d=$depth 'NR>1 && ($1+0)==(d+0) {print $3,$5,$6,$7,$8; exit}' OFS="," "$OUTDT")
      if [[ -n "$line" ]]; then
        case $depth in
          "$depth1")  echo "$RUN_NUM,$line" >> "$MASTER1" ;;
          "$depth2")  echo "$RUN_NUM,$line" >> "$MASTER2" ;;
          "$depth3") echo "$RUN_NUM,$line" >>  "$MASTER3" ;;
        esac
      fi
    done
    
  else

    echo "[MISS] $MOD_NAME (no solution-$(printf '%05d' "$TIMESTEP2").pvtu)"
    for depth in $depth1 $depth2 $depth3; do
      case $depth in
        "$depth1")  echo "$RUN_NUM,NaN,NaN,NaN,NaN,NaN" >> "$MASTER1" ;;
        "$depth2")  echo "$RUN_NUM,NaN,NaN,NaN,NaN,NaN" >> "$MASTER2" ;;
        "$depth3")  echo "$RUN_NUM,NaN,NaN,NaN,NaN,NaN" >> "$MASTER3" ;;
      esac
    done

  fi

done

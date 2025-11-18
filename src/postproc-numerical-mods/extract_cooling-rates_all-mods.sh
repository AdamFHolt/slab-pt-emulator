#!/usr/bin/env bash

# Usage:
#   ./extract_cooling-rates_all-mods.sh TSTEP1 TSTEP2 [DEPTHS] [SUITE]
# Example:
#   ./extract_cooling-rates_all-mods.sh 1 10 5,10,15,...,80 const-vc

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 TSTEP1 TSTEP2 [DEPTHS] [SUITE]"
  echo "  TSTEP1, TSTEP2 = solution indices (e.g., 1 10)"
  echo "  DEPTHS        = comma-separated depths in km (default: 25,50,75)"
  echo "  SUITE         = const-vc or ramped-vc (default: const-vc)"
  exit 1
fi

TIMESTEP1="$1"
TIMESTEP2="$2"
DEPTHS="${3:-25,50,75}"
SUITE="${4:-const-vc}"

# >>> ADD THIS: parse DEPTHS into an array <<<
IFS=',' read -r -a DEPTH_ARR <<< "$DEPTHS"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRACT="${SCRIPT_DIR}/extract_cooling-rates_one-mod.py"

# Repo root = two levels above src/postproc-numerical-mods
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Suite root: <repo>/subd-model-runs/<suite>
SUITE_DIR="${ROOT}/subd-model-runs/${SUITE}"

RUN_ROOT="${SUITE_DIR}/run-outputs"
ANALYSIS_ROOT="${SUITE_DIR}/analysis"
mkdir -p "${ANALYSIS_ROOT}"

echo "Scanning: ${RUN_ROOT} (suite: ${SUITE})"

shopt -s nullglob
RUN_DIRS=( "${RUN_ROOT}"/run_* )

# Single hierarchical master CSV
MASTER="${ANALYSIS_ROOT}/master_DT${TIMESTEP1}-${TIMESTEP2}.csv"
echo "depth_km,run_id,T1_C,T2_C,dT_C,dt_Myr,dTdt_C_per_Myr" > "$MASTER"

i=0
for RUN_DIR in "${RUN_DIRS[@]}"; do
  ((i++))

  MOD_NAME="$(basename "$RUN_DIR")"   # e.g. run_000
  RUN_NUM="${MOD_NAME#run_}"         # extract 000

  printf "=====================================\n"
  printf "%s (t1=%s, t2=%s, depths=%s, suite=%s)\n" \
         "$MOD_NAME" "$TIMESTEP1" "$TIMESTEP2" "$DEPTHS" "$SUITE"

  INNER="${RUN_DIR}/solution/solution-$(printf '%05d' "${TIMESTEP2}").pvtu"

  RUN_ANALYSIS="${ANALYSIS_ROOT}/${MOD_NAME}"
  OUTDT="${RUN_ANALYSIS}/DT_${TIMESTEP1}_${TIMESTEP2}.csv"

  if [[ -f "$INNER" ]]; then
    mkdir -p "$RUN_ANALYSIS"
    if [[ ! -f "$OUTDT" ]]; then
      if ! "$EXTRACT" "$RUN_NUM" "$TIMESTEP1" "$TIMESTEP2" "$DEPTHS" "$SUITE"; then
        echo "[FAIL] ${MOD_NAME}"
        for depth in "${DEPTH_ARR[@]}"; do
          echo "${depth},${RUN_NUM},NaN,NaN,NaN,NaN,NaN" >> "$MASTER"
        done
        continue
      fi
    else
      echo "  [SKIP] ${MOD_NAME} (found ${OUTDT})"
    fi

    # Append per-depth results
    for depth in "${DEPTH_ARR[@]}"; do
      line=$(awk -F, -v d="$depth" '
        NR>1 && ($1+0)==(d+0) {
          if ($3=="" || $5=="" || $6=="" || $7=="" || $8=="")
            print "NaN,NaN,NaN,NaN,NaN";
          else
            print $3,$5,$6,$7,$8;
          found=1; exit
        }
        END { if (!found) print "NaN,NaN,NaN,NaN,NaN" }
      ' OFS="," "$OUTDT")

      echo "${depth},${RUN_NUM},${line}" >> "$MASTER"
    done

  else
    echo "  [MISS] ${MOD_NAME} (no ${INNER})"
    for depth in "${DEPTH_ARR[@]}"; do
      echo "${depth},${RUN_NUM},NaN,NaN,NaN,NaN,NaN" >> "$MASTER"
    done
  fi
done

echo "Done. Master CSV: ${MASTER}"

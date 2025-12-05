#!/usr/bin/env bash

# Usage:
#   ./extract_cooling-rates_all-mods.sh TSTEP1 TSTEP2 TRANGE [DEPTHS] [SUITE]
#
# Examples:
#   ./extract_cooling-rates_all-mods.sh 1 10 0:10 25,50,75 const-vc
#   ./extract_cooling-rates_all-mods.sh 1 10 0:10 5:80:2   const-vc
#
# where:
#   TSTEP1, TSTEP2 = solution indices used for ΔT/Δt (e.g., 1 10)
#   TRANGE         = "start:end" timestep range for AllT files (e.g., 0:10)
#   DEPTHS         = comma-separated depths in km (e.g., 25,50,75)
#                    or range "start:end:step" (e.g., 5:80:2)
#   SUITE          = const-vc or ramped-vc (default: const-vc)

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 TSTEP1 TSTEP2 TRANGE [DEPTHS] [SUITE]"
  echo "  TSTEP1, TSTEP2 = solution indices (e.g., 1 10)"
  echo "  TRANGE        = start:end timestep range for AllT (e.g., 0:10)"
  echo "  DEPTHS        = comma-separated depths in km (e.g. 25,50,75)"
  echo "                  or range 'start:end:step' (e.g. 5:80:2)"
  echo "  SUITE         = const-vc or ramped-vc (default: const-vc)"
  exit 1
fi

TIMESTEP1="$1"
TIMESTEP2="$2"
TRANGE_RAW="$3"
DEPTHS_RAW="${4:-25,50,75}"
SUITE="${5:-const-vc}"

# -------------------------------------------------------------------
# Parse DEPTHS:
#   - if form "start:end:step" use a bash range
#   - otherwise treat as comma-separated list
# -------------------------------------------------------------------
DEPTH_ARR=()

if [[ "$DEPTHS_RAW" == *:*:* ]]; then
  IFS=':' read -r dstart dend dstep <<< "$DEPTHS_RAW"
  if [[ -z "$dstart" || -z "$dend" || -z "$dstep" ]]; then
    echo "ERROR: bad range syntax for DEPTHS: '$DEPTHS_RAW'" >&2
    exit 1
  fi
  for ((d = dstart; d <= dend; d += dstep)); do
    DEPTH_ARR+=( "$d" )
  done
else
  IFS=',' read -r -a DEPTH_ARR <<< "$DEPTHS_RAW"
fi

# Canonical comma-separated depth string (for logs + python arg)
DEPTHS="$(IFS=','; echo "${DEPTH_ARR[*]}")"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRACT="${SCRIPT_DIR}/extract_cooling-rates_one-mod.py"

# Repo root = two levels above src/postproc-numerical-mods
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Suite root: <repo>/subd-model-runs/<suite>
SUITE_DIR="${ROOT}/subd-model-runs/${SUITE}"

RUN_ROOT="${SUITE_DIR}/run-outputs"
ANALYSIS_ROOT="${SUITE_DIR}/analysis"
mkdir -p "${ANALYSIS_ROOT}"

echo "Scanning: ${RUN_ROOT} (suite: ${SUITE}, TRANGE=${TRANGE_RAW}, depths=${DEPTHS})"

shopt -s nullglob
RUN_DIRS=( "${RUN_ROOT}"/run_* )

# Single hierarchical master CSV (unchanged format)
MASTER="${ANALYSIS_ROOT}/master_DT${TIMESTEP1}-${TIMESTEP2}.csv"
echo "depth_km,run_id,T1_C,T2_C,dT_C,dt_Myr,dTdt_C_per_Myr" > "$MASTER"

i=0
for RUN_DIR in "${RUN_DIRS[@]}"; do
  ((i++))

  MOD_NAME="$(basename "$RUN_DIR")"   # e.g. run_000
  RUN_NUM="${MOD_NAME#run_}"         # extract 000

  printf "=====================================\n"
  printf "%s (t1=%s, t2=%s, TRANGE=%s, depths=%s, suite=%s)\n" \
         "$MOD_NAME" "$TIMESTEP1" "$TIMESTEP2" "$TRANGE_RAW" "$DEPTHS" "$SUITE"

  INNER="${RUN_DIR}/solution/solution-$(printf '%05d' "${TIMESTEP2}").pvtu"

  RUN_ANALYSIS="${ANALYSIS_ROOT}/${MOD_NAME}"
  OUTDT="${RUN_ANALYSIS}/DT_${TIMESTEP1}_${TIMESTEP2}.csv"

  if [[ -f "$INNER" ]]; then
    mkdir -p "$RUN_ANALYSIS"
    if [[ ! -f "$OUTDT" ]]; then
      if ! "$EXTRACT" "$RUN_NUM" "$TIMESTEP1" "$TIMESTEP2" "$TRANGE_RAW" "$DEPTHS" "$SUITE"; then
        echo "[FAIL] ${MOD_NAME}"
        for depth in "${DEPTH_ARR[@]}"; do
          echo "${depth},${RUN_NUM},NaN,NaN,NaN,NaN,NaN" >> "$MASTER"
        done
        continue
      fi
    else
      echo "  [SKIP] ${MOD_NAME} (found ${OUTDT})"
    fi

    # Append per-depth results into hierarchical master
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

echo "Done. Master DT CSV: ${MASTER}"

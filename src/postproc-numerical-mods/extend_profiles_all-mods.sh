#!/usr/bin/env bash
set -uo pipefail

# Extend the processed slab-top record of a suite to later output steps.
#
# For every run that has *complete* solution output through the last requested
# timestep, this script
#   1) writes the missing field CSVs  analysis/run_XXX/t{k}.csv   (pvpython)
#   2) writes the missing profiles    analysis/run_XXX/Tprof_{k}.csv
#      and the requested DT pair files analysis/run_XXX/DT_{a}_{b}.csv
#      (venv python, src/postproc-numerical-mods/extract_profiles_range.py)
#
# Existing outputs are left alone unless OVERWRITE=1.
#
# Usage:
#   ./extend_profiles_all-mods.sh [SUITE] [TPROF_STEPS] [DT_PAIRS] [NPROC]
#
# Examples:
#   ./extend_profiles_all-mods.sh const-vc 11:20 "1,20;10,20" 24
#
# Environment overrides:
#   PVPYTHON   path to pvpython
#   DEPTHS     depth spec passed through (default 0:80:1)
#   OVERWRITE  1 to rewrite existing Tprof/DT files

SUITE="${1:-const-vc}"
TPROF_STEPS="${2:-11:20}"
DT_PAIRS="${3:-1,20;10,20}"
NPROC="${4:-24}"

DEPTHS="${DEPTHS:-0:80:1}"
OVERWRITE="${OVERWRITE:-0}"
PVPYTHON="${PVPYTHON:-/home/holt/software/ParaView-5.11.1-MPI-Linux-Python3.9-x86_64/bin/pvpython}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY="${ROOT}/env/bin/python"
[[ -x "$PY" ]] || PY="python3"

SUITE_DIR="${ROOT}/subd-model-runs/${SUITE}"
RUN_ROOT="${SUITE_DIR}/run-outputs"
ANALYSIS_ROOT="${SUITE_DIR}/analysis"
LOG_DIR="${ANALYSIS_ROOT}/logs-extend"
mkdir -p "${ANALYSIS_ROOT}" "${LOG_DIR}"

# Expand the requested Tprof step list (a:b[:s] or comma list).
if [[ "${TPROF_STEPS}" == *:* ]]; then
  IFS=':' read -r s0 s1 ss <<< "${TPROF_STEPS}"
  ss="${ss:-1}"
  STEPS=()
  for ((k = s0; k <= s1; k += ss)); do STEPS+=("$k"); done
else
  IFS=',' read -r -a STEPS <<< "${TPROF_STEPS}"
fi

# All timesteps whose field CSV we need = Tprof steps + every DT pair endpoint.
NEEDED=( "${STEPS[@]}" )
IFS=';' read -r -a PAIRS <<< "${DT_PAIRS}"
for p in "${PAIRS[@]}"; do
  [[ -z "$p" ]] && continue
  IFS=',' read -r a b <<< "$p"
  NEEDED+=( "$a" "$b" )
done
NEEDED_SORTED=$(printf '%s\n' "${NEEDED[@]}" | sort -n -u | tr '\n' ' ')
MAX_STEP=$(printf '%s\n' "${NEEDED[@]}" | sort -n | tail -1)

echo "suite        : ${SUITE}"
echo "Tprof steps  : ${STEPS[*]}"
echo "DT pairs     : ${DT_PAIRS}"
echo "field CSVs   : ${NEEDED_SORTED}"
echo "max step     : ${MAX_STEP}"
echo "depths       : ${DEPTHS}"
echo "parallelism  : ${NPROC}"
echo

# ---- Build the work list: runs with contiguous solution output to MAX_STEP --
WORKLIST="${LOG_DIR}/worklist.txt"
SKIPLIST="${LOG_DIR}/skipped.txt"
: > "${WORKLIST}"
: > "${SKIPLIST}"

shopt -s nullglob
for RUN_DIR in "${RUN_ROOT}"/run_*; do
  MOD="$(basename "${RUN_DIR}")"
  RUN_NUM="${MOD#run_}"
  ok=1
  for k in ${NEEDED_SORTED}; do
    if [[ ! -f "${RUN_DIR}/solution/solution-$(printf '%05d' "$k").pvtu" ]]; then
      ok=0
      break
    fi
  done
  if [[ $ok -eq 1 ]]; then
    echo "${RUN_NUM}" >> "${WORKLIST}"
  else
    echo "${RUN_NUM}" >> "${SKIPLIST}"
  fi
done

echo "runs to process: $(wc -l < "${WORKLIST}")"
echo "runs skipped   : $(wc -l < "${SKIPLIST}")  (see ${SKIPLIST})"
echo

# ---- Per-run worker --------------------------------------------------------
cat > "${LOG_DIR}/_worker.sh" <<WORKER
#!/usr/bin/env bash
set -uo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
RUN_NUM="\$1"
LOG="${LOG_DIR}/run_\${RUN_NUM}.log"
{
  echo "=== run_\${RUN_NUM} start \$(date -Is)"
  "${PVPYTHON}" "${SCRIPT_DIR}/extract_field_csvs.py" "\${RUN_NUM}" "${NEEDED_SORTED// /,}" "${SUITE}" || exit 11
  EXTRA=""
  if [[ "${OVERWRITE}" == "1" ]]; then EXTRA="--overwrite"; fi
  "${PY}" "${SCRIPT_DIR}/extract_profiles_range.py" \
      --suite "${SUITE}" --run "\${RUN_NUM}" \
      --tprof-steps "${TPROF_STEPS}" \
      --dt-pairs "${DT_PAIRS}" \
      --depths "${DEPTHS}" \
      --grid-depth-max-km 90 \
      \${EXTRA} || exit 12
  echo "=== run_\${RUN_NUM} done \$(date -Is)"
} > "\${LOG}" 2>&1
rc=\$?
if [[ \$rc -ne 0 ]]; then echo "[FAIL rc=\$rc] run_\${RUN_NUM}"; else echo "[ok] run_\${RUN_NUM}"; fi
exit 0
WORKER
chmod +x "${LOG_DIR}/_worker.sh"

xargs -a "${WORKLIST}" -n 1 -P "${NPROC}" "${LOG_DIR}/_worker.sh"

echo
echo "Done. Logs in ${LOG_DIR}"

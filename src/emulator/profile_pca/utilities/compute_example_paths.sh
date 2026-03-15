#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./compute_example_paths.sh
# Optional env overrides:
#   SUITE=const-vc START_TIME=0.5 V_CONV=5 AGE_OP=80 DIP_INT=30 ETA_UM=1e21
#   MANY_BURIAL_RATES="2:6:4" MANY_MAX_DEPTHS="20,22.5,25,27.5,30,32.5,35,37.5,40,42.5,45" MANY_EXHUM_TIMES="0:1:4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

SUITE="${SUITE:-const-vc}"
START_TIME="${START_TIME:-0.5}"
V_CONV="${V_CONV:-5}"
AGE_OP="${AGE_OP:-80}"
DIP_INT="${DIP_INT:-30}"
ETA_UM="${ETA_UM:-1e21}"
MANY_BURIAL_RATES="${MANY_BURIAL_RATES:-2:6:4}"
MANY_MAX_DEPTHS="${MANY_MAX_DEPTHS:-20,22.5,25,27.5,30,32.5,35,37.5,40,42.5,45}"
MANY_EXHUM_TIMES="${MANY_EXHUM_TIMES:-0:1:4}"

echo "== Example profile-PCA burial paths =="
echo "suite      : ${SUITE}"
echo "start time : ${START_TIME} Myr"
echo "v_conv     : ${V_CONV}"
echo "age_SP     : median(training)"
echo "age_OP     : ${AGE_OP}"
echo "dip_int    : ${DIP_INT}"
echo "eta_UM     : ${ETA_UM}"
echo "many vb    : ${MANY_BURIAL_RATES}"
echo "many zmax  : ${MANY_MAX_DEPTHS}"
echo "many thold : ${MANY_EXHUM_TIMES}"
echo

echo "[1/2] single-path example"
python3 "${SCRIPT_DIR}/compute_burial_path.py" \
  --suite "${SUITE}" \
  --start-time-myr "${START_TIME}" \
  --burial-rate-cm-per-yr 3.0 \
  --exhumation-rate-cm-per-yr 1.5 \
  --exhumation-time-myr 0.75 \
  --transition-time-myr 0.1 \
  --max-depth-km 35 \
  --v-conv "${V_CONV}" \
  --age-op "${AGE_OP}" \
  --dip-int "${DIP_INT}" \
  --eta-um "${ETA_UM}"

echo
echo "[2/3] uncertain age_SP example"
python3 "${SCRIPT_DIR}/compute_burial_path_uncertain_parameter.py" \
  --suite "${SUITE}" \
  --start-time-myr "${START_TIME}" \
  --burial-rate-cm-per-yr 3.0 \
  --exhumation-rate-cm-per-yr 1.5 \
  --exhumation-time-myr 0.75 \
  --transition-time-myr 0.1 \
  --max-depth-km 35 \
  --uncertain-feature age-sp \
  --v-conv "${V_CONV}" \
  --age-op "${AGE_OP}" \
  --dip-int "${DIP_INT}" \
  --eta-um "${ETA_UM}"

echo
echo "[3/3] many-path example"
python3 "${SCRIPT_DIR}/compute_many_burial_paths.py" \
  --suite "${SUITE}" \
  --start-time-myr "${START_TIME}" \
  --burial-rates-cm-per-yr "${MANY_BURIAL_RATES}" \
  --max-depths-km "${MANY_MAX_DEPTHS}" \
  --exhumation-times-myr "${MANY_EXHUM_TIMES}" \
  --exhumation-rate-cm-per-yr 1.5 \
  --transition-time-myr 0.05 \
  --v-conv "${V_CONV}" \
  --age-op "${AGE_OP}" \
  --dip-int "${DIP_INT}" \
  --eta-um "${ETA_UM}"

echo
echo "== Done =="

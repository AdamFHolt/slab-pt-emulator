#!/usr/bin/env bash
# check_runs_list.sh -- check only the runs named in a list file (same format as submit_from_list.sh:
# one run number per line, zero-padded or not, '#' comments and blank lines allowed).
#
# Lives next to the suites in $WORK/aspect_work/SlabT_emulator/production-runs_v2/ and looks for the
# model outputs in $SCRATCH/aspect_work/outputs/run_### (where run_one.slurm runs ASPECT).
#
# Usage (from production-runs_v2/):
#   ./check_runs_list.sh LIST_FILE [TARGET_STEP=00020] [-p]
#   ./check_runs_list.sh const-vc-dd100/pilot-list.txt
#   ./check_runs_list.sh const-vc-dd100/pilot-list.txt 00004 -p
#
#   TARGET_STEP  zero-padded solution step that marks "done" (00020 = 10 Myr at 0.5 Myr output)
#   -p           problems only (suppress OK rows), like the original check_runs.sh
#   BASE_DIR     env override for the outputs root (default $SCRATCH/aspect_work)
#
# Status per run:
#   OK               solution-<TARGET_STEP>.pvtu present
#   NO_TARGET        running / stalled: shows the latest step found
#   NO_SOLUTION_DIR  outputs/run_###/ exists but no solution/ yet
#   MISSING_DIR      outputs/run_### does not exist (not started, or wrong BASE_DIR)
# Exit code 1 if any listed run is not OK.
set -euo pipefail

LIST="${1:?usage: check_runs_list.sh LIST_FILE [TARGET_STEP] [-p]}"; shift
BASE="${BASE_DIR:-${SCRATCH:-/scratch/04714/adamholt}/aspect_work}"
TARGET_STEP="00020"; PROBLEMS_ONLY=0
for a in "$@"; do
  case "$a" in
    -p) PROBLEMS_ONLY=1 ;;
    *) TARGET_STEP="$a" ;;
  esac
done
[[ -f "$LIST" ]] || { echo "No list file: $LIST"; exit 1; }
[[ -d "$BASE/outputs" ]] || { echo "No outputs dir at $BASE/outputs (set BASE_DIR)"; exit 1; }
echo "outputs root: $BASE   target step: $TARGET_STEP"

printf "%-10s %-16s %-14s %s\n" "RUN" "STATUS" "MAX_PVTU_STEP" "SOLUTION_DIR"
shopt -s nullglob
had_issue=0; n_ok=0; n_all=0

while IFS= read -r raw || [[ -n "${raw:-}" ]]; do
  line="$(sed -E 's/^[[:space:]]+|[[:space:]]+$//g' <<<"${raw:-}")"
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  [[ "$line" =~ ^[0-9]+$ ]] || { echo "SKIP: not a number: '$line'"; continue; }
  run="$(printf 'run_%03d' "$((10#$line))")"
  n_all=$((n_all + 1))
  run_dir="$BASE/outputs/$run"
  sol="$run_dir/solution"

  if [[ ! -d "$run_dir" ]]; then
    had_issue=1; printf "%-10s %-16s %-14s %s\n" "$run" "MISSING_DIR" "-" "-"; continue
  fi
  if [[ ! -d "$sol" ]]; then
    had_issue=1; printf "%-10s %-16s %-14s %s\n" "$run" "NO_SOLUTION_DIR" "-" "$sol"; continue
  fi

  has="no"; max="none"
  for f in "$sol"/solution-*.pvtu; do
    step="${f##*/solution-}"; step="${step%.pvtu}"
    [[ "$step" == "$TARGET_STEP" ]] && has="yes"
    [[ "$max" == "none" || $((10#$step)) -gt $((10#$max)) ]] && max="$step"
  done

  if [[ "$has" == "yes" ]]; then
    n_ok=$((n_ok + 1))
    (( PROBLEMS_ONLY )) || printf "%-10s %-16s %-14s %s\n" "$run" "OK" "$max" "$sol"
  else
    had_issue=1; printf "%-10s %-16s %-14s %s\n" "$run" "NO_TARGET" "$max" "$sol"
  fi
done < "$LIST"

echo "-- $n_ok / $n_all listed runs have step $TARGET_STEP"
(( had_issue )) && exit 1 || exit 0

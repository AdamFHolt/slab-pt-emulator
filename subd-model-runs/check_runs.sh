#!/usr/bin/env bash
# check_runs.sh -- check a range of runs (run_START .. run_END).  Companion of check_runs_list.sh.
# Lives next to the suites in $WORK/aspect_work/SlabT_emulator/production-runs_v2/ and looks for the
# model outputs in $SCRATCH/aspect_work/outputs/run_### (where run_one.slurm runs ASPECT).
#
# Usage (from production-runs_v2/):
#   ./check_runs.sh [TARGET_STEP=00020] [START=0] [END=399] [-a]
#   ./check_runs.sh                 # problem runs only, whole suite, target step 00020
#   ./check_runs.sh 00010 0 199     # first half of the suite, has it reached 5 Myr?
#   ./check_runs.sh 00020 0 399 -a  # every run with its latest step
#
#   TARGET_STEP  zero-padded solution step that marks "done" (00020 = 10 Myr at 0.5 Myr output)
#   -a           show all runs (default prints problem rows only, like the original check_runs.sh)
#   BASE_DIR     env override for the outputs root (default $SCRATCH/aspect_work)
#
# Status per run:
#   OK               solution-<TARGET_STEP>.pvtu present
#   NO_TARGET        running / stalled: shows the latest step found
#   NO_SOLUTION_DIR  outputs/run_###/ exists but no solution/ yet
#   MISSING_DIR      outputs/run_### does not exist (not started, or wrong BASE_DIR)
# Exit code 1 if any run in the range is not OK.
set -euo pipefail

BASE="${BASE_DIR:-${SCRATCH:-/scratch/04714/adamholt}/aspect_work}"
TARGET_STEP="00020"; START=0; END=399; SHOW_ALL=0
pos=()
for a in "$@"; do
  case "$a" in
    -a) SHOW_ALL=1 ;;
    *) pos+=("$a") ;;
  esac
done
[[ ${#pos[@]} -ge 1 ]] && TARGET_STEP="${pos[0]}"
[[ ${#pos[@]} -ge 2 ]] && START="${pos[1]}"
[[ ${#pos[@]} -ge 3 ]] && END="${pos[2]}"
[[ -d "$BASE/outputs" ]] || { echo "No outputs dir at $BASE/outputs (set BASE_DIR)"; exit 1; }
echo "outputs root: $BASE   target step: $TARGET_STEP   runs: $START-$END"

printf "%-10s %-16s %-14s %s\n" "RUN" "STATUS" "MAX_PVTU_STEP" "SOLUTION_DIR"
shopt -s nullglob
had_issue=0; n_ok=0; n_all=0

for i in $(seq "$((10#$START))" "$((10#$END))"); do
  run="$(printf 'run_%03d' "$i")"
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
    (( SHOW_ALL )) && printf "%-10s %-16s %-14s %s\n" "$run" "OK" "$max" "$sol"
  else
    had_issue=1; printf "%-10s %-16s %-14s %s\n" "$run" "NO_TARGET" "$max" "$sol"
  fi
done

echo "-- $n_ok / $n_all runs have step $TARGET_STEP"
(( had_issue )) && exit 1 || exit 0

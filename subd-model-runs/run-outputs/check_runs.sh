#!/usr/bin/env bash
# check_runs_ordered.sh
# Usage:
#   ./check_runs_ordered.sh [BASE_DIR=. ] [TARGET_STEP=00005] [START=0] [END=399]
#
# Prints a single, ordered list of only problem runs:
# - MISSING_DIR        : run_### directory doesn’t exist
# - NO_SOLUTION_DIR    : outputs/.../solution is missing
# - NO_TARGET          : solution-<TARGET_STEP>.pvtu not found (shows MAX_PVTU_STEP)

set -euo pipefail

BASE="${1:-.}"          # directory that should contain run_###
TARGET_STEP="${2:-00006}"
START="${3:-0}"         # inclusive
END="${4:-399}"         # inclusive

printf "%-10s %-16s %-14s %s\n" "RUN" "STATUS" "MAX_PVTU_STEP" "SOLUTION_DIR"

shopt -s nullglob
had_issue=0

for i in $(seq -w "$START" "$END"); do
  run="run_$i"
  run_dir="$BASE/$run"

  # 1) Missing run directory
  if [[ ! -d "$run_dir" ]]; then
    had_issue=1
    printf "%-10s %-16s %-14s %s\n" "$run" "MISSING_DIR" "-" "-"
    continue
  fi

  # 2) Missing solution directory
  sol="$run_dir/outputs/$run/solution"
  if [[ ! -d "$sol" ]]; then
    had_issue=1
    printf "%-10s %-16s %-14s %s\n" "$run" "NO_SOLUTION_DIR" "-" "$sol"
    continue
  fi

  # 3) Check for target step
  has="no"
  steps=()
  for f in "$sol"/solution-*.pvtu; do
    fname="${f##*/}"             # solution-00009.pvtu
    step="${fname#solution-}"    # 00009.pvtu
    step="${step%.pvtu}"         # 00009
    steps+=("$step")
    [[ "$step" == "$TARGET_STEP" ]] && has="yes"
  done

  if [[ "$has" == "no" ]]; then
    had_issue=1
    max="none"
    if ((${#steps[@]})); then
      # steps are zero-padded; -n numeric sort preserves order
      max="$(printf "%s\n" "${steps[@]}" | sort -n | tail -1)"
    fi
    printf "%-10s %-16s %-14s %s\n" "$run" "NO_TARGET" "$max" "$sol"
  fi
done

# Exit non-zero if any problems found
(( had_issue )) && exit 1 || exit 0


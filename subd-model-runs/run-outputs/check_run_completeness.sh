#!/usr/bin/env bash
set -euo pipefail

TARGET_STEP="${1:-00010}"           # step to check for (zero-padded)

printf "%-10s %-8s %-14s %s\n" "RUN" "HAS_TGT" "MAX_PVTU_STEP" "SOLUTION_DIR"

shopt -s nullglob
for run_dir in ./run_*; do
  run=$(basename "$run_dir")
  sol="$run_dir/outputs/$run/solution"
  [[ -d "$sol" ]] || continue

  has="no"
  steps=()

  for f in "$sol"/solution-*.pvtu; do
    fname=${f##*/}              # solution-00009.pvtu
    step=${fname#solution-}     # 00009.pvtu
    step=${step%.pvtu}          # 00009
    steps+=("$step")
    [[ "$step" == "$TARGET_STEP" ]] && has="yes"
  done

  # Only report “didn’t finish”
  if [[ "$has" == "no" ]]; then
    max="none"
    if ((${#steps[@]})); then
      max=$(printf "%s\n" "${steps[@]}" | sort -n | tail -1)
    fi
    printf "%-10s %-8s %-14s %s\n" "$run" "$has" "$max" "$sol"
  fi
done

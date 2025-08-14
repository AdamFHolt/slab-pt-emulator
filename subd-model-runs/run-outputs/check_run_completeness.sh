#!/usr/bin/env bash

set -euo pipefail
shopt -s nullglob

ROOT="${1:-$PWD}"   # directory containing run_* folders
INDEX="00010"       # change if you want a different step

missing=()
total=0

for d in "$ROOT"/run_*; do
  [[ -d "$d" ]] || continue
  run_name="$(basename "$d")"
  target="$d/outputs/$run_name/solution/solution-$INDEX.pvtu"
  if [[ ! -f "$target" ]]; then
    echo "MISSING: $run_name  (expected: $target)"
    missing+=("$run_name")
  fi
done


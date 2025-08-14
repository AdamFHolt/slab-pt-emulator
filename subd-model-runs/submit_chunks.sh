#!/usr/bin/env bash
set -euo pipefail

SLURM_FILE=slabpt-suite.slurm
CHUNK=40            # ≤ per-user submit cap
THROTTLE=10         # concurrent tasks within each array

start=0
while [ $start -lt 400 ]; do
  end=$(( start + CHUNK - 1 ))
  [ $end -gt 399 ] && end=399

  echo "Submitting ${start}-${end}%${THROTTLE} and waiting to finish..."
  sbatch --array=${start}-${end}%${THROTTLE} --wait "$SLURM_FILE"

  start=$(( end + 1 ))
done

echo "All chunks done."

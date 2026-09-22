#!/usr/bin/env bash
# push_runs_to_tacc.sh -- rsync a suite's run-inputs to Stampede3.
# Lives next to the suites; counterpart of pull_runs_from_tacc.sh.
#
#   subd-model-runs/push_runs_to_tacc.sh SUITE [LINK_SUITE] [-n]
#
#   SUITE       local suite name under subd-model-runs/ (also the TACC directory name)
#   LINK_SUITE  optional TACC suite whose run_XXX/inputs are identical; rsync hard-links
#               matching files there (--link-dest) instead of transferring them
#   -n          dry run
#
# Examples:
#   push_runs_to_tacc.sh const-vc-dd100 const-vc-new      # dd100 inputs == const-vc inputs: ~MBs, not 19 GB
#   push_runs_to_tacc.sh const-vc-dd100 const-vc-new -n   # show what would be sent
#   push_runs_to_tacc.sh const-vc-sh                      # plain full transfer
#
# TACC layout (matches run_one.slurm BASE_DIR): $WORK/aspect_work/SlabT_emulator/production-runs_v2/<SUITE>/run_XXX/
# Prompts for TACC password + token; run from your own terminal.  Afterwards, on Stampede3:
#   cd $WORK/aspect_work/SlabT_emulator/production-runs_v2/<SUITE> && ./submit_from_list.sh pilot-list.txt
set -euo pipefail

SUITE="${1:?usage: push_runs_to_tacc.sh SUITE [LINK_SUITE] [-n]}"; shift
LINK_SUITE=""; DRY=()
for a in "$@"; do
  case "$a" in -n|--dry-run) DRY=(-n) ;; *) LINK_SUITE="$a" ;; esac
done

TACC_USER="${TACC_USER:-adamholt}"
TACC_HOST="${TACC_HOST:-stampede3.tacc.utexas.edu}"
TACC_WORK="${TACC_WORK:-/work2/04714/adamholt/stampede3}"
TACC_BASE="$TACC_WORK/aspect_work/SlabT_emulator/production-runs_v2"

RUNS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$RUNS_DIR/$SUITE/run-inputs/"
[[ -d "$SRC" ]] || { echo "no run-inputs for suite '$SUITE' at $SRC"; exit 1; }

LINK=()
[[ -n "$LINK_SUITE" ]] && LINK=(--link-dest="../$LINK_SUITE/")   # relative to the destination dir

set -x
rsync -avh --info=progress2 "${DRY[@]}" "${LINK[@]}" \
  "$SRC" "$TACC_USER@$TACC_HOST:$TACC_BASE/$SUITE/"

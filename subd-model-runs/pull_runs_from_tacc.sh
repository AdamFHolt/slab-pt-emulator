#!/usr/bin/env bash
# pull_runs_from_tacc.sh -- rsync a suite's model outputs back from Stampede3 scratch.
# Lives next to the suites; counterpart of push_runs_to_tacc.sh, next to it.
#
#   subd-model-runs/pull_runs_from_tacc.sh SUITE [LIST_FILE ...] [-a] [-n]
#
#   SUITE      local suite name under subd-model-runs/
#   LIST_FILE  submission list(s) naming the runs to pull (one run number per line, '#' comments
#              allowed -- the same format submit_from_list.sh takes).  Defaults to every
#              run-inputs/*list*.txt in the suite.
#   -a         pull every run_* on scratch, ignoring the lists
#   -n         dry run
#
# -a vs the lists: $SCRATCH/aspect_work/outputs/ can be shared by every suite run there, and
# run_XXX numbers collide across suites, so by default only the runs named in LIST_FILE are
# transferred.  Use -a when you know the outputs dir holds this suite alone -- it also picks up
# runs that are not in any list (one-off reruns, hand-submitted jobs).
#
# Pulls, per run: solution/, solution.pvd, log.txt, statistics, parameters.prm.  Checkpoints
# (restart.*) are left on TACC.  Incremental -- rerun to top up runs that have advanced since.
# ~200 MB per finished run.
#
# Examples:
#   ./pull_runs_from_tacc.sh const-vc-dd100 -a -n                        # preview; -a = whole scratch
#   ./pull_runs_from_tacc.sh const-vc-dd100 -a                            # every run_* on scratch
#   ./pull_runs_from_tacc.sh const-vc-dd100                               # only runs named in the lists
#   ./pull_runs_from_tacc.sh const-vc-dd100 const-vc-dd100/run-inputs/pilot-list.txt
#
# Prompts for TACC password + token; run from your own terminal.
set -euo pipefail

SUITE="${1:?usage: pull_runs_from_tacc.sh SUITE [LIST_FILE ...] [-a] [-n]}"; shift
DRY=(); LISTS=(); ALL=0
for a in "$@"; do
  case "$a" in
    -n|--dry-run) DRY=(-n) ;;
    -a|--all)     ALL=1 ;;
    *)            LISTS+=("$a") ;;
  esac
done

TACC_USER="${TACC_USER:-adamholt}"
TACC_HOST="${TACC_HOST:-stampede3.tacc.utexas.edu}"
TACC_OUT="${TACC_OUT:-/scratch/04714/adamholt/aspect_work/outputs}"

RUNS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_DIR="$RUNS_DIR/$SUITE"
DEST="$SUITE_DIR/run-outputs"
[[ -d "$SUITE_DIR" ]] || { echo "no suite '$SUITE' at $SUITE_DIR"; exit 1; }

if [[ $ALL -eq 1 ]]; then
  LISTS=()
elif [[ ${#LISTS[@]} -eq 0 ]]; then
  shopt -s nullglob
  LISTS=("$SUITE_DIR"/run-inputs/*list*.txt)
  [[ ${#LISTS[@]} -gt 0 ]] || { echo "no run-inputs/*list*.txt in $SUITE_DIR; name a LIST_FILE"; exit 1; }
fi

# union of the lists, zero-padded, deduplicated
if [[ $ALL -eq 1 ]]; then
  RUNS=""; N="all"
  echo "suite $SUITE: every run_* on scratch  ->  $DEST"
else
  RUNS="$(for f in "${LISTS[@]}"; do
            [[ -f "$f" ]] || { echo "no such list: $f" >&2; exit 1; }
            awk '!/^[[:space:]]*(#|$)/ {printf "%03d\n", $1+0}' "$f"
          done | sort -u)"
  N=$(wc -l <<<"$RUNS")
  echo "suite $SUITE: $N runs from ${LISTS[*]##*/}  ->  $DEST"
fi

FILTER="$(mktemp)"; trap 'rm -f "$FILTER"' EXIT
{ if [[ $ALL -eq 1 ]]; then echo "+ /run_*/"
  else while read -r r; do echo "+ /run_$r/"; done <<<"$RUNS"; fi
  echo "- /*"                 # anything else at the top of the shared scratch outputs/
  echo "+ solution/"; echo "+ solution/**"
  echo "+ solution.pvd"; echo "+ log.txt"; echo "+ statistics"; echo "+ parameters.prm"
  echo "- *"; } > "$FILTER"   # restart.* and anything else stays on TACC

mkdir -p "$DEST"
set -x
rsync -avh --info=progress2 --partial-dir=.rsync-partial "${DRY[@]}" \
  --filter="merge $FILTER" \
  "$TACC_USER@$TACC_HOST:$TACC_OUT/" "$DEST/"
set +x

[[ ${#DRY[@]} -eq 0 ]] || exit 0
[[ $ALL -eq 1 ]] && RUNS="$(cd "$DEST" && ls -d run_*/ 2>/dev/null | sed 's#run_##;s#/##')" && N=$(wc -l <<<"$RUNS")
done_n=0
while read -r r; do
  [[ -n "$r" ]] || continue
  [[ -f "$DEST/run_$r/solution/solution-00021.pvtu" ]] && done_n=$((done_n + 1))
done <<<"$RUNS"
echo "-- local: $done_n / $N runs have step 00021 (10.5 Myr); $(du -sh "$DEST" | cut -f1) on disk"

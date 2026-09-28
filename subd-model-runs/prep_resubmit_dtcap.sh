#!/usr/bin/env bash
# prep_resubmit_dtcap.sh -- prepare a list of failed ASPECT 2.5 runs for a checkpoint resume with the
# time-step growth cap that ASPECT 3.0 applies by default.  RUNS ON STAMPEDE3, from production-runs_v2/.
#
#   ./prep_resubmit_dtcap.sh SUITE_DIR OUT_SUBDIR LIST_FILE [-n] [--fresh]
#
#   SUITE_DIR   suite directory under production-runs_v2/ holding run_XXX/run_XXX.prm
#               (const-vc-new, ramped-vc-new)
#   OUT_SUBDIR  where the run's outputs were moved to under $SCRATCH/aspect_work/outputs/
#               (const-v2, ramped-v2); the prm's Output directory becomes outputs/OUT_SUBDIR/run_XXX
#   LIST_FILE   run numbers, one per line, '#' whole-line comments (submit_from_list.sh format)
#   -n          dry run: report what would change, touch nothing
#   --fresh     rerun from t=0 instead of resuming: a run whose outputs/OUT_SUBDIR/run_XXX has no usable
#               checkpoint (restart.mesh_fixed.data) gets that directory moved aside to run_XXX.dead-<date>
#               so ASPECT starts clean (Resume computation = auto finds nothing).  Runs WITH a checkpoint
#               are left to resume.  Without --fresh a missing checkpoint is a warning and nothing moves.
#
# Per run, on run_XXX/run_XXX.prm:
#   1. keeps the untouched prm as run_XXX.prm.orig (only if no .orig exists yet)
#   2. set Output directory = outputs/OUT_SUBDIR/run_XXX
#   3. adds `set Maximum relative increase in time step = 91` after the CFL line (percent: dt may grow
#      at most 1.91x per step).  ASPECT 2.5 leaves this unlimited; the 2.5 suites' deterministic
#      ocrust-advection deaths follow an 8-13x dt jump after a refinement-triggered tiny step, which
#      the 3.0 default (91) prevented in const-vc-sh (400/400).  Idempotent.
#   4. checks that the checkpoint the resume will use exists (outputs/OUT_SUBDIR/run_XXX/restart.mesh)
#      and warns if outputs/run_XXX also exists (another suite's run of the same number).
# Afterwards: ./submit_from_list.sh LIST_FILE from SUITE_DIR (Resume computation = auto picks up the
# checkpoint).  Nothing else in the run dir or the batch scripts changes.
#
# Why: SESSION_NOTES 2026-09-28 (evening), subd-model-runs/const-vc/run-inputs/resubmit-list.2026-09-28.txt
set -euo pipefail

SUITE_DIR="${1:?usage: prep_resubmit_dtcap.sh SUITE_DIR OUT_SUBDIR LIST_FILE [-n]}"
OUT_SUBDIR="${2:?usage: prep_resubmit_dtcap.sh SUITE_DIR OUT_SUBDIR LIST_FILE [-n]}"
LIST_FILE="${3:?usage: prep_resubmit_dtcap.sh SUITE_DIR OUT_SUBDIR LIST_FILE [-n]}"
DRY=0; FRESH=0
for a in "${@:4}"; do
  case "$a" in
    -n) DRY=1 ;;
    --fresh) FRESH=1 ;;
    *) echo "unknown option: $a"; exit 1 ;;
  esac
done
STAMP="$(date +%Y-%m-%d)"

CAP="${CAP:-91}"
BASE="${BASE_DIR:-${WORK:?WORK not set}/aspect_work/SlabT_emulator/production-runs_v2}"
OUTROOT="${OUTROOT:-${SCRATCH:?SCRATCH not set}/aspect_work/outputs}"

[[ -d "$BASE/$SUITE_DIR" ]] || { echo "no suite dir: $BASE/$SUITE_DIR"; exit 1; }
[[ -f "$LIST_FILE" ]]      || { echo "no list file: $LIST_FILE"; exit 1; }
[[ -d "$OUTROOT/$OUT_SUBDIR" ]] || { echo "no outputs dir: $OUTROOT/$OUT_SUBDIR"; exit 1; }
echo "suite $BASE/$SUITE_DIR  ->  outputs/$OUT_SUBDIR/run_XXX,  cap $CAP %,  list $LIST_FILE$( ((DRY)) && echo '  [dry run]')$( ((FRESH)) && echo '  [--fresh: no checkpoint -> rerun from t=0]')"
echo

n=0; n_ok=0; n_warn=0
while IFS= read -r raw || [[ -n "${raw:-}" ]]; do
  line="$(sed -E 's/^[[:space:]]+|[[:space:]]+$//g' <<<"${raw:-}")"
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  [[ "$line" =~ ^[0-9]+$ ]] || { echo "SKIP: not a number: '$line'"; continue; }
  run="$(printf 'run_%03d' "$((10#$line))")"
  n=$((n + 1))
  prm="$BASE/$SUITE_DIR/$run/$run.prm"
  outdir="outputs/$OUT_SUBDIR/$run"
  status="ok"

  if [[ ! -f "$prm" ]]; then echo "$run  MISSING PRM $prm"; n_warn=$((n_warn + 1)); continue; fi
  ckpt="$OUTROOT/$OUT_SUBDIR/$run"
  if [[ -f "$ckpt/restart.mesh" && -f "$ckpt/restart.mesh_fixed.data" ]]; then
    status="ok: resume from checkpoint"
  elif (( FRESH )); then
    if [[ -d "$ckpt" ]]; then
      status="ok: FRESH from t=0 ($ckpt moved aside to $run.dead-$STAMP)"
      (( DRY )) || mv "$ckpt" "$ckpt.dead-$STAMP"
    else
      status="ok: FRESH from t=0 (no previous $ckpt)"
    fi
  else
    status="NO CHECKPOINT at $ckpt (restart.mesh + restart.mesh_fixed.data) -- resume impossible; use --fresh to rerun from t=0"
  fi
  [[ -d "$OUTROOT/$run" ]] && status="$status (note: $OUTROOT/$run also exists -- another suite's $run; untouched)"

  have_out="$(grep -E '^\s*set Output directory\s*=' "$prm" | sed -E 's/.*=\s*//')"
  have_cap="$(grep -E '^\s*set Maximum relative increase in time step\s*=' "$prm" | sed -E 's/.*=\s*//' || true)"
  grep -qE '^\s*set CFL number\s*=' "$prm" || { echo "$run  no 'set CFL number' line to anchor the cap on; skipped"; n_warn=$((n_warn + 1)); continue; }

  if (( ! DRY )); then
    [[ -f "$prm.orig" ]] || cp -p "$prm" "$prm.orig"
    sed -i -E "s|^(\s*set Output directory\s*=).*|\1 $outdir|" "$prm"
    if [[ -z "$have_cap" ]]; then
      sed -i -E "/^\s*set CFL number\s*=/a set Maximum relative increase in time step = $CAP   # dt may grow <= 1.$CAP x per step (ASPECT 3.0 default; 2.5 unlimited). Added for the resume." "$prm"
    else
      sed -i -E "s|^(\s*set Maximum relative increase in time step\s*=)\s*[0-9.]+|\1 $CAP|" "$prm"
    fi
  fi
  printf "%s  Output directory: %s -> %s   cap: %s -> %s   %s\n" \
    "$run" "${have_out:-?}" "$outdir" "${have_cap:-unset}" "$CAP" "$status"
  if [[ "$status" == ok* ]]; then n_ok=$((n_ok + 1)); else n_warn=$((n_warn + 1)); fi
done < "$LIST_FILE"

echo
echo "-- $n runs listed, $n_ok ready, $n_warn with warnings$( ((DRY)) && echo ' (dry run, nothing written)')"
(( DRY )) && echo "-- this was a DRY RUN: the prms are unchanged. Run again without -n before submitting."
(( n_warn )) && echo "-- $n_warn runs are NOT ready: fix the warnings (or use --fresh) before submitting; submitting now would resume from the shared outputs/run_XXX."
(( DRY )) || echo "-- next: cd $BASE/$SUITE_DIR && ./submit_from_list.sh $(realpath --relative-to="$BASE/$SUITE_DIR" "$LIST_FILE" 2>/dev/null || echo "$LIST_FILE")"
(( n_warn )) && exit 1 || exit 0

#!/bin/bash
#SBATCH -J slabpt-suite
#SBATCH -o /scratch/04714/adamholt/aspect_work/logs/slabpt-suite.%A_%a.out
#SBATCH -e /scratch/04714/adamholt/aspect_work/logs/slabpt-suite.%A_%a.err
#SBATCH -p skx
#SBATCH -t 05:00:00
#SBATCH -A TG-EES220058
#SBATCH -N 1
#SBATCH -n 48
#SBATCH --array=0-49%10

set -euo pipefail
set -x # echo commands

# ensure log dir exists (SLURM won't create it)
mkdir -p "$SCRATCH/aspect_work/logs"

# --- input file list ---
MANIFEST="$WORK/aspect_work/SlabT_emulator/production-runs/runlist.txt"
ASPECT_EXE="${asp25_skx:?asp25_skx not set}"   # fail fast if unset

# --- pick the task's directory ---
RUN_DIR_SRC=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MANIFEST" | tr -d '\r' | sed 's/^"\|"$//g')
[[ -d "$RUN_DIR_SRC" ]] || { echo "No such run dir: $RUN_DIR_SRC"; exit 1; }

# --- names/paths ---
RUN_NAME=$(basename -- "$RUN_DIR_SRC")          # e.g. run_123
echo "TASK $SLURM_ARRAY_TASK_ID  ->  DIR=$RUN_DIR_SRC  NAME=$RUN_NAME"
RUN_PRM="$RUN_DIR_SRC/$RUN_NAME.prm"            # e.g. .../run_123/run_123.prm
[[ -f "$RUN_PRM" ]] || { echo "Missing PRM: $RUN_PRM"; exit 1; }

# --- scratch workspace ---
RUN_SCRATCH="$SCRATCH/aspect_work/$RUN_NAME"
mkdir -p "$RUN_SCRATCH" "$RUN_SCRATCH/outputs"

# --- stage exe and model files ---
cp "$ASPECT_EXE" "$RUN_SCRATCH/aspect_local"
cp "$RUN_PRM" "$RUN_SCRATCH/$RUN_NAME.prm"
if [[ -d "$RUN_DIR_SRC/inputs" ]]; then
  # copy entire folder; cheap and simple (pngs ok to copy too)
  cp -r "$RUN_DIR_SRC/inputs" "$RUN_SCRATCH/inputs"
else
  echo "WARNING: inputs/ not found in $RUN_DIR_SRC (continuing anyway)"
fi

# hygiene for pure-MPI jobs
export OMP_NUM_THREADS=1

# run
cd "$RUN_SCRATCH"
ibrun ./aspect_local "$RUN_NAME.prm"

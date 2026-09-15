# TACC (Stampede3) runbook: getting a suite from this repo onto the cluster and back

Everything here was worked out for const-vc-dd100 on 2026-09-15; it applies to any suite.

## Where things live

| what | where |
|---|---|
| local run inputs | `subd-model-runs/<suite>/run-inputs/run_XXX/{run_XXX.prm, inputs/}` (gitignored, rebuilt by `src/build-numerical-mods/build_runs.<suite>.py`) |
| TACC suite dir | `$WORK/aspect_work/SlabT_emulator/production-runs_v2/<suite>/run_XXX/` (`$WORK` = `/work2/04714/adamholt/stampede3`; const-vc is `const-vc-new` there) |
| TACC job scripts | in the suite dir: `run_one.slurm`, `submit_from_list.sh`, `submit_rolling.sh` (copied with the run inputs) |
| check scripts | `production-runs_v2/check_runs.sh`, `check_runs_list.sh` (local copies: `subd-model-runs/`) |
| model outputs on TACC | `$SCRATCH/aspect_work/outputs/run_XXX/` (ASPECT runs in `$SCRATCH/aspect_work`; job logs in `$SCRATCH/aspect_work/logs/`) |
| local outputs | `subd-model-runs/<suite>/run-outputs/run_XXX/` (gitignored) |

Account: user `adamholt`, allocation TG-EAR180026, queue `skx`, 1 node / 48 cores / 10 h per run.
Every TACC login prompts for password + token, so run the transfer commands from your own terminal.

## 1. Push run inputs

```bash
make push-tacc SUITE=<suite> LINK=<tacc-suite-with-identical-inputs> DRY=1   # preview
make push-tacc SUITE=<suite> LINK=<...>                                        # transfer
```

`LINK` hard-links files that already exist in a sibling TACC suite (rsync `--link-dest`), so a paired
ablation such as const-vc-dd100 vs const-vc-new sends only the .prm files instead of ~19 GB. Drop `LINK`
for a plain full transfer. Wrapper: `src/build-numerical-mods/push_runs_to_tacc.sh`.

Submission lists and one-off scripts go with `scp` to the suite dir, e.g.

```bash
scp subd-model-runs/<suite>/run-inputs/<list>.txt \
  adamholt@stampede3.tacc.utexas.edu:/work2/04714/adamholt/stampede3/aspect_work/SlabT_emulator/production-runs_v2/<suite>/
```

## 2. Submit

From the suite dir on Stampede3. List format: one run number per line (zero-padded or not), `#`
comment lines and blank lines allowed, **no inline comments**.

```bash
cd $WORK/aspect_work/SlabT_emulator/production-runs_v2/<suite>
./submit_from_list.sh <list>.txt          # foreground; fine for a short pilot
```

For a long list (hundreds of runs, a day or more of feeding the queue) run it under `nohup` so it
survives logout:

```bash
nohup ./submit_from_list.sh <list>.txt > submit_<list>.log 2>&1 &
echo $! > submit_<list>.pid
```

Behaviour: throttles at 30 active `run_XXX` jobs (all suites count), polls `squeue` every 180 s, then
waits for the active jobs to drain. Once every run is submitted the script may be killed (Ctrl-C or
`kill $(cat submit_<list>.pid)`); Slurm jobs are independent of it. Env overrides: `THROTTLE`, `POLL`,
`BASE_DIR`, `SLURM_FILE`. `submit_rolling.sh START END` is the same for a contiguous run range.

Progress of the feeder:

```bash
tail -n 20 submit_<list>.log
grep -c '^submitted' submit_<list>.log
squeue -u adamholt -h -o "%t %j" | sort | uniq -c
```

## 3. Check

From `production-runs_v2/` (outputs root defaults to `$SCRATCH/aspect_work`; override with `BASE_DIR`):

```bash
./check_runs_list.sh <suite>/<list>.txt              # every listed run with its latest step
./check_runs_list.sh <suite>/<list>.txt 00010 -p     # problems only, target step 10 (5 Myr)
./check_runs.sh 00020 0 399                          # whole range, problem rows only; -a shows all
```

Step k = k x 0.5 Myr; step 20 = 10 Myr, 21 = 10.5 Myr (end). Exit code 1 if anything is short.
Typical pace (skx, 48 cores): 0.2-0.5 wall-h per model-Myr, i.e. 2-5 h to 10.5 Myr.

If the feeder died (login node reboot), make a copy of the list without the finished runs
(`check_runs_list.sh ... -p` gives the unfinished ones) and restart it on that.

## 4. Pull outputs back

```bash
mkdir -p subd-model-runs/<suite>/run-outputs
rsync -avh --info=progress2 \
  --include='run_*/' --include='run_*/solution/' --include='run_*/solution/**' \
  --include='run_*/solution.pvd' --include='run_*/log.txt' --include='run_*/statistics' \
  --include='run_*/parameters.prm' --exclude='*' \
  adamholt@stampede3.tacc.utexas.edu:/scratch/04714/adamholt/aspect_work/outputs/ \
  subd-model-runs/<suite>/run-outputs/
```

Incremental: rerun to pick up later steps. ~200 MB per finished run. Note `outputs/` on scratch is
shared by whatever was run there; restrict with `--include='run_010/'`-style rules if it holds more
than one suite.

## 5. Field CSVs for postprocessing

`src/utils/model_processing.py:extract_csv` (pvpython) writes `subd-model-runs/<suite>/analysis/run_XXX/t{k}.csv`
(point data: velocity, p, T, ocrust, op, viscosity, coordinates; ~33 MB each). Example driver:
`subd-model-runs/const-vc-dd100/analysis/pilot-check/extract_dd100.py RUN STEP...`, run with
`/home/holt/software/ParaView-5.11.1-MPI-Linux-Python3.9-x86_64/bin/pvpython`, parallel via `xargs -P`.

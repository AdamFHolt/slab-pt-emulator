# TACC (Stampede3) runbook: getting a suite from this repo onto the cluster and back

Everything here was worked out for const-vc-dd100 on 2026-09-15; it applies to any suite.

## Where things live

| what | where |
|---|---|
| local run inputs | `subd-model-runs/<suite>/run-inputs/run_XXX/{run_XXX.prm, inputs/}` (gitignored, rebuilt by `src/build-numerical-mods/build_runs.<suite>.py`) |
| TACC suite dir | `$WORK/aspect_work/SlabT_emulator/production-runs_v2/<suite>/run_XXX/` (`$WORK` = `/work2/04714/adamholt/stampede3`; const-vc is `const-vc-new` there) |
| TACC job scripts | in the suite dir: `run_one.slurm`, `submit_from_list.sh`, `submit_rolling.sh` (copied with the run inputs) |
| check scripts | `production-runs_v2/check_runs.sh`, `check_runs_list.sh` (local copies: `subd-model-runs/`) |
| transfer scripts | push: `subd-model-runs/push_runs_to_tacc.sh` (`make push-tacc`); pull: `subd-model-runs/pull_runs_from_tacc.sh` (`make pull-tacc`) |
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
for a plain full transfer. Wrapper: `subd-model-runs/push_runs_to_tacc.sh`.

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

**const-vc-sh and const-vc-v3ctrl differ** (2026-09-22): their jobs are named `sh_XXX` / `v3c_XXX`
(run directories stay `run_XXX`; the prefix is only the Slurm job name and the log-file name), and
the throttle counts *every* active job of ours in the batch script's partition regardless of name,
default 24 = the spr per-user queue limit. So the skx dd100 jobs do not hold them up, and the two
3.0 feeders share one spr counter. A rejected `sbatch` is retried after `POLL` seconds instead of
dropping the run (the 2.5 feeders die on it). Where a recipe below strips `run_` from job names, use
`sed -E 's/^(run|sh|v3c)_//'` for these suites.

Optional: an email when the feeder exits (success or crash). TACC login nodes can send mail
(`echo test | mail -s test aholt@miami.edu` to confirm). Put this in the suite dir as `watcher.sh`
and start it with `nohup ./watcher.sh > watcher.log 2>&1 &`; keep the pid file name in step:

```bash
#!/usr/bin/env bash
PID=$(cat submit_<list>.pid)
echo "watching feeder pid $PID from $(date)"
while ps -p "$PID" > /dev/null 2>&1; do sleep 600; done
{ echo "feeder exited $(date)"; echo; tail -n 15 submit_<list>.log; echo
  echo "jobs by state (today):"
  sacct -u adamholt -S "$(date +%F)" -n -o JobName,State | grep run_ | awk '{print $2}' | sort | uniq -c
} | mail -s "<suite> feeder exited" aholt@miami.edu
echo "mail sent $(date)"
```

Reading the email: last log line `Done at ...` = finished normally; `All listed jobs submitted...` =
died while draining (harmless); `submitted run_XXX` / `At throttle` = died mid-feed -> build a
remainder list (below) and restart. The feeder runs `set -euo pipefail`, so one failed `squeue`/`sbatch`
call kills it; expect to restart it occasionally on a long list.

Progress of the feeder:

```bash
tail -n 20 submit_<list>.log
grep -c '^submitted' submit_<list>.log
squeue -u adamholt -h -o "%t %j" | sort | uniq -c
```

## 3. Check

From `production-runs_v2/` (outputs root defaults to `$SCRATCH/aspect_work`; override with `BASE_DIR`.
**const-vc-sh and const-vc-v3ctrl run in their own workspace**, so for them prefix every check with
`BASE_DIR=$SCRATCH/aspect_work/<suite>` -- the default root holds the 2.5 suites' runs of the same numbers):

```bash
./check_runs_list.sh <suite>/<list>.txt              # every listed run with its latest step
./check_runs_list.sh <suite>/<list>.txt 00010 -p     # problems only, target step 10 (5 Myr)
./check_runs.sh 00020 0 399                          # whole range, problem rows only; -a shows all
```

Step k = k x 0.5 Myr; step 20 = 10 Myr, 21 = 10.5 Myr (end). Exit code 1 if anything is short.
Typical pace (skx, 48 cores): 0.2-0.5 wall-h per model-Myr, i.e. 2-5 h to 10.5 Myr.

If the feeder died, restart it on a remainder list = listed runs that are neither finished nor queued:

```bash
cd $WORK/aspect_work/SlabT_emulator/production-runs_v2
squeue -u adamholt -h -o "%j" | sed -E 's/^(run|sh|v3c)_//' | sort > /tmp/queued.txt
./check_runs_list.sh <suite>/<list>.txt -p | awk '$2!="OK" && $1 ~ /^run_/ {sub("run_","",$1); print $1}' | sort > /tmp/notdone.txt
comm -23 /tmp/notdone.txt /tmp/queued.txt > <suite>/remainder.txt
cd <suite> && nohup ./submit_from_list.sh remainder.txt > submit_remainder.log 2>&1 & echo $! > submit_remainder.pid
```

Failed individual runs: `sacct -u adamholt -S <date> -o JobID,JobName,State,Elapsed,ExitCode | grep -E 'FAILED|TIMEOUT|CANCELLED|OUT_OF_ME'`;
diagnosis in `$SCRATCH/aspect_work/logs/run_XXX.<jobid>.err`. `TIMEOUT` = 10 h wall clock: resubmit the
run number, `Resume computation = auto` continues from the last checkpoint (every 25 steps).

## 4. Pull outputs back

```bash
make pull-tacc SUITE=<suite> ALL=1 DRY=1   # preview
make pull-tacc SUITE=<suite> ALL=1         # transfer
```

Wrapper: `subd-model-runs/pull_runs_from_tacc.sh <suite> [LIST_FILE ...] [-a] [-n]`, which lives next
to the check scripts. Takes `solution/`, `solution.pvd`, `log.txt`, `statistics` and `parameters.prm`
per run into `subd-model-runs/<suite>/run-outputs/`, and leaves `restart.*` checkpoints on TACC so a
timed-out run can still resume from them.

`ALL=1` (`-a`) pulls every `run_*` in the scratch outputs dir. **Only use it when that dir holds one
suite**: `$SCRATCH/aspect_work/outputs/` is shared by whatever was run there (const-vc, ramped-vc,
dd100) and `run_XXX` numbers collide across suites, so an unrestricted pull silently mixes them into
one directory with nothing to tell them apart. const-vc-sh and const-vc-v3ctrl are exempt: they run in
`$SCRATCH/aspect_work/<suite>/`, the pull script defaults to that outputs dir for them, and `ALL=1` is
safe. Without `ALL=1` the transfer is restricted to the runs named in the suite's
`run-inputs/*list*.txt` (for const-vc-dd100 that is 324: the 320-run record plus the four dip < 35
pilot runs kept as evidence of the flattening). Name list files explicitly to narrow it further:

```bash
subd-model-runs/pull_runs_from_tacc.sh <suite> <suite>/run-inputs/pilot-list.txt
```

Incremental: rerun to pick up later steps, and to top up runs that were still going or had died
part-way. ~200 MB per finished run, one password + token prompt per invocation. Partial `.vtu` files
survive an interrupted transfer (`--partial-dir`). The run ends with a local count of runs holding
step 00021, so the completion figure costs no further TACC login.

## 5. Field CSVs for postprocessing

`src/utils/model_processing.py:extract_csv` (pvpython) writes `subd-model-runs/<suite>/analysis/run_XXX/t{k}.csv`
(point data: velocity, p, T, ocrust, op, viscosity, coordinates; ~33 MB each). Example driver:
`subd-model-runs/const-vc-dd100/analysis/pilot-check/extract_dd100.py RUN STEP...`, run with
`/home/holt/software/ParaView-5.11.1-MPI-Linux-Python3.9-x86_64/bin/pvpython`, parallel via `xargs -P`.

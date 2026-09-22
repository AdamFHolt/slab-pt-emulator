# const-vc-sh: const-vc with shear heating

## What it is

One-term extension of const-vc: the viscous dissipation 2 eta (eps_dev : eps_dev) is added to the
temperature equation. Same 400-point design (`data/params/params-list.const-vc-sh.*` are copies of
the const-vc lists), same geometry, rheology, boundary conditions, solver settings and output
cadence, so `run_XXX` pairs with const-vc `run_XXX` and const-vc-dd100 `run_XXX`. Built by
`src/build-numerical-mods/build_runs.const-vc-sh.py` from the const-vc run-inputs (plate input files
hard-linked); reference template `data/ref-model/model_template_fixed-trench.const-vc-sh.prm`.
Planned 2026-05-27 as the third suite (the shear-heating contrast is the likely headline of the
project); the working template came from a collaborator on 2026-09-16 (`example/`); suite built
2026-09-17. Not yet submitted.

Rough size of the effect: in the 6 km crust channel (fixed eta = 1e20) at v_conv = 4 cm/yr the
shear strain rate is ~2e-13 /s, so H = 2 eta eps:eps ~ 4e-6 W/m^3, i.e. ~30 K/Myr against a
channel diffusion time of ~1 Myr. Tens of degrees at the slab top, scaling with v_conv^2 -- large
against the emulator's ~2-6 C/Myr validation RMSE.

## Reconciliation of the collaborator's test file with the production setup (2026-09-17)

Target ASPECT version: **3.0 or a 3.x development build** (the collaborator's requirement; const-vc,
ramped-vc and dd100 ran under 2.5). `example/run_100.prm` is byte-identical to const-vc `run_100.prm`.
`example/run_heating.prm` is that file with shear heating on plus a set of unrelated test-run edits.
Decision per line:

| change in `run_heating.prm` | kept for const-vc-sh? | why |
|---|---|---|
| `subsection Heating model` / `List of model names = shear heating` | **yes** | the physics being added |
| `Formulation = custom`, `Mass conservation = incompressible`, `Temperature equation = reference density profile` (replacing `Boussinesq approximation`) | **yes** | ASPECT aborts when the shear-heating plugin meets the `Boussinesq approximation` keyword ("expects shear heating to be disabled"). The two custom choices are the ones the Boussinesq keyword selects internally, so the equations are unchanged except for the added term; no `adiabatic heating` is listed, so nothing else enters |
| `Limit stress contribution to shear heating = true`, `Cohesion for maximum shear stress = 10e6`, `Friction angle for maximum shear stress = 0.523599` | **yes** | available from 3.0 (absent in 2.5, which is why 3.x is needed). Caps the stress used in the heating term at the Drucker-Prager yield stress cohesion*cos(phi) + P*sin(phi), P clamped at >= 0, with the same 10 MPa / 30 deg as the mantle plasticity. The plugin takes the angle in radians (no degree conversion in `shear_heating.cc`, unlike the material model), so 0.523599 is right. In practice it only binds in the top ~1-2 km: the crust channel is a fixed 1e20 and carries ~20-40 MPa, while the cap is already ~90 MPa at 5 km depth |
| `Stokes solver type = block AMG` | **yes** | not a no-op under 3.x: 3.0 made GMG the default Stokes solver. Setting block AMG restores the solver const-vc ran with (2.5 default); GMG would also need material averaging, which const-vc does not use |
| `Use years instead of seconds` (renamed keyword) | **no** | we keep `Use years in output instead of seconds`: it is the only spelling 3.0.0 knows, and the development branch keeps it as a deprecated alias (a warning on rank 0), so the const-vc line works on both |
| `End time = 2.5e6`, `Resume computation = false` | **no** | test-run settings; production is 10.5 Myr with `auto` resume (10 h wall, checkpoint every 25 steps) |
| `Output directory = /scratch2/09571/zguo4/...`, `Data directory = /scratch2/09571/zguo4/` | **no** | collaborator's paths; ours are `outputs/run_XXX` and `inputs/` |
| `Nonlinear solver tolerance 1e-4` (ours 2e-4), `Linear solver tolerance 1e-5` (ours 1e-4), `Temperature/Composition solver tolerance 1e-7` (ours 1e-9) | **no** | solver settings must match const-vc for a paired ablation; changing them would alter the pair difference for reasons that have nothing to do with heating |
| `List of output variables = strain rate, nonadiabatic pressure, shear stress, stress, material properties, heating, dynamic topography` + `Material properties = viscosity, density`, `Number of grouped files = 16` | **partly** | keep viscosity and add **`heating`** (W/m^3, one scalar) so the dissipation can be inspected; the rest is diagnostic bulk we do not process. Grouped files stays 1. **Corrected 2026-09-22:** the `material properties` + `Material properties = viscosity` form is not optional under 3.x -- 3.0 removed the standalone `viscosity` postprocessor and aborts at startup on the 2.5 spelling, which is why the collaborator's file had it. The vtu field is still named `viscosity`, so the field-CSV extraction is unaffected. The same fix is applied to `const-vc-v3ctrl` (without `heating`), or it would not start either |

Net change per .prm: one solver line, the Formulation block, the Heating model block, and one word in
the output list. `build_runs.const-vc-sh.py` asserts exactly that for every run.

## The version confound, and the control set

const-vc was produced with ASPECT 2.5; const-vc-sh will be produced with 3.x. Anything that changed
between the two versions (solver internals, material-model details, time stepping) lands in the
pair difference alongside the heating. To separate the two:

- `build_runs.const-vc-sh.py --control` also writes **`subd-model-runs/const-vc-v3ctrl/run-inputs/`**:
  the 8 pilot-list const-vc inputs with *only* the `block AMG` line made explicit (no heating).
  Run those under the same 3.x binary and compare slab-top T(z) at 1/5/10 Myr with the existing
  const-vc outputs of the same runs.
- If the version effect is at the level of the solver tolerances (a few C), pair const-vc-sh with the
  2.5 const-vc record as planned. If it is comparable to the heating signal, the clean options are to
  re-run const-vc under 3.x (400 runs, ~10 h each) or to pair const-vc-sh against the v3ctrl subset only.

## Machine, workspace and submission order (settled 2026-09-22)

1. **Binary.** `$asp3_skx` = `$WORK/software/aspect/build-3.0/aspect-release`, ASPECT 3.0.0 on the
   2.5 toolchain (deal.II 9.5.1, Trilinos 15, p4est 2.8.5; recipe and traps in SESSION_NOTES
   2026-09-22). `run_one*.slurm` take it from the environment: **export it before starting the
   feeder** -- `--export=ALL` carries exported variables only, and a batch script does not source
   `~/.bashrc`. `run_900`/`run_901` under `run-inputs/` are hand-made copies of run_089 truncated to
   0.5 Myr (skx and spr timing benchmarks; numbered outside the 000-399 design).
2. **Production runs on spr, 112 ranks** (`SLURM_FILE=run_one.spr.slurm`). Same skx-built binary
   (SPR's instruction set is a superset of SKX's). Verified on run_901: 112 MPI processes; peak node
   RSS 45 GB at 0.5 Myr against 128 GB HBM, where a complete 48-rank 2.5 run peaks at 22 GB total, so
   the adaptive mesh cannot grow into the limit; 6 min to 0.5 Myr against ~11 estimated for skx/48.
   spr costs 2 SU/node-h (skx 1), so about the same SUs per run at half the wall time. **spr allows
   24 jobs in queue per user** (skx 40). The submitters here differ from the 2.5 suites' accordingly:
   jobs are named `sh_XXX` (v3ctrl: `v3c_XXX`; run directories stay `run_XXX`), the throttle counts
   every active job of ours in the batch script's partition regardless of name (default 24), so the
   skx dd100 jobs do not block it and the two 3.0 feeders share one spr counter, and a rejected
   `sbatch` is retried after `POLL` s rather than dropped. (Found the hard way: with the inherited
   name-based throttle, dd100's 30 queued `run_XXX` jobs kept the v3ctrl feeder asleep at 30/24.)
   `const-vc-v3ctrl` runs on spr too (same queue and rank count as production, or it stops
   controlling for the decomposition).
3. **Own scratch workspace.** The batch scripts `cd` to `$SCRATCH/aspect_work/const-vc-sh/` (v3ctrl:
   `.../const-vc-v3ctrl/`), not `$SCRATCH/aspect_work/` where const-vc, ramped-vc and dd100 ran and
   share `outputs/run_XXX`. There, with `Resume computation = auto`, a const-vc-sh run_XXX would have
   resumed from -- or overwritten -- the dd100 run of the same number. Consequences: `check_runs*.sh`
   need `BASE_DIR=$SCRATCH/aspect_work/const-vc-sh`, and `pull_runs_from_tacc.sh` defaults to that
   workspace for these two suites (so `-a` is safe). Job logs stay in the shared `logs/` (job id in
   the name).
4. **Submission order: `run-inputs/full-list.txt`** (`src/build-numerical-mods/make_full_list.const-vc-sh.py`).
   The 8 pilot runs first -- 100 (the collaborator's test case), the three fastest-converging (most
   channel dissipation, ~v_conv^2), two high v_conv^2 x eta_UM (most wedge dissipation), a median
   case and the weakest -- then the other 392 by greedy maximin in the normalised design space, so any
   prefix of the list is roughly space-filling and a suite stopped early still covers the parameter
   space. No separate pilot submission: the pilot is the head of the list, and its first finishers
   (~2 h at spr speed for the fast ones) are where to look before the bulk lands. Checks on them:
   3.x prints only the years-keyword deprecation warning; slab-top T(z) and the `heating` field at 1
   and 5 Myr; the `op` interior stays static; wall time against the const-vc twin. Kill the feeder if
   anything is off (Slurm jobs are independent of it).
5. Size: const-vc's 399 completed runs took 0.17-6.8 h each on skx/48 (median 3.4 h, 1364 node-h in
   all; wall time scales with v_conv, ~1.8 h for the slowest-converging third to ~5 h for the
   fastest). Expect roughly half that per run on spr, i.e. on the order of a day and a half of
   continuous feeding at 24 concurrent jobs, more while dd100 jobs still occupy throttle slots.
   `heating` adds roughly one scalar field to each vtu (const-vc writes T, p, velocity, two
   compositions, viscosity), ~15% more output.

## How to operate it

```bash
python src/build-numerical-mods/build_runs.const-vc-sh.py --control   # .prm + scripts, both suites
python src/build-numerical-mods/make_full_list.const-vc-sh.py          # run-inputs/full-list.txt
make push-tacc SUITE=const-vc-sh LINK=const-vc-new DRY=1     # preview; .prm, scripts and lists travel
make push-tacc SUITE=const-vc-sh LINK=const-vc-new
make push-tacc SUITE=const-vc-v3ctrl LINK=const-vc-new         # 8-run version control
# on Stampede3
export asp3_skx=$WORK/software/aspect/build-3.0/aspect-release
cd $WORK/aspect_work/SlabT_emulator/production-runs_v2/const-vc-v3ctrl
SLURM_FILE=run_one.spr.slurm nohup ./submit_from_list.sh pilot-list.txt > submit_v3ctrl.log 2>&1 &
cd ../const-vc-sh
SLURM_FILE=run_one.spr.slurm nohup ./submit_from_list.sh full-list.txt > submit_full.log 2>&1 &
echo $! > submit_full.pid
# feeders: jobs appear as v3c_XXX / sh_XXX in squeue; both throttle on the spr partition (24)
# progress / pull (own workspace, so -a is safe)
BASE_DIR=$SCRATCH/aspect_work/const-vc-sh ../check_runs_list.sh const-vc-sh/full-list.txt -p
make pull-tacc SUITE=const-vc-sh ALL=1
```

Then `docs/tacc-runbook.md` sections 3-5 as for the other suites, with `BASE_DIR` pointing at the
suite's own workspace. Downstream, the suite slots into
the standard pipeline (`extend_profiles_all-mods.sh const-vc-sh ...`, `build_master_dt.py`,
per-suite emulator configs) once outputs exist; wrappers that hardcode `const-vc ramped-vc` need
`const-vc-sh` added.

## Layout

- `example/` -- collaborator's files as received (`run_100.prm`, `run_heating.prm`, `readme.txt`, `shear_heating.png`)
- `run-inputs/run_XXX/` -- 400 derived inputs (gitignored; regenerate with the builder)
- `run-inputs/pilot-list.txt` -- the 8 pilot runs (also the head of the full list, and the v3ctrl set)
- `run-inputs/full-list.txt` -- all 400 in submission order (`make_full_list.const-vc-sh.py`)
- `run-inputs/bench-{skx,spr}-list.txt`, `run_900/`, `run_901/` -- timing benchmarks (hand-made, not
  builder output; on disk only)
- `run-inputs/run_one.slurm` / `run_one.spr.slurm` -- one-job batch scripts for skx (48 ranks) and
  spr (112 ranks); both generated by the builder, picked with `SLURM_FILE`; both work in
  `$SCRATCH/aspect_work/const-vc-sh/`
- `../const-vc-v3ctrl/run-inputs/` -- version-control set (same 8 runs, no heating, for the 3.x binary)

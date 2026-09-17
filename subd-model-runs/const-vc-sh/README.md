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
| `List of output variables = strain rate, nonadiabatic pressure, shear stress, stress, material properties, heating, dynamic topography` + `Material properties = viscosity, density`, `Number of grouped files = 16` | **partly** | keep our `viscosity` (the field-CSV extraction expects it) and add **`heating`** (W/m^3, one scalar) so the dissipation can be inspected; the rest is diagnostic bulk we do not process. Grouped files stays 1 |

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

## Open points before submission

1. **Binary and machine.** `run_one.slurm` still loads the 2.5 toolchain and now stops with a TODO on
   `asp30_skx`; the 3.x build (module set + executable path) has to be filled in. The collaborator's
   paths are on a `/scratch2/...` filesystem, i.e. probably Frontera rather than Stampede3.
2. **Pilot first.** `run-inputs/pilot-list.txt` (8 runs): 100 (the collaborator's test case), the three
   fastest-converging runs (most channel dissipation, ~v_conv^2), two high v_conv^2 x eta_UM runs
   (most wedge dissipation), a median case and the weakest case. Check: parameters accepted (3.x
   prints a deprecation warning for the years keyword, nothing more), wall time per Myr vs the
   const-vc twin, slab-top T(z) and the `heating` field at 1 and 5 Myr, and that the `op` interior
   stays static. Submit the v3ctrl set with it.
3. `heating` output adds roughly one scalar field to each vtu (const-vc writes T, p, velocity, two
   compositions, viscosity), ~15% more output.

## How to operate it

```bash
make push-tacc SUITE=const-vc-sh LINK=const-vc-new DRY=1     # preview; only .prm + scripts travel
make push-tacc SUITE=const-vc-sh LINK=const-vc-new
make push-tacc SUITE=const-vc-v3ctrl LINK=const-vc-new         # 8-run version control
scp subd-model-runs/const-vc-sh/run-inputs/pilot-list.txt adamholt@stampede3.tacc.utexas.edu:/work2/04714/adamholt/stampede3/aspect_work/SlabT_emulator/production-runs_v2/const-vc-sh/
# on Stampede3
cd $WORK/aspect_work/SlabT_emulator/production-runs_v2/const-vc-sh
nohup ./submit_from_list.sh pilot-list.txt > submit_pilot.log 2>&1 &
```

Then `docs/tacc-runbook.md` sections 3-5 as for the other suites. Downstream, the suite slots into
the standard pipeline (`extend_profiles_all-mods.sh const-vc-sh ...`, `build_master_dt.py`,
per-suite emulator configs) once outputs exist; wrappers that hardcode `const-vc ramped-vc` need
`const-vc-sh` added.

## Layout

- `example/` -- collaborator's files as received (`run_100.prm`, `run_heating.prm`, `readme.txt`, `shear_heating.png`)
- `run-inputs/run_XXX/` -- 400 derived inputs (gitignored; regenerate with the builder)
- `run-inputs/pilot-list.txt` -- suggested pilot
- `../const-vc-v3ctrl/run-inputs/` -- version-control set (same 8 runs, no heating, for the 3.x binary)

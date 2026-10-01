# const-vc-shp: shear-heating pilot with a BRITTLE-DUCTILE crust channel

("shp" = shear heating + plasticity.)

## What it is

The 8 pilot-list runs of const-vc-sh (100 310 195 210 072 217 270 064) with the plate-interface channel
-- the 6 km ocrust layer, its first phase (z < 150 km) -- given a realistic interface rheology instead of
sh's fixed 1e20 Pa s: frictional where cold, creeping where hot. Spot check 2 of 2 in
`docs/paper-plan.md` section 2; spot check 1 is const-vc-sh-mu05 (heating cap only, flow unchanged).
Built by `build_runs.const-vc-sh.py --shp` from the const-vc-sh .prm files (inputs hard-linked). Nine
lines change per .prm, all in the ocrust composition's first phase:

1. **Drucker-Prager friction**: `Cohesions` 1e10 -> 1e6 Pa, `Angles of internal friction` 30 -> 2.866 deg
   (= asin 0.05, degrees in the material model), so tau_y = 1 MPa cos(phi) + P sin(phi) ~ 1 MPa + 0.05 rho g z
   (Kohn et al. 2018 mu' = 0.05).
2. **Wet-quartzite dislocation creep** (Hirth et al. 2001: log10 A = -11.2 MPa^-n s^-1, n = 4,
   Q = 135 kJ/mol, water-fugacity exponent 1, f_H2O fixed at 1 GPa), activation volume 0. ASPECT's creep
   law is written in invariants, so A_ASPECT = A_lab f 3^((n+1)/2) / 2 = 4.917826e-32 Pa^-4 s^-1. The
   stress exponent and activation energy become per-composition maps (3.5 / 530 kJ elsewhere, unchanged).
3. **Diffusion creep off** in the channel (prefactor 1e-40); it was the mantle's olivine law.
4. **Viscosity window** 0.995e20-1.005e20 -> 2.5e18-2.5e23: neither bound controls the channel. The floor
   has to come down with the yield because ASPECT clamps after yielding (visco_plastic.cc l. 387).

The **shear-heating limiter is unchanged** (sh's 10 MPa / 30 deg backstop): the channel's stress is set by
its mechanics. A global mu' = 0.05 cap was tried in mu05 and mostly cut heating inside the incoming plate
(SESSION_NOTES 2026-10-01), which is not a channel effect.

Why a lab flow law and not an idealized n = 1 law: the stress exponent sets how heating scales with
convergence rate (n = 4: stress ~ v^0.25, heating ~ v^1.25; sh's linear channel: heating ~ v^2), and that
scaling drives the paper's deep Sobol result. Deeper ocrust phases (> 150 km), the mantle, the plates,
geometry, BCs, solver, output cadence and ASPECT 3.0 on spr/112 are const-vc-sh's, so run_XXX pairs with
const-vc / v3ctrl / sh / mu05. Jobs `shp_XXX`, own scratch workspace `$SCRATCH/aspect_work/const-vc-shp/`.

The formulas were checked against the ASPECT 3.0 source on TACC (2026-10-01): creep
`dislocation_creep.cc` l. 98-105, 2D yield C cos(phi) + max(P, 0) sin(phi) `drucker_prager.cc` l. 96-101 +
`visco_plastic.cc` l. 324, temperature for viscosity T + 9.24e-9 K/Pa * P l. 149, clamp after yield l. 387.

## What to expect

Pre-build check: `src/build-numerical-mods/channel_strength_envelope.py` ->
`plots/science-numerical-mods/const-vc-shp/channel_strength_envelope.png` (envelopes along median const-vc
slab-top paths per v_conv tercile). Per pilot run, along its OWN const-vc slab-top path (no heating), with
the channel taking the whole convergence as simple shear (eps_II = v / 2h):

| run | v_conv (cm/yr) | brittle-ductile transition at 1 / 5 / 10 Myr (km) | stress at 40 km, 5 Myr: sh / shp (MPa) | at 80 km, 5 Myr: sh / shp (MPa) |
|---|---|---|---|---|
| 100 | 3.7 | 22 / 35 / 43 | 19 / 32 | 19 / 3.9 |
| 310 | 8.0 | 24 / 41 / 51 | 42 / 66 | 42 / 6.2 |
| 195 | 8.0 | 30 / 49 / 62 | 42 / 66 | 42 / 14.6 |
| 210 | 7.9 | 25 / 40 / 49 | 42 / 64 | 42 / 2.5 |
| 072 | 7.7 | 22 / 39 / 48 | 41 / 55 | 41 / 2.9 |
| 217 | 7.2 | 25 / 41 / 53 | 38 / 66 | 38 / 6.4 |
| 270 | 3.1 | 22 / 34 / 43 | 16 / 27 | 16 / 2.7 |
| 064 | 1.0 | 17 / 24 / 30 | 5 / 3 | 5 / 0.5 |

The transition sits at 250-320 C and deepens with time and convergence rate as the slab top cools. So,
compared with sh: MORE heating above the transition (20-50 km; peak flux 90-150 mW/m2 at mid/fast v) and
MUCH LESS below it (60-100 km stresses 0.5-15 MPa against sh's 5-42). This is a prediction to test, not a
result, because: (a) the paths have no heating -- shp's own heating warms the channel and makes the
transition shallower, which limits itself; (b) the slab top is the cold edge of the 6 km channel, so creep
will localise on the hot wedge side; (c) dynamic pressure is ignored; (d) f_H2O x3 changes creep stress by
1.3x and the transition by a few km. The friction also acts on ALL ocrust above 150 km, including the crust
of the incoming plate (low-friction crust in the outer-rise bending zone), as in any composition-based channel.

## What to check (in this order)

1. **Startup / solver** (smoke test, mandatory for a material-model change): one run first. Watch the first
   timesteps' nonlinear iteration counts and that no `linear_solver_failed` appears. The n = 4 creep plus
   friction is more nonlinear than sh's fixed channel, so iteration count and wall time are the risk.
2. **Implementation, live**: channel viscosity from the first outputs against the envelope prediction at
   the local T and P (the `viscosity` and `T` fields are in the output; strain rate from `velocity`).
3. **Flow**: trench coupling (is the overriding plate dragged?), slab dip / flattening vs const-vc-sh at 2,
   5, 10 Myr, wedge decoupling depth, and where along the interface the channel is frictional vs creeping.
4. **Temperatures**: slab-top T(z) at 1 / 5 / 10 Myr for the 8 runs across const-vc / sh / mu05 / shp
   (`src/science-numerical-mods/compare_pilot_triplets.py --suites const-vc const-vc-sh const-vc-sh-mu05
   const-vc-shp`), channel stress vs depth, the `shear_heating` field. Decision rule (paper-plan section 2):
   30-80 km heating and the eta_UM Sobol role qualitatively as in sh -> the 8 are the robustness paragraph;
   otherwise the full paired shp suite (400) becomes the paper's heating ablation.

## How to operate it

```bash
python src/build-numerical-mods/build_runs.const-vc-sh.py --shp                # (re)build; const-vc-sh byte-identical
make push-tacc SUITE=const-vc-shp LINK=const-vc-sh DRY=1                        # inputs hard-link against const-vc-sh
make push-tacc SUITE=const-vc-shp LINK=const-vc-sh
# on Stampede3
export asp3_skx=$WORK/software/aspect/build-3.0/aspect-release
cd $WORK/aspect_work/SlabT_emulator/production-runs_v2/const-vc-shp
# smoke test = ONE run as a normal batch job with a short limit (no idev needed): watch the log, then decide
SBATCH_TIMELIMIT=00:40:00 sbatch -J shp_100 -o $SCRATCH/aspect_work/logs/shp_100.%j.out -e $SCRATCH/aspect_work/logs/shp_100.%j.err \
    --export=ALL,RUN_ID=100,BASE_DIR=$PWD run_one.spr.slurm
grep -E 'Timestep|Nonlinear|Exception|linear_solver_failed' $SCRATCH/aspect_work/logs/shp_100.*.out | tail -20
# if it steps cleanly, feed the other seven (run_100 resumes from its checkpoint when resubmitted):
grep -v '^100$' pilot-list.txt > pilot-list.rest.txt
SLURM_FILE=run_one.spr.slurm nohup ./submit_from_list.sh pilot-list.rest.txt > submit_shp.log 2>&1 &
# from production-runs_v2/ (the lists live in the suite dir on TACC, not in run-inputs/):
BASE_DIR=$SCRATCH/aspect_work/const-vc-shp ./check_runs_list.sh const-vc-shp/pilot-list.txt -p
# local
subd-model-runs/pull_runs_from_tacc.sh const-vc-shp -a
src/postproc-numerical-mods/extend_profiles_all-mods.sh const-vc-shp 0:20 "1,20;10,20" 8
```
If the 40 min smoke job is killed by its limit before you decide, a plain resubmit resumes it
(`Resume computation = auto`, own workspace). Unset `SBATCH_TIMELIMIT` for the feeder.

## Not in this pilot

mu' or f_H2O as design dimensions (paper 2), other channel flow laws (blueschist, serpentinite),
rate-and-state friction, a channel thinner than the 6 km crust (resolution), yield in the crust's deeper
phases (> 150 km). History: the first --shp build (2026-10-01 morning, never pushed) used friction on the
mantle olivine law with a 1e21 ceiling and the mu' = 0.05 heating cap -- frictional to 150 km, 130 MPa at
80 km; replaced the same day (SESSION_NOTES).

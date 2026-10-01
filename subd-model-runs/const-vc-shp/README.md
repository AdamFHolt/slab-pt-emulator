# const-vc-shp: shear-heating pilot with a FRICTIONAL crust channel (mu' = 0.05)

## What it is

The 8 pilot-list runs of const-vc-sh (100 310 195 210 072 217 270 064) with the plate-interface channel
made mechanics-consistent with the thermal-modelling convention tau = mu' rho g z. Spot check 2 of 2 in
`docs/paper-plan.md` section 2 (decided 2026-09-29); spot check 1 is const-vc-sh-mu05 (heating cap only,
flow unchanged). Built by `build_runs.const-vc-sh.py --shp` from the const-vc-sh .prm files (inputs
hard-linked); three edits per .prm, all in the ocrust composition's first phase (z < 150 km, the 6 km
channel) and the heating model:

1. **Drucker-Prager yield in the channel**: `Cohesions` ocrust 1e10 -> 1e6 Pa, `Angles of internal
   friction` ocrust 30 -> 2.866 deg (= asin 0.05; the material model takes degrees), so
   tau_y = 1 MPa cos(phi) + P sin(phi) ~ 1 MPa + 0.05 rho g z (Kohn et al. 2018 mu' = 0.05).
2. **Viscosity window opened**: ocrust `Minimum viscosity` 0.995e20 -> 2.5e18 (the global floor) and
   `Maximum viscosity` 1.005e20 -> 1e21 (`--shp-eta-max=1e22` for the stiffer variant). The channel is no
   longer a fixed 1e20 Pa s layer: eta = min(eta_max, tau_y / 2 eps_II). The floor HAS to come down with the
   yield -- ASPECT clamps the yield-limited viscosity to [min, max] afterwards, so at the old 0.995e20 floor
   the friction law could only ever stiffen the channel.
3. **Heating cap set to the same law**: cohesion 1e6, friction angle 0.050021 rad (as in mu05), so the stress in
   the dissipation term is the mechanical stress.

Everything else (design, geometry, mantle/plate rheology incl. the mantle's 10 MPa / 30 deg plasticity,
boundary conditions, solver, output cadence, ASPECT 3.0 on spr/112) is const-vc-sh's, so run_XXX pairs with
const-vc / v3ctrl / sh / mu05. Scripts: jobs `shp_XXX`, own scratch workspace
`$SCRATCH/aspect_work/const-vc-shp/`, spr feeder throttle 24 shared with sh_/mu_ jobs.

## What to expect

With the shear in the channel, sh's stress is tau_sh = eta v/h (1e20 x v_conv / 6 km: 5-42 MPa, depth-
independent), shp's is min(10 tau_sh, 1 MPa + 0.05 rho g z). So shp is WEAKER than sh above
z* = (tau_sh - 1 MPa)/(0.05 rho g) (where mu05 also caps the heating) and STRONGER below it, up to the
depth z_c where the 1e21 viscous ceiling takes over (10 tau_sh). Heating ~ tau x strain rate, so if the
channel keeps accommodating v_conv the heating ratio shp/sh is tau_y(z)/tau_sh (ratios below assume that;
the point of the pilot is to see whether it holds, see "What to check"):

| run | v_conv (cm/yr) | tau_sh (MPa) | z* (km): shp weaker above | z_c (km): 1e21 cap binds below | heating shp/sh at 50 km | at 80 km |
|---|---|---|---|---|---|---|
| 100 | 3.7 | 19 | 11 | 117 | 4.3 | 6.9 |
| 310 | 8.0 | 42 | 25 | 150+ | 2.0 | 3.1 |
| 195 | 8.0 | 42 | 25 | 150+ | 2.0 | 3.1 |
| 210 | 7.9 | 42 | 25 | 150+ | 2.0 | 3.1 |
| 072 | 7.7 | 41 | 25 | 150+ | 2.0 | 3.2 |
| 217 | 7.2 | 38 | 23 | 150+ | 2.2 | 3.4 |
| 270 | 3.1 | 16 | 9 | 98 | 5.1 | 8.2 |
| 064 | 1.0 | 5 | 2 | 30 | 16.4 | 26.1 |

z_c > 150 km means the channel is yield-limited over its whole depth range; for the two slow runs the 1e21
ceiling binds below ~100 km and for run_064 everywhere below 30 km (a 1e21 layer, 10x sh's). The strong
deep channel (tau_y = 130 MPa at 80 km, 245 MPa at 150 km) is the regime the thermal-model literature
assumes down to the decoupling depth; the mantle wedge at eta_UM (2e19-1e21) cannot sustain such stresses
at the channel strain rate, so part of the convergence will move out of the channel into the wedge corner
(the eta_UM redistribution seen in sh, stronger). That is the physics being tested, not a bug -- but it is
also why the FLOW must be checked before any temperature is read.

## What to check (in this order)

1. **Startup / solver** (smoke test, mandatory for a material-model change): one run first, watch the first
   timesteps' nonlinear iteration counts and that no `linear_solver_failed` appears. The mantle already uses
   DP plasticity with the same scheme (single Advection, iterated Stokes, tol 2e-4, 200 it), so the risk is
   iteration count / wall time, not a startup abort.
2. **Flow**: trench coupling (does the overriding plate get dragged?), slab dip / flattening vs const-vc-sh
   at 2, 5, 10 Myr, wedge decoupling depth (where does the slab-top velocity jump disappear?), and the
   channel's effective viscosity along the interface (is the yield branch active where intended?). dd100
   taught that a stiffer deep interface changes the deep flow (+37 C at 100 km); here the deep channel is
   up to 10x stiffer.
3. **Temperatures**: slab-top T(z) at 1 / 5 / 10 Myr for the 8 runs across const-vc / sh / mu05 / shp
   (the pilot figure), channel shear stress vs depth, the `heating` field. Decision rule (paper-plan
   section 2): 30-80 km heating and the eta_UM Sobol role qualitatively as in sh -> the 8 are the
   robustness paragraph; otherwise the full paired shp suite (400, mu' = 0.05 fixed) becomes the paper's
   heating ablation.

## How to operate it

```bash
python src/build-numerical-mods/build_runs.const-vc-sh.py --shp                # (re)build; const-vc-sh untouched
make push-tacc SUITE=const-vc-shp LINK=const-vc-sh DRY=1                        # inputs hard-link against const-vc-sh
make push-tacc SUITE=const-vc-shp LINK=const-vc-sh
# on Stampede3
export asp3_skx=$WORK/software/aspect/build-3.0/aspect-release
cd $WORK/aspect_work/SlabT_emulator/production-runs_v2/const-vc-shp
# smoke test = ONE run as a normal batch job with a short limit (no idev needed): watch the log, then cancel
SBATCH_TIMELIMIT=00:40:00 sbatch -J shp_100 -o $SCRATCH/aspect_work/logs/shp_100.%j.out -e $SCRATCH/aspect_work/logs/shp_100.%j.err \
    --export=ALL,RUN_ID=100,BASE_DIR=$PWD run_one.spr.slurm
grep -E 'Timestep|Nonlinear|Exception|linear_solver_failed' $SCRATCH/aspect_work/logs/shp_100.*.out | tail -20
# if it steps cleanly, let it run on (it is run_100 itself) and feed the other seven:
grep -v '^100$' pilot-list.txt > pilot-list.rest.txt
SLURM_FILE=run_one.spr.slurm nohup ./submit_from_list.sh pilot-list.rest.txt > submit_shp.log 2>&1 &
BASE_DIR=$SCRATCH/aspect_work/const-vc-shp ../check_runs_list.sh const-vc-shp/pilot-list.txt -p
# local
make pull-tacc SUITE=const-vc-shp ALL=1
src/postproc-numerical-mods/extend_profiles_all-mods.sh const-vc-shp 0:20 "1,20;10,20" 8
```
If the 40 min smoke job is killed by its limit before you decide, a plain resubmit resumes it
(`Resume computation = auto`, own workspace). Set `SBATCH_TIMELIMIT` back (or unset) for the feeder.

## Not in this pilot

mu' as a design dimension (paper 2), temperature-dependent channel viscosity, rate-dependent friction, yield
in the crust's deeper phases (> 150 km; they keep the mantle window and no yield, as in sh).

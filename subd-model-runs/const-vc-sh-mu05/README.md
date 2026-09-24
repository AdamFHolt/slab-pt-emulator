# const-vc-sh-mu05: shear-heating pilot with a low-friction stress cap (mu' = 0.05)

## What it is

The 8 pilot-list runs of const-vc-sh (100 310 195 210 072 217 270 064) with ONE change: the stress
that enters the shear-heating term is capped at a Drucker-Prager law with cohesion 1 MPa and
friction angle asin(0.05) = 0.050021 rad, i.e.

    heating stress = min( 2 eta eps_II ,  1 MPa cos(phi) + P sin(phi) ),  phi = asin(0.05)
    (2 eta eps_II = deviatoric stress invariant; const-vc-sh: cohesion 10 MPa, phi = 30 deg)

so the dissipation follows the thermal-modelling convention tau = mu' rho g z with mu' = 0.05, the
global best fit of Kohn et al. (2018) (0.05 +- 0.015; Ishii & Wallis 2020 give 0.03-0.13 by belt).
The FLOW is untouched -- ASPECT's shear-heating limiter only rescales the stress in the heat term --
so run_XXX still pairs with const-vc (no heating), const-vc-v3ctrl (no heating, 3.0) and const-vc-sh
(viscous-stress heating). Built by `build_runs.const-vc-sh.py --mu05` from the const-vc-sh .prm
files (inputs hard-linked); decided 2026-09-24 after the Elicit literature report on early
subduction thermal evolution flagged interface friction as the least-constrained control.

## What to expect, and what this pilot cannot show

The limiter can only LOWER the heating relative to const-vc-sh. The 6 km crust channel is a fixed
1e20 Pa s in simple shear, so its shear stress (the second invariant of the deviatoric stress,
2 eta eps_II with eps_II = v/2h) is eta v/h: independent of depth and linear in v_conv. The cap
C cos(phi) + P sin(phi) is a yield stress in terms of the mean (full) stress, as in any
Drucker-Prager / Coulomb law -- comparing a deviatoric invariant against a mean-stress-dependent
yield surface is the definition of the criterion, not a units mix (the friction analogue
tau = mu' sigma_n has the same structure). It grows with depth, so it binds above
z* = (tau_visc - 1 MPa) / (0.05 rho g), and below z* the heating is exactly const-vc-sh's. Per pilot
run (channel stress eta v/h; an earlier draft had 2 eta v/h, a factor 2 too high):

| run | v_conv (cm/yr) | age_SP | age_OP | dip | eta_UM | tau_visc (MPa) | z* (km) |
|---|---|---|---|---|---|---|---|
| 100 | 3.7 | 44 | 89 | 67 | 5.6e+20 | 19 | 11 |
| 310 | 8.0 | 45 | 51 | 44 | 9.9e+20 | 42 | 25 |
| 195 | 8.0 | 91 | 77 | 73 | 3.3e+20 | 42 | 25 |
| 210 | 7.9 | 100 | 21 | 31 | 2.2e+20 | 42 | 25 |
| 072 | 7.7 | 70 | 21 | 52 | 1.0e+21 | 41 | 25 |
| 217 | 7.2 | 84 | 43 | 47 | 9.6e+20 | 38 | 23 |
| 270 | 3.1 | 90 | 47 | 52 | 4.0e+20 | 16 | 10 |
| 064 | 1.0 | 65 | 33 | 28 | 5.0e+19 | 5 | 3 |

So the cap acts only in the top ~25 km even for the 8 cm/yr runs, and only in the top ~3-12 km
for the slow ones (064, 270, 100); where it binds it also replaces the v^2 scaling of the
fixed-channel heating by a linear one.
Compare all three heating states -- const-vc, const-vc-sh, const-vc-sh-mu05 -- at 1/5/10 Myr on the
slab-top T(z) and the `heating` field; the mu05 - sh difference is the shallow over-heating of the
fixed-viscosity channel, the mu05 - const-vc difference is the friction-capped heating signal.

What it cannot do: RAISE the deep heating to the frictional level, because the viscous channel
stress (5-42 MPa here) is below mu' rho g z beyond z*. Kohn's 100-500 C model-rock gap at 30-80 km
would need channel stresses up to mu' rho g z there -- a stronger channel limited by yield, which is
a change to the mechanics, not to the heating term. That is the deferred suite below.

## Deferred: the mechanics-consistent "sh + plasticity" suite (not built)

Plan recorded 2026-09-24 for a later suite (working name const-vc-shp):
- Drucker-Prager yield in the `ocrust` composition of the visco-plastic material model (currently
  switched off: cohesion 1e10, 30 deg): cohesion ~1-5 MPa, friction angle asin(mu'), so the channel
  viscosity becomes min(eta_visc, tau_y / 2 eps).
- Raise the crust's viscous ceiling (Maximum viscosity for ocrust, now 1.005e20) to ~1e21-1e22 so the
  channel is yield-limited over the depth range that matters (stress = mu' rho g z until the viscous
  branch takes over), instead of viscosity-limited at ~40 MPa everywhere. Check trench coupling and
  the wedge / decoupling behaviour on the pilot 8 first (cf. dd100 flattening).
- Keep the heating limiter consistent with the material law (same cohesion / friction).
- mu' as a 6th design dimension (log-uniform 0.01-0.15, new LHS, unpaired) is the scientifically
  valuable version -- it is the parameter the rock record wants inverted; fixing mu' = 0.05 keeps the
  400-run pairing. PI to decide when the sh record is in.
- Not planned: temperature-dependent channel viscosity (emergent brittle-ductile transition) or
  rate-dependent friction -- unconstrained parameters and solver risk. Note for the write-up:
  Schmalholz (2026) / Gerya (2022) attribute much of the model-rock gap to exhumation advection,
  which fixed-trench viscous models cannot produce; do not let mu' absorb it.

## How to operate it

```bash
python src/build-numerical-mods/build_runs.const-vc-sh.py --mu05     # (re)build; const-vc-sh untouched
make push-tacc SUITE=const-vc-sh-mu05 LINK=const-vc-sh DRY=1          # inputs hard-link against const-vc-sh
make push-tacc SUITE=const-vc-sh-mu05 LINK=const-vc-sh
# on Stampede3 (same binary, queue and workspace conventions as const-vc-sh)
export asp3_skx=$WORK/software/aspect/build-3.0/aspect-release
cd $WORK/aspect_work/SlabT_emulator/production-runs_v2/const-vc-sh-mu05
SLURM_FILE=run_one.spr.slurm nohup ./submit_from_list.sh pilot-list.txt > submit_mu05.log 2>&1 &
# jobs mu_XXX; own workspace $SCRATCH/aspect_work/const-vc-sh-mu05/; shares the spr 24-job throttle with sh_XXX
BASE_DIR=$SCRATCH/aspect_work/const-vc-sh-mu05 ../check_runs_list.sh const-vc-sh-mu05/pilot-list.txt -p
make pull-tacc SUITE=const-vc-sh-mu05 ALL=1
src/postproc-numerical-mods/extend_profiles_all-mods.sh const-vc-sh-mu05 0:20 "1,20;10,20" 8
```
Smoke-test one .prm with a 48-rank idev session before the feeder, as for every .prm change.

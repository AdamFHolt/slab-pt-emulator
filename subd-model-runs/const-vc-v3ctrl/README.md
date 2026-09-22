# const-vc-v3ctrl: version control for the shear-heating suite

**Why this exists.** const-vc (and ramped-vc, dd100) ran under ASPECT 2.5. const-vc-sh has to run
under ASPECT 3.x (its stress-limited shear heating does not exist in 2.5). So a const-vc-sh vs
const-vc pair mixes two things: the heating, and whatever changed in ASPECT between 2.5 and 3.x.
This set measures the second one alone.

**What it is.** The 8 pilot-list runs of const-vc (`run-inputs/pilot-list.txt`: 100, 310, 195, 210,
072, 217, 270, 064), with const-vc physics, **no heating**, and one line added:
`Stokes solver type = block AMG` (the 2.5 default; 3.0 defaults to GMG). Inputs hard-linked from
const-vc. Written by `src/build-numerical-mods/build_runs.const-vc-sh.py --control`.

**How to use it.** Run these 8 with the same 3.x binary as const-vc-sh, **and on the same queue with
the same rank count** -- `run_one.slurm` (skx, 48) or `run_one.spr.slurm` (spr, 112). The decomposition
is part of what this set controls for: const-vc ran skx/48, so if const-vc-sh goes to spr/112 this set
must too, and its difference from the 2.5 outputs then covers version and decomposition together. Compare slab-top T(z) at
1/5/10 Myr with the existing const-vc (2.5) outputs of the same runs. A few C = version effect is
negligible, pair const-vc-sh against the 2.5 const-vc record. Comparable to the heating signal
(tens of C at the slab top) = either re-run const-vc under 3.x or pair const-vc-sh only against
these 8.

Details: `subd-model-runs/const-vc-sh/README.md` ("The version confound, and the control set").

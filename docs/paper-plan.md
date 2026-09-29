# Paper 1 plan and the channel-rheology spot checks (agreed 2026-09-29)

Living planning note. Decisions and the reasoning behind them; the chronological record is in
`SESSION_NOTES.md`. Update this file when a decision changes, not the session log.

## 1. What paper 1 is

**Central claim.** In the first 10 Myr of subduction the slab-top temperature has a two-layer control
structure -- overriding-plate age above ~40 km, convergence rate below -- and that structure is robust to the
physics choices people argue about. A baseline ensemble (const-vc, 400 runs) and three one-term ablations
say what changes and what does not:

| ablation | suite | what it changes | headline |
|---|---|---|---|
| initiation history | ramped-vc (500) | v(t) ramps to v_conv over t_conv | the only perturbation that changes the SHALLOW layer: age_OP / v_conv crossover moves to 10-20 km (0.5-5 Myr) and 7 + 45 km (0.5-10 Myr) vs 40 / 53 km in const-vc |
| shear heating | const-vc-sh (400, paired) | viscous dissipation in the 1e20 Pa s channel | crossover unchanged (40 / 53-54 km, none in 5-10 Myr); control below it moves from v_conv to eta_UM (ST 0.19 -> 0.48 at 60 km, 0.5-10 Myr) and age_SP; heating +53 / +91 C at 40 / 80 km at 10 Myr, dT ~ v^2.7 at 2 Myr flattening to v^1.3 at 10 Myr (wedge-throttled) |
| decoupling depth | const-vc-dd100 (dip >= 35, paired) | weak crust cut at 100 km | nothing above 80 km (median pair dT 2 C at 40 km); +34 C at 100 km |

The GP emulators (single-depth dTdt 5-100 km in three windows; profile-PCA T(z) at 0.5-10 Myr) are the
method that makes the sensitivity structure computable; their validation is one compact figure, not the point.

**Out of scope for paper 1 (decided 2026-09-29):** the rock P-T comparison. Reasons: it is the most attackable
section (rock P-T uncertainty, per-locality timing, valid-depth caveats), needs the most new work, and serves a
different audience. It becomes paper 2 (emulator-based reachability / inversion: for each rock locality, the
fraction of parameter space whose slab top reaches its P-T in 0.5-10 Myr, with and without heating; mu' as a
design dimension). Rocks stay in the introduction as motivation: a sentence or two and a citation, no comparison.
Note for that section: with the fixed-viscosity channel, heating warms the slab top but barely changes the
fraction of exhumed rocks warmer than the hottest model (48 % -> 45 % at 10 Myr); heating homogenises
(10 Myr band at 40 km 49 C vs 74 C) rather than reaches.

**Figures (5):** (1) ensemble overview + design; (2) compact emulator validation (single-depth + profile-PCA,
all suites); (3) Sobol windows for the baseline; (4) the three ablations against the baseline (Sobol ST vs depth,
three windows -- `plot_sobol_windows.py` with the suite list; the two-suite const-vc / sh version exists at
`plots/science-emulator/summary/sobol_windows_const-vc_const-vc-sh`); (5) heating magnitude and scaling vs
v_conv (`const-vc-sh/sh_pairs`, panel D).

**Status of the ingredients (2026-09-29):** all four suites complete (dd100 314/324, 10 dt-cap resumes pending on
TACC), all extracted, all emulators built and gated (driver `misc/rebuild_emulators_2026-09-29.sh`), Sobol in
three windows per suite, paired-ablation figures, session-level science reading of sh vs const-vc done
(SESSION_NOTES 2026-09-29 14:10-14:45). Estimate: science ~70 % there; 4-8 weeks of writing and figure work to a
submittable draft, assuming no suite beyond the rheology spot checks below. The one science risk is the
channel rheology.

**Caveats to state in the paper:** the 24 const-vc / ramped-vc reruns carry the ASPECT-3.0-style dt-growth cap
(91 %/step) for their whole run, the 5 finished dd100 resumes CFL 0.125 for their last Myr (SESSION_NOTES
2026-09-28/29); const-vc-sh ran under ASPECT 3.0 vs 2.5 for const-vc (v3ctrl: version effect ~1 C); sh dTdt
emulators at >= 60 km are R2 0.80-0.95 (one design-corner run, run_201, behind the 0.5-10 Myr dip at 55-65 km);
the T(z,t) profile emulator is as good for sh as for const-vc.

## 2. The channel-rheology problem and the spot checks

**Why it matters.** const-vc-sh heats with a fixed 1e20 Pa s channel in simple shear: channel stress eta v/h ~ 5-42
MPa, depth-independent, heating ~ eta (v/h)^2. The thermal-modelling literature uses tau = mu' rho g z (mu' ~
0.05, Kohn et al. 2018; 0.03-0.13 by belt, Ishii & Wallis 2020): depth-dependent, linear in v. Everything in the
heating section -- the scaling exponents, the eta_UM redistribution, "the crossover does not move" for sh --
inherits the fixed-channel caveat, and a reviewer from that camp can dismiss it in one sentence.

**Three options (cost, what each can show):**

1. **const-vc-sh-mu05 pilot** -- built 2026-09-24, not pushed, 8 runs (pilot list 100 310 195 210 072 217 270
   064), heating stress capped at cohesion 1 MPa / friction angle asin(0.05); FLOW unchanged, so it pairs with
   const-vc, v3ctrl and sh. Cheap (8 spr jobs, hours). Limited: the cap only LOWERS heating and only binds above
   z* = 3-12 km (slow runs) to ~25 km (8 cm/yr); below z* it is exactly sh. Answers "is the shallow channel
   over-heated?", nothing about 30-80 km. Run because it is free.
2. **const-vc-shp pilot** (mechanics-consistent, the real spot check) -- Drucker-Prager yield in the `ocrust`
   composition (cohesion 1-5 MPa, friction angle asin(mu'), mu' = 0.05), crust Maximum viscosity raised from
   1.005e20 to ~1e21-1e22 so the channel is yield-limited (tau = mu' rho g z) over the depths that matter, heating
   limiter set consistently (same cohesion / friction). Same 8 pilot runs. Cost: 1-2 days of prm work + smoke
   tests (48-rank idev on one prm before the feeder, as for every prm change) + ~1 day of runs. Risk: it changes
   the FLOW -- check trench coupling, wedge / decoupling behaviour and flattening on the 8 (cf. the dd100 lesson)
   before reading any temperature. Tests whether a frictional channel changes the 30-80 km heating magnitude and
   whether eta_UM keeps its new role.
3. **Full const-vc-shp suite** -- 400 paired runs, mu' = 0.05 fixed: ~2 days of spr time (sh took ~1.5 days of
   feeding at 24 jobs), 70 min extraction, ~1 h emulator build (pipeline is suite-generic now). Only if 2 says
   the story changes.

**Decision rule (agreed 2026-09-29):** run 1 then 2 as spot checks; 2 decides 3. If the shp pilot leaves the
30-80 km heating and the Sobol redistribution qualitatively where sh has them, the 8 shp runs are the robustness
paragraph and paper 1 stands on sh. If it moves them, run the full paired shp suite; shp becomes the paper's
heating ablation and sh stays as the viscous end member. mu' stays FIXED for paper 1 (keeps the 400-run
pairing); mu' as a 6th LHS dimension (log-uniform 0.01-0.15, unpaired) is paper 2.

**Pilot comparison figure:** slab-top T(z) at 1 / 5 / 10 Myr for the 8 runs across const-vc / sh / mu05 / shp,
plus channel shear stress vs depth (is the yield branch active where intended?) and the `heating` field.
Do not let mu' absorb the exhumation-advection part of the model-rock gap (Schmalholz 2026, Gerya 2022), which
fixed-trench models cannot produce.

**Not planned:** temperature-dependent channel viscosity (emergent brittle-ductile transition), rate-dependent
friction -- unconstrained and solver risk.

## 3. Next steps, in order

1. Abstract first (30 min): forces the choice of central claim. First sentence = the claim in section 1.
2. Push + submit the mu05 pilot (`make push-tacc SUITE=const-vc-sh-mu05 LINK=const-vc-sh`, then the feeder
   from the suite dir on Stampede3 with `SLURM_FILE=run_one.spr.slurm`; README "How to operate it").
3. Build the shp pilot prms as a `--shp` variant of `build_runs.const-vc-sh.py` (Claude to draft); smoke-test one;
   submit the 8.
4. Paper figures 1-5; the three-ablation Sobol figure is a suite-list change to `plot_sobol_windows.py`.
5. dd100: pull the 10 with `run-inputs/resubmit-list.2026-09-29.txt` only, extract, masters, rebuild its emulator
   with the driver, re-seed its gates.
6. Housekeeping: re-seed the const-vc gates (two marginal misses) if agreed; tidy the notes.
7. Paper 2 list: rock reachability / inversion with the emulators; mu' as a design dimension.

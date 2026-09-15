# const-vc-dd100: const-vc with the weak crust cut off at 100 km

## What it is

One-term ablation of const-vc. The `ocrust` compositional field's first phase transition, below which
the crust takes the background mantle rheology (so the slab can decouple from the wedge no deeper
than this), moves from 150 km to 100 km: a prescribed maximum decoupling depth. Every other line of
every .prm is identical and the design is the same 400-point Latin hypercube, so `run_XXX` pairs with
const-vc `run_XXX`. Built by `src/build-numerical-mods/build_runs.const-vc-dd100.py` from the const-vc
run-inputs (inputs hard-linked); reference template `data/ref-model/model_template_fixed-trench.const-vc-dd100.prm`.
PI decision 2026-09-10 (fixed 100 km; shallower cutoffs make the plates merge). Not an NSF-proposal suite.

## Decisions (2026-09-15)

1. **The 100 km cutoff does not weld the slab to the overriding plate.** Pre-run check on const-vc
   (`analysis/pre-run-wedge-check/`) and a 19-run pilot compared with const-vc twins at 1/2/5 Myr
   (`analysis/pilot-check/`): OP interior static in every run; cold nose truncated at 100 km with hot
   corner flow beneath (1180-1290 C vs 780-930 C in const-vc, 10 km off the slab top).
2. **Low-dip slabs flatten along the cutoff.** For initial dip 25-26 deg the 80-100 km slab-top dip
   drops to 8-17 deg by 5 Myr (const-vc twins: 18-23); nothing at dip >= 38. The cutoff level acts as
   a preferred sliding surface (const-vc does the same at 150 km). Not what the ablation is meant to
   test, so:
3. **The suite is run for dip_int >= 35 deg only** (320 of the 400 design points), by filtering the
   existing design rather than redrawing it. Pairing with const-vc is kept and the design stays
   uniform (KS p >= 0.98, |corr| <= 0.09, all pairwise 4x4 cells filled). age_OP is not restricted
   (the worst flattening was run 090 with age_OP 58; old-OP steep runs are clean).
   `src/build-numerical-mods/make_submit_list.const-vc-dd100.py` writes `run-inputs/full-list.dip35.txt`.
   Downstream, define dd100 membership by `dip_int >= 35` on the parameter table and apply the same
   filter to const-vc for paired comparisons.

## What was submitted (Stampede3, 2026-09-15)

| list | runs | notes |
|---|---|---|
| `run-inputs/pilot-list.txt` | 19 | 16 at-risk (old OP / stiff / low dip) + 3 young-OP controls; jobs 3501029-3501049 |
| `run-inputs/full-list.dip35.txt` | 305 | the 320 with dip >= 35 minus the 15 pilot runs that pass; 35-40 deg band (39 runs) first |
| **dd100 record** | **320** | pilot (dip >= 35) + full list |
| excluded | 80 | dip < 35; four of them (090, 038, 039, 343) ran in the pilot and are kept as evidence of the flattening |

All 400 run directories stay on TACC (cheap: .prm + hard-linked inputs); only the 320 are run.

## How to operate it

See `docs/tacc-runbook.md` for the full procedure. Short version, on Stampede3:

```bash
cd $WORK/aspect_work/SlabT_emulator/production-runs_v2/const-vc-dd100
nohup ./submit_from_list.sh full-list.dip35.txt > submit_full-list.dip35.log 2>&1 &   # feeder, survives logout
cd ..; ./check_runs_list.sh const-vc-dd100/pilot-list.txt; ./check_runs_list.sh const-vc-dd100/full-list.dip35.txt
```

Locally: `make push-tacc` (inputs), rsync outputs to `run-outputs/` (runbook section 4), field CSVs with
`analysis/pilot-check/extract_dd100.py`, paired diagnostics with `analysis/pilot-check/pair_diag.py`
and sections with `pairlook.py RUNS STEP OUT.png [T|eta|v]`.

## Next

- When the 35-40 deg band reaches ~5 Myr: rsync, run `pair_diag.py` and the local-dip table on those 39
  runs. If any flatten, raise the threshold post hoc (drop runs; nothing to resubmit).
- When the pilot reaches step 20: rerun `extract_dd100.py RUN 20` + `pair_diag.py` (STEPS) for the
  10 Myr check, especially 377 (dip 38) and 316 (dip 41); do the step 15 low-dip comparison.
- Then the science: slab-top T(z) above the cutoff, dd100 vs const-vc, on the 320 pairs; extend the
  processed-record tooling (`extract_profiles_range.py`) to the suite.

Chronology and numbers: `SESSION_NOTES.md`, entries dated 2026-09-15.

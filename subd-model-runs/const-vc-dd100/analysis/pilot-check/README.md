# const-vc-dd100 pilot check (2026-09-15, steps 2/4/10 = 1/2/5 Myr)

Does the 100 km crust cutoff weld the slab to the overriding plate?  Paired comparison of the 19
pilot runs against their const-vc twins at the same output step.

- `extract_dd100.py RUN STEP...` (pvpython) -> `../run_XXX/t{k}.csv` field CSVs from `../../run-outputs/`.
- `pair_diag.py [outdir]` -> `pair_diag.csv` + printed tables: slab-top x and T at 50-250 km,
  OP-side T / log-viscosity / |v| 10 and 20 km off the slab top, op-layer entrainment depth,
  OP-interior velocity (op >= 0.5, 30-90 km depth, 20-100 km behind the slab top), op thickness.
- `pairlook.py RUNS STEP OUT.png [T|eta|v]` -> const-vc (left) vs dd100 (right) sections.

Result: no welding at 5 Myr in any pilot run.  See SESSION_NOTES.md (2026-09-15 pilot entry).
Rerun `extract_dd100.py ... 20` and `pair_diag.py` (STEPS) once the runs reach 10 Myr.

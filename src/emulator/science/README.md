# src/emulator/science

Cross-cutting emulator science figures (single-depth and profile-PCA products together). The
per-product science scripts stay in `src/emulator/single_depth/science/` and `src/emulator/profile_pca/`.

- `plot_emulator_validation_sobol.py SUITE [--pca-set profile_pca_10myr|profile_pca]` -- the NSF
  proposal's Figure 3 generalised to a suite: (A) held-out emulated vs modelled slab-top T at
  5 Myr, (B) held-out profile RMSE vs depth at 0.5/3/5/10 Myr, (C) Sobol S_T vs depth for the
  0.5-5 and 5-10 Myr dTdt windows. Times / windows whose products do not exist are skipped
  (ramped-vc currently has no 10 Myr profile-PCA model and no `sobol_dt10-20/`). Output
  `plots/science-emulator/summary/<suite>/emulator_validation_sobol_<suite>.{pdf,svg,png}`;
  `make emulator-summary-plot SUITE=...`. Copied 2026-09-24 from
  `nsf-slab-pt/figures/scripts/fig03_emulator_sobol.py` (2026-09-07).

## Multi-suite summary figures (2026-09-24)

Shared style/loaders: `emu_style.py` (Myriad Pro house style; suite -> line style, time -> plasma,
parameter -> Okabe-Ito, window -> three fixed colours; readers for the profile-PCA quality reports and
the Sobol JSONs). Both scripts take a list of suites (default `const-vc ramped-vc`) and write to
`plots/science-emulator/summary/`; `make emulator-figures [SUITES="..."]` runs both.

- `plot_emulator_validation.py` -> `emulator_validation.{pdf,svg,png}`: per suite (rows) held-out
  emulated-vs-modelled slab-top T at 5 Myr with pooled and p95 per-run RMSE, and held-out profile RMSE
  vs depth at 0.5/3/5/10 Myr over the PCA-truncation floor; right column, all suites: pooled RMSE vs
  time for every 0.5-10 Myr slice (with p95 and PCA floor) and held-out R^2 of the single-depth dTdt
  emulators vs depth for the three cooling windows.
- `plot_sobol_windows.py` -> `sobol_windows.{pdf,svg,png}`: Sobol S_T vs depth (5-100 km) per window
  (0.5-5, 0.5-10, 5-10 Myr) with bootstrap CI bands, suites overlaid, t_ramp included for ramped-vc,
  80 km deep-end validity line; bottom row the emulators' held-out R^2 per depth. Prints the
  age_OP/v_conv crossover depths.
- `plot_emulator_validation_sobol.py` stays as the proposal-style single-suite 3-panel view.

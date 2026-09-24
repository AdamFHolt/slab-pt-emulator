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

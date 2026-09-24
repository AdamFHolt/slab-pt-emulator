# src/science-numerical-mods

Science (not QC) figures built directly from the processed slab-top record
`subd-model-runs/<suite>/analysis/run_XXX/Tprof_k.csv`, i.e. from the numerical models rather
than the emulator. Output goes to `plots/science-numerical-mods/<suite>/`.

- `explore_all_models_rocks.py SUITE [--color-by P] [--zmax 80] [--dip-min 35]` -- every run's
  slab-top T(z) at 0.5, 3 and 10 Myr coloured by a design parameter, plus median and 5-95 %
  envelope at seven times, all against the Agard et al. (2018) peak P-T compilation
  (`data/rocks/`). Prints the fraction of rock points inside / warmer than the model range per
  time. `make rocks-plot SUITE=... [COLOR_BY=...] [DIP_MIN=...]`. Copied 2026-09-24 from the NSF
  proposal folder (`nsf-slab-pt/figures/scripts/explore_all_models_rocks.py`, 2026-09-10) and
  generalised to any suite. For const-vc-dd100 vs const-vc use `--dip-min 35` on const-vc, since
  dd100 was only run for dip >= 35.

- `plot_suite_summary.py [SUITE ...] [--refresh]` -- six-panel science summary of the numerical models
  (default const-vc + ramped-vc, house style via `src/emulator/science/emu_style.py`): slab-top T(z)
  median + 5-95 % envelope at 0.5/2/5/10 Myr per suite with the Agard rocks; T at 40 and 80 km vs
  time (median, IQR); t90 = time to 90 % of the 0.5-10 Myr cooling vs depth; mean cooling rate vs
  depth for the 0.5-5 and 5-10 Myr windows; T(40 km, 5 Myr) vs v_conv coloured by age_OP. Uses only
  runs with the full 20-step record (const-vc 384, ramped-vc 491); caches the loaded record in
  `<suite>/analysis/slabtop_record_0-80km.npz`. `make numerical-summary-plot`. Output
  `plots/science-numerical-mods/suite_summary.{pdf,svg,png}`.

The emulator-side counterpart (held-out accuracy + Sobol, the proposal's Fig. 3) is
`src/emulator/science/plot_emulator_validation_sobol.py` -> `plots/science-emulator/summary/<suite>/`.

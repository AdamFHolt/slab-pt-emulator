# Pre-run check for const-vc-dd100 (2026-09-15)

Question: with the weak crust cut off at 100 km instead of 150 km, does the slab weld to the
overriding plate (OP)?  Answered as far as possible from the paired const-vc outputs.

- `wedge_nose_100km.py` — for every const-vc run at t = 0, 1, 5, 10 Myr (steps 0/2/10/20), grids the
  wedge (x 1700–2600 km, 0–200 km depth, 2 km), picks the slab top (ocrust >= 0.5, rightmost), and
  samples T / log10(viscosity) / |v| / op-field at 10, 20, 40 km on the OP side of the slab top at
  80–140 km depth, plus the depth of the 1000–1300 C isotherms next to the slab and far-field.
  Run with: `env/bin/python wedge_nose_100km.py <outdir>`  (~15 min on 32 cores).  -> `wedge_nose_100km.csv`
- `summarize.py` — suite-wide statistics, ranking, pilot-list check -> `wedge_nose_summary.png`
- `below_cutoff.py` — the coupled shear zone just below the existing 150 km cutoff (150–190 km) for
  the 12 coldest-nose runs + 3 controls at 10 Myr, and how deep the op field is entrained.
- `quicklook.py RUNS STEP OUT.png` — T / viscosity sections of the wedge nose with the slab top,
  op-field outline and 1100/1200/1300 C isotherms.  `quicklook_t20.png` (010, 135, 286),
  `quicklook_lowdip_t20.png` (090, 038).

Findings are summarised in SESSION_NOTES.md (2026-09-15 entry).

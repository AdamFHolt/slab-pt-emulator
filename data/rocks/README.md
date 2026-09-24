# data/rocks

Hand-supplied natural-rock P-T data used as overlays on model figures. Model-derived data never
lives here.

- `Agard_2018.xlsx` -- peak P-T conditions of exhumed oceanic HP/LT rocks, credited Agard et al.
  (2018) in the PI's RSMAS talk (slide 9). Supplied by the PI 2026-09-05 for the NSF proposal
  figures; file itself dated 2019. One sheet, 130 rows, columns T (degC), P (GPa), t, with the
  terrane name in column A on the first row of each block only (22 blocks: Japan, Franciscan x4,
  Cascades, Alaska, New Zealand, New Caledonia, Ecuador, Chile, Central America x4, Alps x2,
  Medit./Mid-East x3, Asia, metamorphic soles).
- `Agard_2018.csv` -- the same table with the terrane name forward-filled, columns
  `terrane,T_C,P_GPa,t`. Scripts read this one (no openpyxl needed). Regenerate with
  `pd.read_excel(xlsx, names=["terrane","T_C","P_GPa","t"]).assign(terrane=lambda d: d.terrane.ffill())`.
- P is converted to depth with rho = 3000 kg/m3 (z_km = P_GPa*1e9/(3000*9.81)/1e3); 7 of the 130
  points are deeper than 80 km.
- **Open:** the meaning of `t` (0.02-1) is unconfirmed (a normalised time?); it is carried through
  and not plotted.

Copied from `~/Documents/InProgress/nsf-slab-pt/figures/data/` on 2026-09-24. Used by
`src/science-numerical-mods/explore_all_models_rocks.py`.

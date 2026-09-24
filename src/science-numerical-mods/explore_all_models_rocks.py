"""Slab-top T(z) of ALL runs of one suite against the exhumed-rock P-T record.

Panels 1-3: every run's slab-top profile at 0.5, 3 and 10 Myr after initiation,
coloured by one design parameter. Panel 4: median and 5-95 % envelope at seven
times (0.5, 1, 2, 3, 5, 7.5, 10 Myr). Every panel carries the Agard et al. (2018)
peak P-T compilation (data/rocks/Agard_2018.csv, P -> depth with rho = 3000
kg/m3). Depth increases upward (P-T convention). Prints, per time, the fraction
of rock points inside the model range and warmer than the hottest model.

Copied into the repo 2026-09-24 from the NSF proposal folder
(nsf-slab-pt/figures/scripts/explore_all_models_rocks.py, written 2026-09-10),
generalised to any suite and to the repo's own paths.

Usage (from the repo root, repo env):
  env/bin/python src/science-numerical-mods/explore_all_models_rocks.py SUITE \
      [--color-by v_conv|age_SP|age_OP|dip_int|eta_UM] [--zmax 80] [--dip-min 35] [--out PNG]

  SUITE      const-vc | ramped-vc | const-vc-dd100 | const-vc-sh (anything with
             subd-model-runs/<suite>/analysis/run_XXX/Tprof_k.csv and
             data/params/params-list.<suite>.csv)
  --zmax     depth extent of the plot / interpolation grid (default 80 km; the
             record reaches 100 km since 2026-09-17 but 85-100 km is only valid
             after crust arrival, see const-vc-dd100/README "deep end")
  --dip-min  keep only runs with dip_int >= value; const-vc-dd100 is run for dip >= 35 only, so
             `const-vc --dip-min 35` is its paired reference (suffix _dip35 on the output)
  default output: plots/science-numerical-mods/<suite>/explore_all_models_rocks_<suite>_<color_by>[_dipNN].png

Runs whose profile at a given time does not reach zmax - 1 km are dropped for
that time (n is printed in each panel title).
"""
import argparse, glob, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("suite", nargs="?", default="const-vc")
ap.add_argument("--color-by", default="age_OP", choices=["v_conv", "age_SP", "age_OP", "dip_int", "eta_UM"])
ap.add_argument("--zmax", type=float, default=80.0)
ap.add_argument("--out", default=None)
ap.add_argument("--dpi", type=int, default=130)
ap.add_argument("--dip-min", type=float, default=None,
                help="keep only runs with dip_int >= this (e.g. 35 to pair const-vc with const-vc-dd100)")
a = ap.parse_args()
SUITE, COLOR_BY, ZMAX = a.suite, a.color_by, a.zmax

SFX = f"_dip{a.dip_min:g}" if a.dip_min is not None else ""
OUT = a.out or os.path.join(ROOT, "plots", "science-numerical-mods", SUITE,
                            f"explore_all_models_rocks_{SUITE}_{COLOR_BY}{SFX}.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
AN = os.path.join(ROOT, "subd-model-runs", SUITE, "analysis")
_par_base = os.path.join(ROOT, "data", "params", f"params-list.{SUITE}")
if os.path.exists(_par_base + ".csv"):
    PAR = pd.read_csv(_par_base + ".csv")
else:   # the .csv is gitignored; the tracked .npy has the same columns (ramped-vc adds t_conv)
    _arr = np.load(_par_base + ".npy")
    _cols = {5: ["v_conv", "age_SP", "age_OP", "dip_int", "eta_UM"],
             6: ["v_conv", "t_conv", "age_SP", "age_OP", "dip_int", "eta_UM"]}[_arr.shape[1]]
    PAR = pd.DataFrame(_arr, columns=_cols)
ROCK_CSV = os.path.join(ROOT, "data", "rocks", "Agard_2018.csv")

zgrid = np.arange(0, ZMAX + 0.01, 1.0)
K = {1: "0.5 Myr", 6: "3 Myr", 20: "10 Myr"}
K_ALL = [1, 2, 4, 6, 10, 15, 20]

runs = sorted(glob.glob(os.path.join(AN, "run_[0-9][0-9][0-9]")))
prof = {k: {} for k in K_ALL}
for rd in runs:
    rid = int(os.path.basename(rd).split("_")[1])
    if rid >= len(PAR):        # benchmark / one-off runs (>= 900) have no design row
        continue
    if a.dip_min is not None and PAR.dip_int[rid] < a.dip_min:
        continue
    for k in K_ALL:
        f = os.path.join(rd, f"Tprof_{k}.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f).dropna(subset=["depth_km", "T_C"]).sort_values("depth_km")
        d = d.drop_duplicates("depth_km")
        if d.depth_km.max() < ZMAX - 1:
            continue
        prof[k][rid] = np.interp(zgrid, d.depth_km, d.T_C)
print(SUITE, {f"{0.5*k:g} Myr": len(v) for k, v in prof.items()})
if not prof[20]:
    raise SystemExit(f"no 10 Myr profiles found under {AN}")

rock = pd.read_csv(ROCK_CSV).dropna(subset=["T_C", "P_GPa"])
rock_z = rock.P_GPa * 1e9 / (3000 * 9.81) / 1e3

vals = PAR[COLOR_BY]
norm = LogNorm(vals.min(), vals.max()) if COLOR_BY == "eta_UM" else Normalize(vals.min(), vals.max())
cmap = plt.get_cmap("viridis")

fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)
for ax, (k, lab) in zip(axes[:3], K.items()):
    order = np.argsort(vals.values)  # draw low values first
    for rid in order:
        if rid not in prof[k]:
            continue
        ax.plot(prof[k][rid], zgrid, color=cmap(norm(vals[rid])), lw=0.5, alpha=0.35)
    ax.scatter(rock.T_C, rock_z, s=10, color="0.25", zorder=5, label="Agard et al. 2018 peak P-T")
    ax.set_title(f"{lab} after initiation  (n = {len(prof[k])})", fontsize=10)
    ax.set_xlabel("Slab-top temperature (°C)")
    ax.set_xlim(0, 1300); ax.set_ylim(0, ZMAX); ax.grid(alpha=0.25)
axes[0].set_ylabel("Depth (km)")
axes[0].legend(loc="lower right", fontsize=8, frameon=False)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cax = axes[2].inset_axes([0.76, 0.08, 0.03, 0.40])
cb = fig.colorbar(sm, cax=cax)
cb.ax.tick_params(labelsize=7)
cb.set_label({"age_OP": "age$_{OP}$ (Myr)", "age_SP": "age$_{SP}$ (Myr)", "v_conv": "v$_{conv}$ (cm/yr)",
              "dip_int": "dip (°)", "eta_UM": "η$_{UM}$ (Pa s)"}[COLOR_BY], fontsize=8)

# panel 4: median + 5-95% envelope through time
ax = axes[3]
plasma = plt.get_cmap("plasma")
for k, s in zip(K_ALL, np.linspace(0.08, 0.82, len(K_ALL))):
    if not prof[k]:
        continue
    A = np.array(list(prof[k].values()))
    lo, med, hi = np.percentile(A, [5, 50, 95], axis=0)
    ax.fill_betweenx(zgrid, lo, hi, color=plasma(s), alpha=0.12, lw=0)
    ax.plot(med, zgrid, color=plasma(s), lw=1.6, label=f"{0.5*k:g}")
ax.scatter(rock.T_C, rock_z, s=10, color="0.25", zorder=5)
ax.set_title(f"Median and 5–95 % envelope, all {SUITE} runs" + (f" (dip ≥ {a.dip_min:g}°)" if a.dip_min is not None else ""), fontsize=10)
ax.set_xlabel("Slab-top temperature (°C)"); ax.set_xlim(0, 1300); ax.grid(alpha=0.25)
ax.legend(title="Myr since initiation", fontsize=8, title_fontsize=8, loc="lower right", ncol=2, frameon=False)

# fraction of rocks inside / warmer than the model range at their depth, printed
inz = rock_z <= ZMAX
for k in K_ALL:
    if not prof[k]:
        continue
    A = np.array(list(prof[k].values()))
    hot = np.interp(rock_z, zgrid, A.max(axis=0)); cold = np.interp(rock_z, zgrid, A.min(axis=0))
    inside = ((rock.T_C <= hot) & (rock.T_C >= cold))[inz].mean()
    warmer = (rock.T_C > hot)[inz].mean()
    print(f"{0.5*k:4g} Myr: rocks (z <= {ZMAX:g} km, n={inz.sum()}) inside model range {inside:5.1%}, "
          f"warmer than hottest model {warmer:5.1%}")

fig.suptitle(f"{SUITE}{SFX.replace('_dip', ', dip >= ')}: slab-top T(z) of all runs vs. exhumed-rock peak P-T", fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(OUT, dpi=a.dpi, bbox_inches="tight")
print("wrote", OUT)

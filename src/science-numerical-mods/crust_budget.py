#!/usr/bin/env python3
"""Oceanic-crust budget for a run, across suites: where is the crust lost -- on the incoming plate or in the channel?

Written 2026-11-03 to test the 2026-10-05 diagnosis of the const-vc-shp pilot (incoming-plate crust at the
viscosity floor is dragged to the trench, thins below resolution and is lost) against the alternative (the
crust thins inside the channel itself).

Reads analysis/run_XXX/t{k}.csv (extract_field_csvs.py point data, output every 0.5 Myr), grids
x 600-2400 km, z 0-400 km at --dx, and measures per step:
  - crust_int   : integral of ocrust (clipped 0..1) over z < 400 km (km2) -- what the field still carries
  - crust_raw   : same, unclipped (clipping hides over/undershoot)
  - crust_a05   : area with ocrust > 0.5 (km2)                              -- what a threshold sees
  - incoming plate, at d = 50 / 200 / 500 km before the trench:
      h_in_d    : column integral of ocrust over z < 20 km (km)
      dv_in_d   : crust-weighted vx minus vx at 20 km depth (cm/yr, > 0 = crust outruns its plate)
  - Q_in        : crust flux through x = trench - 50 km, integral(ocrust vx dz) (km2/Myr)
  - channel at z = 20 / 50 / 100 km (slab side, x > trench - 10):
      h_ch_z    : horizontal crust width x sin(dip) = normal thickness (km)
      Q_ch_z    : downward crust flux through the horizontal line, integral(ocrust (-vy) dx) (km2/Myr)
Steady transport with no loss: Q_in ~ Q_ch_20 ~ Q_ch_50 ~ Q_ch_100 (slab-tip effects aside).

Usage:
  crust_budget.py --run 072 [--suites const-vc-sh const-vc-shp] [--steps 0 1 2 3 4 6 8 12 16 20]
Output: plots/science-numerical-mods/<last suite>/crust_budget_run_XXX.{csv,png,pdf}
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "emulator", "science"))
import emu_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="072")
ap.add_argument("--suites", nargs="+", default=["const-vc-sh", "const-vc-shp"])
ap.add_argument("--steps", nargs="+", type=int, default=[0, 1, 2, 3, 4, 6, 8, 12, 16, 20])
ap.add_argument("--dx", type=float, default=0.5)
args = ap.parse_args()
S.apply_style()

RUN = f"run_{int(args.run):03d}"
YMAX, X0, X1, ZMAX = 1000.0, 600.0, 2400.0, 400.0
KM2_PER_MYR = 1e3          # (km) x (m/yr) -> km2/Myr
OUT = os.path.join(ROOT, "plots", "science-numerical-mods", args.suites[-1])
os.makedirs(OUT, exist_ok=True)
DIST = (50, 200, 500)
ZCH = (20, 50, 100)


def load(suite, k):
    f = os.path.join(ROOT, "subd-model-runs", suite, "analysis", RUN, f"t{k}.csv")
    d = pd.read_csv(f, usecols=["velocity:0", "velocity:1", "ocrust", "Points:0", "Points:1"])
    d["x"] = d["Points:0"] / 1e3
    d["z"] = YMAX - d["Points:1"] / 1e3
    d = d[(d.x > X0 - 5) & (d.x < X1 + 5) & (d.z < ZMAX + 5)]
    return d.groupby([d.x.round(4), d.z.round(4)], sort=False).mean().reset_index(drop=True)


def grid(d):
    xs = np.arange(X0, X1 + args.dx / 2, args.dx)
    zs = np.arange(0, ZMAX + args.dx / 2, args.dx)
    X, Z = np.meshgrid(xs, zs)
    pts = (d.x.values, d.z.values)
    g = {c: griddata(pts, d[c].values, (X, Z), method="linear") for c in ["ocrust", "velocity:0", "velocity:1"]}
    g["c"] = np.clip(np.nan_to_num(g["ocrust"]), 0, 1)
    return xs, zs, g


def trench_x(xs, zs, c):
    """Leftmost x where the crust reaches 10 km depth (the slab-top hinge just behind the trench)."""
    i = np.argmin(np.abs(zs - 10))
    j = np.where(c[i] > 0.3)[0]
    return float(xs[j.min()]) if len(j) else np.nan


def measure(suite, k):
    xs, zs, g = grid(load(suite, k))
    c, vx, vy = g["c"], np.nan_to_num(g["velocity:0"]), np.nan_to_num(g["velocity:1"])
    dA = args.dx ** 2
    r = {"suite": suite, "step": k, "t_myr": 0.5 * k,
         "crust_int": c.sum() * dA, "crust_raw": np.nansum(g["ocrust"]) * dA, "crust_a05": (c > 0.5).sum() * dA}
    xt = trench_x(xs, zs, c)
    r["x_trench"] = xt
    shallow = zs < 20
    i20 = np.argmin(np.abs(zs - 20))
    for dd in DIST:
        j = np.argmin(np.abs(xs - (xt - dd)))
        col = c[shallow, j]
        h = col.sum() * args.dx
        r[f"h_in_{dd}"] = h
        r[f"dv_in_{dd}"] = ((col * vx[shallow, j]).sum() / col.sum() - vx[i20, j]) * 100 if h > 0.05 else np.nan
        if dd == 50:
            r["Q_in"] = (col * vx[shallow, j]).sum() * args.dx * KM2_PER_MYR
            r["v_plate"] = vx[i20, j] * 100
    for zc in ZCH:
        i = np.argmin(np.abs(zs - zc))
        sel = xs > xt - 10
        row = c[i, sel]
        w = row.sum() * args.dx
        # dip from the crust centroid 5 km above / below
        cen = []
        for dz in (-5, 5):
            ii = np.argmin(np.abs(zs - (zc + dz)))
            rr = c[ii, sel]
            cen.append((rr * xs[sel]).sum() / rr.sum() if rr.sum() > 0 else np.nan)
        dip = np.degrees(np.arctan2(10.0, cen[1] - cen[0])) if np.all(np.isfinite(cen)) else np.nan
        r[f"dip_{zc}"] = dip
        r[f"h_ch_{zc}"] = w * np.sin(np.radians(dip)) if np.isfinite(dip) else np.nan
        r[f"Q_ch_{zc}"] = (row * -vy[i, sel]).sum() * args.dx * KM2_PER_MYR
    # thickness profile of the incoming crust vs distance before the trench
    dprof = np.arange(10, 1001, 10)
    prof = []
    for dd in dprof:
        j = np.argmin(np.abs(xs - (xt - dd)))
        prof.append(c[shallow, j].sum() * args.dx)
    r["_prof"] = np.array(prof)
    r["_dprof"] = dprof
    print(f"{suite} {RUN} t={0.5 * k:4.1f}  int {r['crust_int']:7.0f} raw {r['crust_raw']:7.0f}  a05 {r['crust_a05']:7.0f}  "
          f"h_in50/200/500 {r['h_in_50']:.2f}/{r['h_in_200']:.2f}/{r['h_in_500']:.2f}  "
          f"dv50 {r['dv_in_50']:+.1f}  Q_in {r['Q_in']:.0f}  "
          f"h_ch {r['h_ch_20']:.2f}/{r['h_ch_50']:.2f}/{r['h_ch_100']:.2f}  "
          f"Q_ch {r['Q_ch_20']:.0f}/{r['Q_ch_50']:.0f}/{r['Q_ch_100']:.0f}", flush=True)
    return r


rows = [measure(su, k) for su in args.suites for k in args.steps]
df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
df.to_csv(os.path.join(OUT, f"crust_budget_{RUN}.csv"), index=False, float_format="%.4g")

# ------------------------------------------------------------------ figure
fig, axs = plt.subplots(2, 3, figsize=(14, 7.5))
ls = {su: s for su, s in zip(args.suites, ["--", "-", ":", "-.", (0, (5, 1)), (0, (1, 3)), (0, (3, 1, 1, 1, 1, 1))])}
cols = {"int": "k", "a05": "0.5"}
ax = axs[0, 0]
for su in args.suites:
    d = df[df.suite == su]
    ax.plot(d.t_myr, d.crust_int, ls[su], color="k", label=f"{su} integral")
    ax.plot(d.t_myr, d.crust_a05, ls[su], color="C3", label=f"{su} area > 0.5")
ax.set(xlabel="t (Myr)", ylabel="crust, z < 400 km (km$^2$)", title="(a) crust carried vs crust seen")
ax.legend(fontsize=7)

ax = axs[0, 1]
cm = plt.get_cmap("viridis")
for su in args.suites:
    for r in [r for r in rows if r["suite"] == su]:
        ax.plot(r["_dprof"], r["_prof"], ls[su], color=cm(r["t_myr"] / 10), lw=1)
ax.set(xlabel="distance before trench (km)", ylabel="incoming crust thickness (km)",
       title="(b) incoming crust; colour = t (0-10 Myr)")
ax.invert_xaxis()

ax = axs[0, 2]
for su in args.suites:
    d = df[df.suite == su]
    for dd, cc in zip(DIST, ["C0", "C1", "C2"]):
        ax.plot(d.t_myr, d[f"dv_in_{dd}"], ls[su], color=cc, label=f"{su} {dd} km" if su == args.suites[-1] else None)
ax.axhline(0, color="0.6", lw=0.5)
ax.set(xlabel="t (Myr)", ylabel="v_crust - v_plate(20 km) (cm/yr)", title="(c) crust outrunning its plate")
ax.legend(fontsize=7)

ax = axs[1, 0]
for su in args.suites:
    d = df[df.suite == su]
    ax.plot(d.t_myr, d.h_in_50, ls[su], color="k", label=f"{su} incoming, 50 km" if True else None)
    for zc, cc in zip(ZCH, ["C0", "C1", "C2"]):
        ax.plot(d.t_myr, d[f"h_ch_{zc}"], ls[su], color=cc, label=f"{su} channel {zc} km")
ax.set(xlabel="t (Myr)", ylabel="crust thickness (km)", title="(d) incoming vs channel thickness")
ax.legend(fontsize=6, ncol=2)

ax = axs[1, 1]
for su in args.suites:
    d = df[df.suite == su]
    ax.plot(d.t_myr, d.Q_in, ls[su], color="k", label=f"{su} into trench")
    for zc, cc in zip(ZCH, ["C0", "C1", "C2"]):
        ax.plot(d.t_myr, d[f"Q_ch_{zc}"], ls[su], color=cc, label=f"{su} down past {zc} km")
ax.set(xlabel="t (Myr)", ylabel="crust flux (km$^2$/Myr)", title="(e) crust flux")
ax.legend(fontsize=6, ncol=2)

ax = axs[1, 2]
for su in args.suites:
    d = df[df.suite == su]
    ax.plot(d.t_myr, d.v_plate, ls[su], color="k", label=f"{su} plate at 20 km, trench - 50")
ax.set(xlabel="t (Myr)", ylabel="cm/yr", title="(f) incoming plate speed")
ax.legend(fontsize=7)
fig.suptitle(f"{RUN}: oceanic-crust budget ({' vs '.join(args.suites)})")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"crust_budget_{RUN}.{ext}"), dpi=150)
print("wrote", os.path.join(OUT, f"crust_budget_{RUN}.png"))

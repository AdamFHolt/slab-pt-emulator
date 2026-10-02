#!/usr/bin/env python3
"""Channel sanity plots for the const-vc-shp pilot (brittle-ductile crust channel) vs its sh parent.

Reads analysis/run_XXX/t{k}.csv (point data written by extract_field_csvs.py) for one run and one output
step in each suite, grids a window around the wedge (0.5 km), and derives the strain rate from the
gridded velocity (eps_II = sqrt(0.5 e_ij e_ij), deviatoric) and the stress tau_II = 2 eta eps_II.

Figure 1 (<run>_t<k>_fields): rows = suites, cols = log10 viscosity, log10 strain rate, tau_II,
  log10 shear heating; ocrust 0.5 contour (white) and isotherms every 200 C.
Figure 2 (<run>_t<k>_channel): channel (ocrust > 0.5, below the incoming-plate crust) binned by depth --
  median eta, eps_II, tau_II, T; for shp the predicted DP yield (C cos phi + max(p,0) sin phi, binned p)
  and the wet-quartzite creep stress at the binned T and eps_II, so the frictional / creeping / clamped
  branches can be read off; plus the slab geometry of all suites (ocrust 0.5 contours, steps 1..k).

Usage:
  plot_channel_fields.py --run 100 --step 5 [--suites const-vc-sh const-vc-shp] [--xlim 1800 2100] [--zmax 160]
Output: plots/science-numerical-mods/<last suite>/run_XXX_t<k>_{fields,channel}.{pdf,png}
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
ap.add_argument("--run", default="100")
ap.add_argument("--step", type=int, default=5)
ap.add_argument("--suites", nargs="+", default=["const-vc-sh", "const-vc-shp"])
ap.add_argument("--xlim", nargs=2, type=float, default=None, help="km; default: trench - 90 .. trench + 210")
ap.add_argument("--zmax", type=float, default=160.0)
ap.add_argument("--dx", type=float, default=0.5)
args = ap.parse_args()
S.apply_style()

RUN = f"run_{int(args.run):03d}"
YMAX = 1000.0                      # km, Y extent
R, ADIA = 8.314, 9.24e-9
C_DP, PHI = 1e6, np.deg2rad(2.866)
QTZ_A, QTZ_N, QTZ_E = 4.917826e-32, 4.0, 135e3
OUT = os.path.join(ROOT, "plots", "science-numerical-mods", args.suites[-1])
os.makedirs(OUT, exist_ok=True)


def load(suite, k):
    f = os.path.join(ROOT, "subd-model-runs", suite, "analysis", RUN, f"t{k}.csv")
    if not os.path.exists(f):
        raise SystemExit(f"missing {f} (extract_field_csvs.py {args.run} \"{k}\" {suite})")
    d = pd.read_csv(f, usecols=["velocity:0", "velocity:1", "p", "T", "ocrust", "viscosity",
                                "shear_heating", "Points:0", "Points:1"])
    d["x"] = d["Points:0"] / 1e3
    d["z"] = YMAX - d["Points:1"] / 1e3
    # vertices shared by cells / MPI pieces appear once per piece: average them, or griddata's linear
    # triangulation stripes along the mesh lines
    return d.groupby([d.x.round(4), d.z.round(4)], sort=False).mean().reset_index(drop=True)


def trench_x(d):
    s = (d.ocrust > 0.5) & ((d.z - 10).abs() < 1)
    return float(d.x[s].min())


def grid(d, xlim):
    w = d[(d.x > xlim[0] - 5) & (d.x < xlim[1] + 5) & (d.z < args.zmax + 5)]
    xs = np.arange(xlim[0], xlim[1] + args.dx, args.dx)
    zs = np.arange(0, args.zmax + args.dx, args.dx)
    X, Z = np.meshgrid(xs, zs)
    pts = (w.x.values, w.z.values)
    g = {c: griddata(pts, w[c].values, (X, Z), method="linear")
         for c in ["velocity:0", "velocity:1", "p", "T", "ocrust", "shear_heating"]}
    g["eta"] = 10 ** griddata(pts, np.log10(w.viscosity.values), (X, Z), method="linear")
    h = args.dx * 1e3
    vx, vy = g["velocity:0"] / S_PER_YR, g["velocity:1"] / S_PER_YR      # m/yr -> m/s; y up, z down
    dvx_dx = np.gradient(vx, h, axis=1)
    dvy_dy = -np.gradient(vy, h, axis=0)                                   # d/dy = -d/dz
    dvx_dy = -np.gradient(vx, h, axis=0)
    dvy_dx = np.gradient(vy, h, axis=1)
    div = (dvx_dx + dvy_dy) / 3.0                                          # 3D convention, as ASPECT
    exx, eyy, exy = dvx_dx - div, dvy_dy - div, 0.5 * (dvx_dy + dvy_dx)
    g["eps"] = np.sqrt(0.5 * (exx ** 2 + eyy ** 2 + div ** 2) + exy ** 2)
    g["tau"] = 2 * g["eta"] * g["eps"]
    g["T"] = g["T"] - 273.15
    return X, Z, g


S_PER_YR = 3.15576e7
steps = {su: load(su, args.step) for su in args.suites}
xt = trench_x(steps[args.suites[0]])
xlim = args.xlim or (xt - 90, xt + 210)
G = {su: grid(d, xlim) for su, d in steps.items()}
t_myr = args.step * 0.5

# ------------------------------------------------------------------ figure 1: fields
cols = [("eta", "log$_{10}$ $\\eta$ (Pa s)", "viridis", (18.4, 23.4), np.log10),
        ("eps", "log$_{10}$ $\\dot\\varepsilon_{II}$ (s$^{-1}$)", "magma", (-16, -12), np.log10),
        ("tau", "$\\tau_{II}$ (MPa)", "cividis", (0, 150), lambda a: a / 1e6),
        ("shear_heating", "log$_{10}$ shear heating (W m$^{-3}$)", "inferno", (-7, -4),
         lambda a: np.log10(np.maximum(a, 1e-12)))]
fig, axs = plt.subplots(len(args.suites), len(cols), figsize=(4.2 * len(cols), 2.6 * len(args.suites) + 0.6),
                        sharex=True, sharey=True, squeeze=False)
for i, su in enumerate(args.suites):
    X, Z, g = G[su]
    for j, (key, lab, cmap, lim, f) in enumerate(cols):
        ax = axs[i, j]
        im = ax.pcolormesh(X, Z, f(g[key]), cmap=cmap, vmin=lim[0], vmax=lim[1], shading="auto", rasterized=True)
        ax.contour(X, Z, g["ocrust"], [0.5], colors="w", linewidths=0.8)
        cs = ax.contour(X, Z, g["T"], np.arange(200, 1401, 200), colors="0.85", linewidths=0.4, linestyles="--")
        if j == 0:
            ax.clabel(cs, fmt="%d", fontsize=6)
            ax.set_ylabel(f"{su}\ndepth (km)")
        if i == 0:
            fig.colorbar(im, ax=axs[:, j], orientation="horizontal", location="top", shrink=0.85, pad=0.02,
                         label=lab)
        ax.set_aspect("equal")
        ax.set_ylim(args.zmax, 0)
for ax in axs[-1]:
    ax.set_xlabel("x (km)")
fig.suptitle(f"{RUN}, t = {t_myr:g} Myr", y=1.0)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"{RUN}_t{args.step}_fields.{ext}"), dpi=200, bbox_inches="tight")
plt.close(fig)


# ------------------------------------------------------------------ figure 2: channel by depth
def channel_profile(g, X, Z):
    m = (g["ocrust"] > 0.5) & (Z > 3) & (X > xt - 15)
    zb = np.arange(3, args.zmax, 2.0)
    out = {k: [] for k in ["z", "eta", "eps", "tau", "T", "p", "sh"]}
    for z0 in zb:
        s = m & (Z >= z0) & (Z < z0 + 2)
        if s.sum() < 5:
            continue
        out["z"].append(z0 + 1)
        for k, src in [("eta", "eta"), ("eps", "eps"), ("tau", "tau"), ("T", "T"), ("p", "p"), ("sh", "shear_heating")]:
            out[k].append(np.nanmedian(g[src][s]))
    return {k: np.array(v) for k, v in out.items()}


fig, axs = plt.subplots(1, 5, figsize=(18, 5.2), sharey=True, gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 2.2]))
colors = {su: c for su, c in zip(args.suites, ["0.45", "#D55E00", "#0072B2", "#009E73"])}
for su in args.suites:
    X, Z, g = G[su]
    pr = channel_profile(g, X, Z)
    kw = dict(color=colors[su], lw=1.4, label=su)
    axs[0].plot(np.log10(pr["eta"]), pr["z"], **kw)
    axs[1].plot(np.log10(pr["eps"]), pr["z"], **kw)
    axs[2].plot(pr["tau"] / 1e6, pr["z"], **kw)
    axs[3].plot(pr["T"], pr["z"], **kw)
    if su.endswith("shp"):
        ty = C_DP * np.cos(PHI) + np.maximum(pr["p"], 0) * np.sin(PHI)
        TK, pp = pr["T"] + 273.15, np.maximum(pr["p"], 0)
        eta_c = 0.5 * QTZ_A ** (-1 / QTZ_N) * pr["eps"] ** ((1 - QTZ_N) / QTZ_N) * np.exp(QTZ_E / (QTZ_N * R * (TK + ADIA * pp)))
        axs[2].plot(ty / 1e6, pr["z"], color=colors[su], lw=0.9, ls="--", label="DP yield (binned p)")
        axs[2].plot(2 * eta_c * pr["eps"] / 1e6, pr["z"], color=colors[su], lw=0.9, ls=":",
                    label="qtz creep (binned T, $\\dot\\varepsilon$)")
for k in range(1, args.step + 1):
    for su in args.suites:
        d = load(su, k) if k != args.step else steps[su]
        d = d[(d.x > xlim[0]) & (d.x < xlim[1]) & (d.z < args.zmax)]
        axs[4].tricontour(d.x, d.z, d.ocrust, [0.5], colors=colors[su], linewidths=0.4 + 0.8 * k / args.step,
                          alpha=0.35 + 0.65 * k / args.step)
axs[4].set_xlim(xlim)
axs[4].set_aspect("equal")
axs[4].set_xlabel("x (km)")
axs[4].set_title(f"ocrust 0.5, t = 0.5..{t_myr:g} Myr", fontsize=9)
for ax, lab in zip(axs[:4], ["log$_{10}$ $\\eta$ (Pa s)", "log$_{10}$ $\\dot\\varepsilon_{II}$ (s$^{-1}$)",
                             "$\\tau_{II}$ (MPa)", "T (°C)"]):
    ax.set_xlabel(lab)
axs[2].set_xlim(0, 200)
axs[0].set_ylabel("depth (km)")
axs[0].set_ylim(args.zmax, 0)
axs[2].legend(fontsize=7, loc="lower right")
fig.suptitle(f"{RUN}, t = {t_myr:g} Myr: channel medians (ocrust > 0.5, 2 km bins)", y=1.0)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"{RUN}_t{args.step}_channel.{ext}"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", os.path.join(OUT, f"{RUN}_t{args.step}_" + "{fields,channel}.{pdf,png}"))

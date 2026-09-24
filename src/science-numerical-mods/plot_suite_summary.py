#!/usr/bin/env python3
"""Numerical-model science summary for one or more suites (default const-vc ramped-vc), built from the
processed slab-top record subd-model-runs/<suite>/analysis/run_XXX/Tprof_k.csv (k = 0..20, 0.5 Myr
steps) on the 0-80 km grid. Only runs with the full 20-step record are used.

 (A),(B) slab-top T(z) per suite at 0.5, 2, 5 and 10 Myr: median (line) and 5-95 % envelope (band)
         across the design, with the Agard et al. (2018) peak P-T compilation (data/rocks/).
 (C)     slab-top T at 40 and 80 km vs time since initiation: median and interquartile range,
         both suites -- the transient and its approach to a quasi-steady state.
 (D)     time to 90 % of the 0.5-10 Myr cooling (t90) vs depth: median and IQR per suite -- how long
         the transient lasts, and where.
 (E)     mean cooling rate vs depth for the 0.5-5 and 5-10 Myr windows: median and IQR per suite.
 (F)     T at 40 km at 5 Myr against convergence rate, coloured by overriding-plate age, both suites --
         the two dominant controls and how the ramp reshuffles them.

Style / conventions: src/emulator/science/emu_style.py (suite -> line style + marker fill,
time -> plasma, window -> fixed colours, Myriad Pro, fonts as text).

Usage:  env/bin/python src/science-numerical-mods/plot_suite_summary.py [SUITE ...] [--refresh]
Output: plots/science-numerical-mods/suite_summary[_<suites>].{pdf,svg,png}
A cache of the loaded record is kept at subd-model-runs/<suite>/analysis/slabtop_record_0-80km.npz
(--refresh rebuilds it).
"""
import argparse
import glob
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "emulator", "science"))
import emu_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import warnings  # noqa: E402
warnings.filterwarnings("ignore", message="All-NaN slice")
import pandas as pd  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("suites", nargs="*", default=["const-vc", "ramped-vc"])
ap.add_argument("--refresh", action="store_true")
ap.add_argument("--zmax", type=float, default=80.0)
args = ap.parse_args()
SUITES, ZMAX = args.suites, args.zmax
S.apply_style()

ZG = np.arange(0.0, ZMAX + 0.01, 1.0)
KS = list(range(0, 21))
TIMES = 0.5 * np.array(KS)


def load_params(suite):
    base = os.path.join(ROOT, "data", "params", f"params-list.{suite}")
    if os.path.exists(base + ".csv"):
        return pd.read_csv(base + ".csv")
    arr = np.load(base + ".npy")
    cols = {5: ["v_conv", "age_SP", "age_OP", "dip_int", "eta_UM"],
            6: ["v_conv", "t_conv", "age_SP", "age_OP", "dip_int", "eta_UM"]}[arr.shape[1]]
    return pd.DataFrame(arr, columns=cols)


def load_record(suite):
    """T[n_runs, n_times, n_depth] on ZG for runs with Tprof_0..20 present and finite on 0-ZMAX."""
    an = os.path.join(ROOT, "subd-model-runs", suite, "analysis")
    cache = os.path.join(an, f"slabtop_record_0-{ZMAX:g}km.npz")
    if os.path.exists(cache) and not args.refresh:
        c = np.load(cache)
        return c["run_ids"], c["T"]
    ids, T = [], []
    for rd in sorted(glob.glob(os.path.join(an, "run_[0-9][0-9][0-9]"))):
        prof = []
        for k in KS:
            f = os.path.join(rd, f"Tprof_{k}.csv")
            if not os.path.exists(f):
                prof = None; break
            d = pd.read_csv(f).dropna(subset=["depth_km", "T_C"]).sort_values("depth_km").drop_duplicates("depth_km")
            if d.depth_km.max() < ZMAX - 1e-6:
                prof = None; break
            prof.append(np.interp(ZG, d.depth_km, d.T_C))
        if prof is None:
            continue
        P = np.array(prof)
        if not np.isfinite(P).all():
            continue
        ids.append(int(os.path.basename(rd)[4:])); T.append(P)
    ids, T = np.array(ids), np.array(T)
    np.savez_compressed(cache, run_ids=ids, T=T, depth_km=ZG, time_myr=TIMES)
    return ids, T


REC, PAR = {}, {}
for s in SUITES:
    ids, T = load_record(s)
    REC[s] = (ids, T); PAR[s] = load_params(s)
    print(f"{s}: {len(ids)} runs with the full 0-10 Myr record on 0-{ZMAX:g} km")

rock = pd.read_csv(os.path.join(ROOT, "data", "rocks", "Agard_2018.csv")).dropna(subset=["T_C", "P_GPa"])
rock_z = rock.P_GPa * 1e9 / (3000 * 9.81) / 1e3
rock = rock[rock_z <= ZMAX]; rock_z = rock_z[rock_z <= ZMAX]

# ---------------------------------------------------------------- layout (inches)
n = len(SUITES)
L, GAP, R_STRIP, RM = 0.50, 0.52, 1.05, 0.10
W_P, H_P = 1.62, 1.62
TOP, ROW_GAP, BOT = 0.26, 0.55, 0.42
ncol = 3
FIG_W = L + ncol * W_P + (ncol - 1) * GAP + R_STRIP + RM
FIG_H = TOP + 2 * H_P + ROW_GAP + BOT
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax_in = lambda x, y, w, h: fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H])
X = [L + i * (W_P + GAP) for i in range(ncol)]
Y1, Y0 = BOT + H_P + ROW_GAP, BOT
axA, axB, axC = ax_in(X[0], Y1, W_P, H_P), ax_in(X[1], Y1, W_P, H_P), ax_in(X[2], Y1, W_P, H_P)
axD, axE, axF = ax_in(X[0], Y0, W_P, H_P), ax_in(X[1], Y0, W_P, H_P), ax_in(X[2], Y0, W_P, H_P)

# ---------------------------------------------------------------- (A),(B) T(z) envelopes per suite
ENV_TIMES = [0.5, 2.0, 5.0, 10.0]
for ax, s in zip([axA, axB], SUITES[:2]):
    ids, T = REC[s]
    for t in ENV_TIMES:
        k = int(round(t * 2)); A = T[:, k, :]
        lo, med, hi = np.percentile(A, [5, 50, 95], axis=0)
        ax.fill_betweenx(ZG, lo, hi, color=S.time_color(t), alpha=0.14, lw=0)
        ax.plot(med, ZG, color=S.time_color(t), lw=1.3, ls=S.SUITE_LS.get(s, "-"))
    ax.scatter(rock.T_C, rock_z, s=5, color="0.25", zorder=5, linewidths=0)
    ax.set_xlim(0, 1200); ax.set_ylim(ZMAX, 0)
    ax.set_xticks([0, 400, 800, 1200]); ax.set_yticks(np.arange(0, ZMAX + 1, 20))
    ax.set_xlabel("Slab-top T (" + S.DEG + "C)")
    ax.text(0.04, 0.04, f"{S.SUITE_LABEL.get(s, s)}\nn = {len(ids)}", transform=ax.transAxes, fontsize=6.5,
            ha="left", va="bottom", color="0.25")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
axA.set_ylabel("Depth (km)"); axB.tick_params(labelleft=False)
h_t = [Line2D([0], [0], color=S.time_color(t), lw=1.3, label=f"{t:g} Myr") for t in ENV_TIMES]
h_t.append(Line2D([0], [0], marker="o", color="0.25", lw=0, ms=2.5, label="Agard et al. 2018\npeak P-T"))
# (placed in the strip right of (C) after that panel is drawn)

# ---------------------------------------------------------------- (C) T at 40 / 80 km vs time
C_DEPTHS = [40.0, 80.0]
depth_cmap = plt.get_cmap("viridis"); depth_norm = Normalize(0, ZMAX)
for s in SUITES:
    ids, T = REC[s]
    for z in C_DEPTHS:
        iz = int(np.argmin(np.abs(ZG - z))); A = T[:, 1:, iz]   # from 0.5 Myr on
        lo, med, hi = np.percentile(A, [25, 50, 75], axis=0)
        c = depth_cmap(depth_norm(z) * 0.85)
        axC.fill_between(TIMES[1:], lo, hi, color=c, alpha=0.12 if S.SUITE_LS.get(s) == "-" else 0.08, lw=0)
        axC.plot(TIMES[1:], med, **S.suite_kw(s, c))
        print(f"{s} T({z:g} km): median {med[0]:.0f} -> {med[-1]:.0f} C over 0.5-10 Myr; "
              f"IQR at 10 Myr {hi[-1]-lo[-1]:.0f} C")
axC.set_xlim(0, 10.5); axC.set_xticks([0, 2.5, 5, 7.5, 10])
axC.set_xlabel("Time since initiation (Myr)")
axC.set_ylabel("Slab-top T (" + S.DEG + "C)")
axC.spines["top"].set_visible(False); axC.spines["right"].set_visible(False)
h_c = [Line2D([0], [0], color=depth_cmap(depth_norm(z) * 0.85), lw=1.3, label=f"{z:g} km") for z in C_DEPTHS]
leg = axC.legend(handles=h_c, fontsize=6, loc="upper right", frameon=True, facecolor="white", edgecolor="black",
                 framealpha=0.9, handlelength=1.4, labelspacing=0.25, borderpad=0.35, title="depth (median, IQR)",
                 title_fontsize=6)
leg.get_frame().set_linewidth(0.5)
S.outside_legend(axC, h_t, 0.36, handlelength=1.4, title="(A),(B): median, 5-95 %", title_fontsize=6.5)
S.outside_legend(axC, S.suite_handles(SUITES), -0.02, handlelength=2.2, title="suite", title_fontsize=6.5)

# ---------------------------------------------------------------- (D) t90 vs depth
for s in SUITES:
    ids, T = REC[s]
    T0, T10 = T[:, 1, :], T[:, 20, :]                      # 0.5 and 10 Myr
    total = T0 - T10                                        # total cooling (positive)
    frac = (T0[:, None, :] - T[:, 1:, :]) / np.where(np.abs(total) > 1e-9, total, np.nan)[:, None, :]
    t90 = np.full(T0.shape, np.nan)
    for i in range(T.shape[0]):
        for j in range(len(ZG)):
            if total[i, j] < 20:                            # < 20 C of net cooling: transient undefined
                continue
            f = frac[i, 1:, j]; tt = TIMES[2:]
            idx = np.argmax(f >= 0.9)
            if f[idx] >= 0.9:
                t90[i, j] = tt[idx] if idx == 0 else np.interp(0.9, [f[idx - 1], f[idx]], [tt[idx - 1], tt[idx]])
    t90[:, ZG < 5] = np.nan                                # top 5 km: little net cooling, noisy
    lo, med, hi = np.nanpercentile(t90, [25, 50, 75], axis=0)
    ok = np.isfinite(med)
    axD.fill_betweenx(ZG[ok], lo[ok], hi[ok], color="0.15", alpha=0.12 if S.SUITE_LS.get(s) == "-" else 0.07, lw=0)
    axD.plot(med[ok], ZG[ok], color="0.15", lw=1.3, ls=S.SUITE_LS.get(s, ":"))
    for z in [20, 40, 60, 80]:
        j = int(np.argmin(np.abs(ZG - z)))
        print(f"{s} t90 at {z} km: median {med[j]:.1f} Myr (IQR {lo[j]:.1f}-{hi[j]:.1f}), defined for "
              f"{np.isfinite(t90[:, j]).mean():.0%} of runs")
axD.set_ylim(ZMAX, 0); axD.set_xlim(0, 10)
axD.set_yticks(np.arange(0, ZMAX + 1, 20)); axD.set_xticks([0, 2.5, 5, 7.5, 10])
axD.set_xlabel(r"$t_{90}$ of the 0.5-10 Myr cooling (Myr)")
axD.set_ylabel("Depth (km)")
axD.spines["top"].set_visible(False); axD.spines["right"].set_visible(False)
axD.text(0.04, 0.04, "time at which 90 % of the\n0.5-10 Myr cooling is done\n(median, IQR; runs cooling\n> 20 " + S.DEG + "C at that depth)",
         transform=axD.transAxes, fontsize=6, color="0.35", ha="left", va="bottom", linespacing=1.25)

# ---------------------------------------------------------------- (E) cooling rate vs depth, two windows
for s in SUITES:
    ids, T = REC[s]
    for w, (k0, k1) in [("0.5-5 Myr", (1, 10)), ("5-10 Myr", (10, 20))]:
        rate = -(T[:, k1, :] - T[:, k0, :]) / (0.5 * (k1 - k0))    # C/Myr, positive = cooling
        lo, med, hi = np.percentile(rate, [25, 50, 75], axis=0)
        axE.fill_betweenx(ZG, lo, hi, color=S.WINDOW_COLOR[w], alpha=0.12 if S.SUITE_LS.get(s) == "-" else 0.08, lw=0)
        axE.plot(med, ZG, color=S.WINDOW_COLOR[w], lw=1.3, ls=S.SUITE_LS.get(s, ":"))
        j = int(np.argmin(np.abs(ZG - 40)))
        print(f"{s} cooling rate {w} at 40 km: median {med[j]:.1f} C/Myr (IQR {lo[j]:.1f}-{hi[j]:.1f})")
axE.set_ylim(ZMAX, 0); axE.set_yticks(np.arange(0, ZMAX + 1, 20))
axE.set_xlim(left=0)
axE.set_xlabel("Mean cooling rate (" + S.DEG + "C/Myr)")
axE.spines["top"].set_visible(False); axE.spines["right"].set_visible(False)
axE.tick_params(labelleft=False)
h_w = [Line2D([0], [0], color=S.WINDOW_COLOR[w], lw=1.3, label=w) for w in ["0.5-5 Myr", "5-10 Myr"]]
leg = axE.legend(handles=h_w, fontsize=6, loc="lower right", frameon=True, facecolor="white", edgecolor="black",
                 framealpha=0.9, handlelength=1.4, labelspacing=0.25, borderpad=0.35, title="window (median, IQR)",
                 title_fontsize=6)
leg.get_frame().set_linewidth(0.5)

# ---------------------------------------------------------------- (F) T(40 km, 5 Myr) vs v_conv, colour age_OP
age_cmap = plt.get_cmap("viridis")
allop = np.concatenate([PAR[s].age_OP.values for s in SUITES])
age_norm = Normalize(allop.min(), allop.max())
zF, tF = 40.0, 5.0
jz, kt = int(np.argmin(np.abs(ZG - zF))), int(round(tF * 2))
for s in SUITES:
    ids, T = REC[s]; P = PAR[s].iloc[ids]
    y = T[:, kt, jz]
    filled = S.SUITE_LS.get(s) == "-"
    cols = age_cmap(age_norm(P.age_OP.values))
    if filled:
        axF.scatter(P.v_conv, y, c=cols, s=7, linewidths=0, alpha=0.85, zorder=3)
    else:
        axF.scatter(P.v_conv, y, facecolors="none", edgecolors=cols, s=8, linewidths=0.6, alpha=0.85, zorder=2)
    r = np.corrcoef(np.log(P.v_conv), y)[0, 1]
    print(f"{s} T(40 km, 5 Myr): {y.min():.0f}-{y.max():.0f} C; corr with log v_conv {r:+.2f}, "
          f"with age_OP {np.corrcoef(P.age_OP, y)[0, 1]:+.2f}")
axF.set_xscale("log")
axF.set_xticks([1, 2, 3, 5, 8]); axF.set_xticklabels(["1", "2", "3", "5", "8"]); axF.minorticks_off()
axF.set_xlabel("Convergence rate (cm/yr)")
axF.set_ylabel("T at 40 km, 5 Myr (" + S.DEG + "C)")
axF.spines["top"].set_visible(False); axF.spines["right"].set_visible(False)
sm = plt.cm.ScalarMappable(cmap=age_cmap, norm=age_norm); sm.set_array([])
cax = axF.inset_axes([1.06, 0.55, 0.05, 0.42], transform=axF.transAxes)
cb = fig.colorbar(sm, cax=cax)
cb.set_label(r"$\mathrm{age}_{\mathrm{OP}}$ (Myr)", fontsize=6.5, labelpad=2)
cb.ax.tick_params(labelsize=6.5, length=2, pad=1.5); cb.outline.set_linewidth(0.5)
h_f = [Line2D([0], [0], marker="o", color="0.3", lw=0, ms=3, mfc="0.3", label=S.SUITE_LABEL.get(SUITES[0], SUITES[0]))]
if n > 1:
    h_f.append(Line2D([0], [0], marker="o", color="0.3", lw=0, ms=3, mfc="white", mew=0.7,
                      label=S.SUITE_LABEL.get(SUITES[1], SUITES[1])))
S.outside_legend(axF, h_f, 0.10, handlelength=1.0, title="suite", title_fontsize=6.5)

S.panel_labels(fig, [(axA, "(A)"), (axB, "(B)"), (axC, "(C)"), (axD, "(D)"), (axE, "(E)"), (axF, "(F)")])
os.makedirs(os.path.join(ROOT, "plots", "science-numerical-mods"), exist_ok=True)
stem = "suite_summary" + ("" if SUITES == ["const-vc", "ramped-vc"] else "_" + "_".join(SUITES))
base = os.path.join(ROOT, "plots", "science-numerical-mods", stem)
for ext in ("pdf", "svg"):
    fig.savefig(base + "." + ext, facecolor="white")
fig.savefig(base + ".png", facecolor="white", dpi=300)
print("wrote", base + ".{pdf,svg,png}")

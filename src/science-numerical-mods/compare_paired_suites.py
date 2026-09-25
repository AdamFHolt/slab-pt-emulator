#!/usr/bin/env python3
"""Paired comparison of a one-term ablation suite against its reference (default const-vc): what
does the changed term do to the slab-top thermal record, run by run?

The paired suites share const-vc's 400-point design, so run_XXX pairs with run_XXX:
  const-vc-dd100  decoupling depth capped at 100 km (150 km in const-vc); run for dip >= 35 only,
                  so --dip-min 35 is applied by default (drops its four dip < 35 pilot runs and the
                  same runs of the reference).
  const-vc-sh     shear heating on (ASPECT 3.0; the 3.0-vs-2.5 version effect is noise-level, see
                  const-vc-v3ctrl), all dips.
Only runs with the full 0-10 Myr record in BOTH suites are used. dT = T(suite) - T(reference) on the
0-100 km slab-top grid (85-100 km only exists after the crust arrives, so NaNs there are allowed and
the statistics are NaN-aware).

 (A) dT(z) at 0.5, 2, 5 and 10 Myr: median and IQR across the pairs.
 (B) dT at 40, 80 and 100 km vs time since initiation: median and IQR.
 (C) slab-top T(z) at 2 and 10 Myr, both suites on the paired subset: median and 5-95 % envelope
     with the Agard et al. (2018) rocks -- the envelopes the ablation does (not) move.
 (D) per-pair dT at 10 Myr at the depth (20-80 km) where the median effect is largest -- 100 km, the
     cutoff level, for dd100 -- or --d-depth, against its strongest design control (Spearman),
     coloured by the second. When that control is v_conv and the effect has one sign (shear
     heating), the axes are log-log, the pairs at 2, 5 and 10 Myr are shown coloured by time and a
     least-squares power law dT ~ v_conv^n is drawn per time (exponent and r in the legend).

Usage:  env/bin/python src/science-numerical-mods/compare_paired_suites.py [SUITE] [--ref const-vc]
            [--dip-min D] [--d-depth Z] [--refresh]
Output: plots/science-numerical-mods/<suite>/<short>_pairs.{pdf,svg,png}  (short = suite minus 'const-vc-')
Cache:  subd-model-runs/<suite>/analysis/slabtop_record_0-100km_nan.npz (--refresh rebuilds).
"""
import argparse
import glob
import os
import sys
import warnings

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "emulator", "science"))
import emu_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

warnings.filterwarnings("ignore", message="All-NaN slice")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("suite", nargs="?", default="const-vc-dd100")
ap.add_argument("--ref", default="const-vc")
ap.add_argument("--refresh", action="store_true")
ap.add_argument("--dip-min", type=float, default=None,
                help="keep pairs with dip_int >= D in both suites (default 35 for const-vc-dd100, none otherwise)")
ap.add_argument("--zmax", type=float, default=100.0)
ap.add_argument("--d-depth", type=float, default=None,
                help="depth of the per-pair scatter in (D); default = depth (20-80 km) of the largest median |dT| "
                     "at 10 Myr, or 100 km (the cutoff level) for const-vc-dd100")
args = ap.parse_args()
S.apply_style()

SUITE_A, SUITE_B = args.suite, args.ref
SHORT = SUITE_A[len("const-vc-"):] if SUITE_A.startswith("const-vc-") else SUITE_A
DIP_MIN = args.dip_min if args.dip_min is not None else (35.0 if SUITE_A == "const-vc-dd100" else None)
ZMAX = args.zmax
ZG = np.arange(0.0, ZMAX + 0.01, 1.0)
KS = list(range(0, 21))
TIMES = 0.5 * np.array(KS)
ZVALID = 80.0          # the record must be finite on 0-80 km at every step to count as complete
OUT_DIR = os.path.join(ROOT, "plots", "science-numerical-mods", SUITE_A)
PARAM_LABEL = {"v_conv": r"v$_{\mathrm{conv}}$ (cm/yr)", "age_SP": r"age$_{\mathrm{SP}}$ (Myr)", "age_OP": r"age$_{\mathrm{OP}}$ (Myr)",
               "dip_int": "dip (" + S.DEG + ")", "eta_UM": r"$\eta_{UM}$ (Pa s)"}


def load_params(suite):
    return pd.read_csv(os.path.join(ROOT, "data", "params", f"params-list.{suite}.csv"))


def load_record(suite):
    """T[n_runs, 21, n_depth] on ZG; runs with Tprof_0..20 present and finite on 0-ZVALID km.
    Deeper values may be NaN (slab top not yet there)."""
    an = os.path.join(ROOT, "subd-model-runs", suite, "analysis")
    cache = os.path.join(an, f"slabtop_record_0-{ZMAX:g}km_nan.npz")
    if os.path.exists(cache) and not args.refresh:
        c = np.load(cache)
        return c["run_ids"], c["T"]
    ids, T = [], []
    jv = int(np.argmin(np.abs(ZG - ZVALID)))
    for rd in sorted(glob.glob(os.path.join(an, "run_[0-9][0-9][0-9]"))):
        prof = []
        for k in KS:
            f = os.path.join(rd, f"Tprof_{k}.csv")
            if not os.path.exists(f):
                prof = None; break
            d = pd.read_csv(f).dropna(subset=["depth_km", "T_C"]).sort_values("depth_km").drop_duplicates("depth_km")
            p = np.interp(ZG, d.depth_km, d.T_C, left=np.nan, right=np.nan)
            prof.append(p)
        if prof is None:
            continue
        P = np.array(prof)
        if not np.isfinite(P[:, :jv + 1]).all():
            continue
        ids.append(int(os.path.basename(rd)[4:])); T.append(P)
    ids, T = np.array(ids), np.array(T)
    np.savez_compressed(cache, run_ids=ids, T=T, depth_km=ZG, time_myr=TIMES)
    return ids, T


PAR = load_params(SUITE_B)                       # same design in both suites
REC = {s: load_record(s) for s in (SUITE_A, SUITE_B)}
for s, (ids, T) in REC.items():
    print(f"{s}: {len(ids)} runs with the full 0-10 Myr record on 0-{ZVALID:g} km")
keep = set(np.where(PAR.dip_int.values >= DIP_MIN)[0]) if DIP_MIN is not None else set(range(len(PAR)))
pair_ids = sorted(set(REC[SUITE_A][0]) & set(REC[SUITE_B][0]) & keep)
dropped = sorted(set(REC[SUITE_A][0]) - keep)
print(f"pairs: {len(pair_ids)}" + (f" (dip >= {DIP_MIN:g}; {SHORT} runs dropped for dip < {DIP_MIN:g}: "
      f"{' '.join(f'{i:03d}' for i in dropped) or 'none'})" if DIP_MIN is not None else " (all dips)"))
if len(pair_ids) < 5:
    sys.exit("too few pairs -- has the suite been extracted (extend_profiles_all-mods.sh) yet?")
TA = REC[SUITE_A][1][[list(REC[SUITE_A][0]).index(i) for i in pair_ids]]
TB = REC[SUITE_B][1][[list(REC[SUITE_B][0]).index(i) for i in pair_ids]]
P = PAR.iloc[pair_ids].reset_index(drop=True)
DT = TA - TB                                     # [pair, time, depth]

rock = pd.read_csv(os.path.join(ROOT, "data", "rocks", "Agard_2018.csv")).dropna(subset=["T_C", "P_GPa"])
rock_z = rock.P_GPa * 1e9 / (3000 * 9.81) / 1e3
rock = rock[rock_z <= ZMAX]; rock_z = rock_z[rock_z <= ZMAX]

# ---------------------------------------------------------------- numbers
iz = lambda z: int(np.argmin(np.abs(ZG - z)))
print(f"\ndT = {SUITE_A} - {SUITE_B} (C) per depth and time: median [IQR] (p95 |dT|), n finite")
for z in (20, 40, 60, 80, 90, 100):
    row = []
    for t in (0.5, 2, 5, 10):
        d = DT[:, int(round(2 * t)), iz(z)]; n = np.isfinite(d).sum()
        if n < 5:
            row.append(f"{t:>4g} Myr: n={n:<3d}"); continue
        lo, med, hi = np.nanpercentile(d, [25, 50, 75]); p95 = np.nanpercentile(np.abs(d), 95)
        row.append(f"{t:>4g} Myr: {med:+5.1f} [{lo:+5.1f},{hi:+5.1f}] ({p95:4.1f}) n={n}")
    print(f"  {z:>3d} km | " + " | ".join(row))
d80 = DT[:, 20, iz(80)]; d40 = DT[:, 20, iz(40)]
m = np.isfinite(DT[:, 20, :iz(80) + 1])
print(f"\n0-80 km, 10 Myr, all pairs pooled: mean {np.nanmean(DT[:, 20, :iz(80) + 1]):+.1f} C, "
      f"median |dT| {np.nanmedian(np.abs(DT[:, 20, :iz(80) + 1])):.1f}, p95 |dT| "
      f"{np.nanpercentile(np.abs(DT[:, 20, :iz(80) + 1]), 95):.1f}")
print("\nSpearman rho of dT(z, 10 Myr) with the design parameters:")
rho = {}
for z in (40, 80, 100):
    d = DT[:, 20, iz(z)]; ok = np.isfinite(d)
    rho[z] = {p: spearmanr(P[p].values[ok], d[ok]).correlation for p in P.columns}
    print(f"  {z:>3d} km: " + ", ".join(f"{p} {r:+.2f}" for p, r in rho[z].items()) + f"  (n={ok.sum()})")
if args.d_depth is not None:
    ZD = args.d_depth
elif SUITE_A == "const-vc-dd100":
    ZD = 100.0
else:
    cand = [20, 40, 60, 80]
    ZD = float(max(cand, key=lambda z: abs(np.nanmedian(DT[:, 20, iz(z)]))))
dD = DT[:, 20, iz(ZD)]; okD = np.isfinite(dD)
rhoD = {p: spearmanr(P[p].values[okD], dD[okD]).correlation for p in P.columns}
ctrl = sorted(rhoD.items(), key=lambda kv: -abs(kv[1]))
X1, X2 = ctrl[0][0], ctrl[1][0]
print(f"strongest controls of dT({ZD:g} km, 10 Myr): {X1} ({ctrl[0][1]:+.2f}), then {X2} ({ctrl[1][1]:+.2f})")

# ---------------------------------------------------------------- layout: 2 x 2 + right strip
L, GAP, R_STRIP, RM = 0.50, 0.62, 1.05, 0.10
W_P, H_P = 2.05, 1.72
TOP, ROW_GAP, BOT = 0.26, 0.55, 0.42
FIG_W = L + 2 * W_P + GAP + R_STRIP + RM
FIG_H = TOP + 2 * H_P + ROW_GAP + BOT
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax_in = lambda x, y, w, h: fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H])
X0, XR = L, L + W_P + GAP
Y0, Y1 = BOT + H_P + ROW_GAP, BOT
axA, axB = ax_in(X0, Y0, W_P, H_P), ax_in(XR, Y0, W_P, H_P)
axC, axD = ax_in(X0, Y1, W_P, H_P), ax_in(XR, Y1, W_P, H_P)
for ax in (axA, axB, axC, axD):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# ---------------------------------------------------------------- (A) dT(z) at four times
A_TIMES = [0.5, 2.0, 5.0, 10.0]
for t in A_TIMES:
    k = int(round(2 * t)); d = DT[:, k, :]
    n = np.isfinite(d).sum(axis=0); ok = n >= 5
    lo, med, hi = np.nanpercentile(d[:, ok], [25, 50, 75], axis=0)
    axA.fill_betweenx(ZG[ok], lo, hi, color=S.time_color(t), alpha=0.14, lw=0)
    axA.plot(med, ZG[ok], color=S.time_color(t), lw=1.3)
axA.axvline(0, color="0.5", lw=0.6, ls=":")
axA.set_ylim(ZMAX, 0); axA.set_yticks(np.arange(0, ZMAX + 1, 20))
qlo = np.nanpercentile(DT[:, [1, 4, 10, 20], :], 25, axis=0); qhi = np.nanpercentile(DT[:, [1, 4, 10, 20], :], 75, axis=0)
x0, x1 = min(0.0, np.nanmin(qlo)), max(0.0, np.nanmax(qhi)); pad = 0.12 * (x1 - x0)
axA.set_xlim(x0 - pad, x1 + pad)
axA.set_xlabel(r"$\Delta$T slab top (" + S.DEG + "C)")        # sign convention (suite - reference) goes in the caption
axA.set_ylabel("Depth (km)")
subtitle = f"{len(pair_ids)} pairs" + (f"\ndip $\\geq$ {DIP_MIN:g}" + S.DEG if DIP_MIN is not None else "")
axA.text(0.96, 0.04, subtitle, transform=axA.transAxes, fontsize=6.5, ha="right", va="bottom", color="0.25")
h_t = [Line2D([0], [0], color=S.time_color(t), lw=1.3, label=f"{t:g} Myr") for t in A_TIMES]
S.outside_legend(axB, h_t, 0.55, handlelength=1.4, title="median, IQR", title_fontsize=6.5)

# ---------------------------------------------------------------- (B) dT vs time at three depths
B_DEPTHS = [40.0, 80.0, 100.0]
depth_cmap = plt.get_cmap("viridis"); depth_norm = Normalize(0, ZMAX)
dcol = lambda z: depth_cmap(depth_norm(z) * 0.85)
for z in B_DEPTHS:
    d = DT[:, 1:, iz(z)]; n = np.isfinite(d).sum(axis=0); ok = n >= 5
    lo, med, hi = np.nanpercentile(d[:, ok], [25, 50, 75], axis=0)
    axB.fill_between(TIMES[1:][ok], lo, hi, color=dcol(z), alpha=0.12, lw=0)
    axB.plot(TIMES[1:][ok], med, color=dcol(z), lw=1.3, marker="o", ms=2.4, mfc=dcol(z), mec=dcol(z), mew=0.7)
axB.axhline(0, color="0.5", lw=0.6, ls=":")
axB.set_xlim(0, 10.5); axB.set_xticks([0, 2.5, 5, 7.5, 10])
axB.set_xlabel("Time since initiation (Myr)")
axB.set_ylabel(r"$\Delta$T slab top (" + S.DEG + "C)")
h_d = [Line2D([0], [0], color=dcol(z), lw=1.3, label=f"{z:g} km") for z in B_DEPTHS]
S.outside_legend(axB, h_d, 0.05, handlelength=1.4, title="depth; median, IQR", title_fontsize=6.5)

# ---------------------------------------------------------------- (C) both suites' envelopes, paired subset
C_TIMES = [2.0, 10.0]
for s, T in ((SUITE_B, TB), (SUITE_A, TA)):
    for t in C_TIMES:
        k = int(round(2 * t)); A = T[:, k, :]
        n = np.isfinite(A).sum(axis=0); ok = n >= 5
        lo, med, hi = np.nanpercentile(A[:, ok], [5, 50, 95], axis=0)
        if s == SUITE_B:
            axC.fill_betweenx(ZG[ok], lo, hi, color=S.time_color(t), alpha=0.14, lw=0)
        else:
            axC.plot(lo, ZG[ok], color=S.time_color(t), lw=0.7, ls=S.SUITE_LS.get(s, ":"))
            axC.plot(hi, ZG[ok], color=S.time_color(t), lw=0.7, ls=S.SUITE_LS.get(s, ":"))
        axC.plot(med, ZG[ok], color=S.time_color(t), lw=1.3, ls=S.SUITE_LS.get(s, ":"))
axC.scatter(rock.T_C, rock_z, s=5, color="0.25", zorder=5, linewidths=0)
axC.set_xlim(0, 1200); axC.set_ylim(ZMAX, 0)
axC.set_xticks([0, 400, 800, 1200]); axC.set_yticks(np.arange(0, ZMAX + 1, 20))
axC.set_xlabel("Slab-top T (" + S.DEG + "C)"); axC.set_ylabel("Depth (km)")
h_c = [Line2D([0], [0], color=S.time_color(t), lw=1.3, label=f"{t:g} Myr") for t in C_TIMES]
h_c += [Line2D([0], [0], color="0.15", lw=1.3, ls=S.SUITE_LS[SUITE_B], label=f"{SUITE_B} (band)"),
        Line2D([0], [0], color="0.15", lw=1.3, ls=S.SUITE_LS.get(SUITE_A, ":"), label=f"{SHORT} (lines)"),
        Line2D([0], [0], marker="o", color="0.25", lw=0, ms=2.5, label="Agard et al. 2018\npeak P-T")]
S.outside_legend(axD, h_c, 0.0, handlelength=2.0, title="(C): 5 / 50 / 95 %", title_fontsize=6.5)          # strip right of (D), below its colourbar

# ---------------------------------------------------------------- (D) per-pair dT at ZD vs the control
# Scaling mode (control = v_conv, effect one-signed; the shear-heating case): log-log axes, pairs at
# D_TIMES coloured by time, one least-squares power law dT ~ v_conv^n per time.  Otherwise: the 10 Myr
# pairs on linear axes coloured by the second control (dd100).
from matplotlib.ticker import NullFormatter, NullLocator  # noqa: E402
ok = okD
x = P[X1].values[ok]
fit_txt = f"Spearman $\\rho$ = {ctrl[0][1]:+.2f}"
scaling = X1 == "v_conv" and (dD[ok] > 0).mean() > 0.95
if scaling:
    D_TIMES = [2.0, 5.0, 10.0]
    lines = []; ymin, ymax = np.inf, -np.inf
    for t in D_TIMES:
        k = int(round(2 * t)); d = DT[:, k, iz(ZD)]; m = np.isfinite(d) & (d > 0)
        xv = P[X1].values[m]; col = S.time_color(t)
        axD.scatter(xv, d[m], color=col, s=9, linewidths=0.3, edgecolors="0.3", alpha=0.85, zorder=3)
        n_exp, lna = np.polyfit(np.log(xv), np.log(d[m]), 1)
        r_fit = np.corrcoef(np.log(xv), np.log(d[m]))[0, 1]
        xx = np.array([xv.min(), xv.max()])
        axD.plot(xx, np.exp(lna) * xx ** n_exp, color=col, lw=1.1, ls="--", zorder=2)
        lines.append(Line2D([0], [0], color=col, lw=1.1, ls="--", marker="o", ms=2.6, mfc=col, mec="0.3", mew=0.3,
                            label=f"{t:g} Myr\nn = {n_exp:.2f}, r = {r_fit:.2f}"))
        ymin, ymax = min(ymin, d[m].min()), max(ymax, d[m].max())
        print(f"(D) log-log fit at {t:g} Myr: dT({ZD:g} km) = {np.exp(lna):.1f} * v_conv^{n_exp:.2f} C, r = {r_fit:.2f}, "
              f"{m.sum()} pairs; at 1/4/8 cm/yr: {np.exp(lna):.0f}/{np.exp(lna)*4**n_exp:.0f}/{np.exp(lna)*8**n_exp:.0f} C")
    axD.set_xscale("log"); axD.set_yscale("log")
    axD.set_xticks([1, 2, 4, 8]); axD.set_xticklabels(["1", "2", "4", "8"])
    axD.set_yticks([3, 10, 30, 100, 300]); axD.set_yticklabels(["3", "10", "30", "100", "300"])
    for axis in (axD.xaxis, axD.yaxis):
        axis.set_minor_formatter(NullFormatter()); axis.set_minor_locator(NullLocator())
    axD.set_ylim(max(1.0, 0.7 * ymin), 1.3 * ymax)
    fit_txt = r"$\Delta$T $\propto$ v$_{\mathrm{conv}}^{\,n}$" + f"\n{ok.sum()} pairs"
    axD.set_ylabel(r"$\Delta$T (" + f"{ZD:g} km) (" + S.DEG + "C)")
    S.outside_legend(axD, lines, 0.56, handlelength=2.2, title="time; log-log fit", title_fontsize=6.5)
else:
    c = P[X2].values[ok]
    cnorm = LogNorm(c.min(), c.max()) if X2 == "eta_UM" else Normalize(c.min(), c.max())
    sc = axD.scatter(x, dD[ok], c=c, cmap="viridis", norm=cnorm, s=9, linewidths=0.3, edgecolors="0.3", alpha=0.9)
    if X1 == "eta_UM":
        axD.set_xscale("log")
    axD.axhline(0, color="0.5", lw=0.6, ls=":")
    axD.set_ylabel(r"$\Delta$T (" + f"{ZD:g} km, 10 Myr) (" + S.DEG + "C)")
    bD = axD.get_position()
    cax = fig.add_axes([bD.x1 + 0.08 / FIG_W, bD.y0 + 0.62 * bD.height, 0.09 / FIG_W, 0.36 * bD.height])
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(PARAM_LABEL[X2], fontsize=7); cb.ax.tick_params(labelsize=6)
axD.set_xlabel(PARAM_LABEL[X1])
axD.text(0.04, 0.96, fit_txt, transform=axD.transAxes, fontsize=6.5, ha="left", va="top", color="0.25")

S.panel_labels(fig, [(axA, "A"), (axB, "B"), (axC, "C"), (axD, "D")])
os.makedirs(OUT_DIR, exist_ok=True)
base = os.path.join(OUT_DIR, f"{SHORT}_pairs")
for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
    fig.savefig(f"{base}.{ext}", facecolor="white", **kw)
print("wrote", base + ".{pdf,svg,png}")

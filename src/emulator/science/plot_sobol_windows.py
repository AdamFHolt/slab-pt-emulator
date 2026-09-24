#!/usr/bin/env python3
"""Sobol sensitivity of the single-depth cooling-rate (dTdt) emulators vs depth, one column per
cooling window (0.5-5, 0.5-10, 5-10 Myr), one or more suites overlaid (suite -> line style).

Top row: total-effect index S_T vs depth (5-100 km every 5 km) for every sampled parameter
(Okabe-Ito colours, fixed per parameter; t_ramp exists only in ramped-vc), with the bootstrap 95 %
confidence interval (S_T +- ST_conf) as a light band. The 80 km deep-end validity line marks where the
slab top only exists after crust arrival, so 0.5-5 Myr cooling rates below it mix arrival and cooling.
Bottom row: held-out R^2 of the emulator behind each depth (the same suites, same line styles).
Printed: the depth(s) where S_T(age_OP) and S_T(v_conv) cross, per suite and window.

Usage:  env/bin/python src/emulator/science/plot_sobol_windows.py [SUITE ...]   (default const-vc ramped-vc)
Output: plots/science-emulator/summary/sobol_windows[_<suites>].{pdf,svg,png}
Inputs: plots/science-emulator/single_depth/<suite>/{sobol,sobol_dt1-20,sobol_dt10-20}/{d}km_dTdt_gp_m25_sobol.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emu_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("suites", nargs="*", default=["const-vc", "ramped-vc"])
ap.add_argument("--model-tag", default="gp_m25")
ap.add_argument("--no-ci", action="store_true", help="omit the confidence bands")
args = ap.parse_args()
SUITES, TAG = args.suites, args.model_tag
S.apply_style()

TAB = {s: {w: S.sobol_table(s, sub, TAG) for w, sub in S.WINDOWS} for s in SUITES}
present_params = []
for s in SUITES:
    for w, _ in S.WINDOWS:
        for j in TAB[s][w].values():
            for c in j["feature_cols"]:
                if c in S.PARAMS and c not in present_params:
                    present_params.append(c)
            break
PARAMS = [p for p in S.PARAMS if p in present_params]
zmax = max(max(t) for s in SUITES for t in TAB[s].values() if t)

# ---------------------------------------------------------------- layout (inches)
nw = len(S.WINDOWS)
L, GAP, R_STRIP, RM = 0.50, 0.38, 1.05, 0.10
W_P = 1.55
TOP, H_TOP, ROW_GAP, H_BOT, BOT = 0.28, 2.15, 0.62, 0.75, 0.42
FIG_W = L + nw * W_P + (nw - 1) * GAP + R_STRIP + RM
FIG_H = TOP + H_TOP + ROW_GAP + H_BOT + BOT
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax_in = lambda x, y, w, h: fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H])
axT = [ax_in(L + i * (W_P + GAP), BOT + H_BOT + ROW_GAP, W_P, H_TOP) for i in range(nw)]
axR = [ax_in(L + i * (W_P + GAP), BOT, W_P, H_BOT) for i in range(nw)]


def crossings(z, a, b):
    d = a - b
    out = []
    for i in range(len(z) - 1):
        if d[i] * d[i + 1] < 0:
            out.append(z[i] + (z[i + 1] - z[i]) * (-d[i]) / (d[i + 1] - d[i]))
    return out


for iw, (w, sub) in enumerate(S.WINDOWS):
    ax, axr = axT[iw], axR[iw]
    for s in SUITES:
        tab = TAB[s][w]
        ls = S.SUITE_LS.get(s, ":")
        if not tab:
            print(f"{s} {w}: missing ({sub})"); continue
        z = np.array(sorted(tab))
        cols = tab[z[0]]["feature_cols"]
        ST = {p: np.array([tab[d]["ST"][cols.index(p)] for d in z]) for p in PARAMS if p in cols}
        CI = {p: np.array([tab[d]["ST_conf"][cols.index(p)] for d in z]) for p in PARAMS if p in cols}
        for p in ST:
            if not args.no_ci:
                ax.fill_betweenx(z, np.clip(ST[p] - CI[p], 0, None), ST[p] + CI[p], color=S.PARAM_COLOR[p],
                                 alpha=0.10 if ls == "-" else 0.07, lw=0)
            ax.plot(ST[p], z, color=S.PARAM_COLOR[p], ls=ls, lw=1.2, marker="o" if ls == "-" else "s",
                    ms=2.0 if ls == "-" else 2.4, mfc=S.PARAM_COLOR[p] if ls == "-" else "white", mew=0.6)
        r2 = np.array([tab[d]["val_r2"] for d in z])
        axr.plot(r2, z, color="0.15", lw=1.1, ls=ls)
        if "age_OP" in ST and "v_conv" in ST:
            c = crossings(z, ST["age_OP"], ST["v_conv"])
            lead = "age_OP" if (ST["age_OP"] - ST["v_conv"])[0] > 0 else "v_conv"
            print(f"{s} {w}: age_OP/v_conv crossover " + (", ".join(f"{x:.1f} km" for x in c) if c else
                  f"none ({lead} leads throughout)") + f"; R2 {r2.min():.2f}-{r2.max():.2f}")
        extra = [p for p in ST if p not in ("age_OP", "v_conv", "age_SP", "dip_int", "eta_UM")]
        for p in extra:
            print(f"   {s} {w}: S_T({p}) {ST[p].min():.2f}-{ST[p].max():.2f}, max at {z[np.argmax(ST[p])]:g} km")
    for a in (ax, axr):
        a.axhline(S.DEEP_END_KM, color="0.5", lw=0.6, ls=":")
        a.set_ylim(zmax, 0)
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    ax.set_yticks(np.arange(0, zmax + 1, 20)); axr.set_yticks(np.arange(0, zmax + 1, 50))
    ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.5, 1.0])
    ax.set_title(w, fontsize=8, pad=4)
    ax.set_xlabel(r"Sobol $S_T$")
    axr.set_xlim(0.7, 1.0); axr.set_xticks([0.7, 0.85, 1.0])
    axr.set_xlabel(r"held-out $R^2$")
    if iw == 0:
        ax.set_ylabel("Depth (km)"); axr.set_ylabel("Depth (km)")
        axr.text(0.02, S.DEEP_END_KM - 2.0, "deep-end validity", fontsize=5.5, color="0.45", ha="left",
                 va="bottom", transform=axr.get_yaxis_transform())
    else:
        ax.tick_params(labelleft=False); axr.tick_params(labelleft=False)

h_p = [Line2D([0], [0], color=S.PARAM_COLOR[p], lw=1.2, label=S.PARAM_LABEL[p]
              + (" (ramped only)" if p == "t_conv" else "")) for p in PARAMS]
S.outside_legend(axT[-1], h_p, 0.42, handlelength=1.2, title="parameter", title_fontsize=6.5)
h_s = [Line2D([0], [0], color="0.15", lw=1.2, ls=S.SUITE_LS.get(s, ":"), marker="o" if S.SUITE_LS.get(s) == "-" else "s",
              ms=2.2, mfc="0.15" if S.SUITE_LS.get(s) == "-" else "white", mew=0.6, label=S.SUITE_LABEL.get(s, s))
       for s in SUITES]
S.outside_legend(axT[-1], h_s, 0.08, handlelength=2.0, title="suite", title_fontsize=6.5)
if not args.no_ci:
    fig.text((L + nw * W_P + (nw - 1) * GAP + 0.06) / FIG_W, (BOT + H_BOT) / FIG_H,
             "bands: bootstrap\n95 % CI of $S_T$", fontsize=6, color="0.35", ha="left", va="top")

S.panel_labels(fig, [(axT[i], f"({'ABC'[i]})") for i in range(nw)] + [(axR[i], f"({'DEF'[i]})") for i in range(nw)])
stem = "sobol_windows" + ("" if SUITES == ["const-vc", "ramped-vc"] else "_" + "_".join(SUITES))
S.save(fig, stem)

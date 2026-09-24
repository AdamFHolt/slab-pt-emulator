#!/usr/bin/env python3
"""Emulator validation summary for one or more suites (default: const-vc ramped-vc).

Rows = suites (one per suite):
  (A,C) held-out emulated vs modelled slab-top T at 5 Myr, every validation run and depth, coloured
        by depth; pooled RMSE, per-run p95 RMSE and n printed.
  (B,D) held-out profile RMSE vs depth at 0.5 / 3 / 5 / 10 Myr (plasma time colours), with the
        PCA-truncation floor (grey band: min-max over those times) -- the part of the error the GP
        cannot remove.
Right column, all suites together (suite -> line style):
  (E)   pooled held-out profile RMSE vs time since initiation for every scored slice (0.5-10 Myr),
        emulator (dark) and PCA floor (light grey); the p95 per-run RMSE is printed in (A)/(C) instead.
  (F)   held-out R^2 of the single-depth cooling-rate (dTdt) emulators vs depth for the three
        windows (0.5-5, 0.5-10, 5-10 Myr) -- the models the Sobol indices come from.

Usage:  env/bin/python src/emulator/science/plot_emulator_validation.py [SUITE ...]
Output: plots/science-emulator/summary/emulator_validation[_<suites>].{pdf,svg,png}
Companion: plot_sobol_windows.py (sensitivity), plot_emulator_validation_sobol.py (proposal-style, one suite).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import emu_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("suites", nargs="*", default=["const-vc", "ramped-vc"])
ap.add_argument("--model-tag", default="gp_m25")
ap.add_argument("--scatter-time", type=float, default=5.0)
ap.add_argument("--depth-times", default="0.5,3,5,10")
args = ap.parse_args()
SUITES, TAG = args.suites, args.model_tag
B_TIMES = [float(x) for x in args.depth_times.split(",")]
S.apply_style()

# ---------------------------------------------------------------- data
Q = {s: S.quality_reports(s, TAG) for s in SUITES}
for s in SUITES:
    if not Q[s]:
        sys.exit(f"no profile-PCA quality reports for {s}")
    print(f"{s}: PCA set {S.pca_set_for(s)}, {len(Q[s])} scored times "
          f"({min(Q[s]):g}-{max(Q[s]):g} Myr)")
SC = {s: S.reconstruct_val_profiles(s, args.scatter_time, TAG) for s in SUITES}
SOB = {s: {w: S.sobol_table(s, sub, TAG) for w, sub in S.WINDOWS} for s in SUITES}

# ---------------------------------------------------------------- layout (inches)
n = len(SUITES)
L, GAP_AB, GAP_R, R_STRIP, RM = 0.50, 0.50, 0.60, 1.05, 0.10
W_A, W_B, W_R = 1.60, 1.40, 1.55
TOP, ROW_GAP, BOT = 0.22, 0.48, 0.42
H_ROW = W_A
FIG_W = L + W_A + GAP_AB + W_B + GAP_R + W_R + R_STRIP + RM
FIG_H = TOP + n * H_ROW + (n - 1) * ROW_GAP + BOT
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax_in = lambda x, y, w, h: fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H])

xA, xB, xR = L, L + W_A + GAP_AB, L + W_A + GAP_AB + W_B + GAP_R
rows_y = [BOT + (n - 1 - i) * (H_ROW + ROW_GAP) for i in range(n)]
axA = [ax_in(xA, y, W_A, H_ROW) for y in rows_y]
axB = [ax_in(xB, y, W_B, H_ROW) for y in rows_y]
# right column: two panels sharing the full height
H_right = n * H_ROW + (n - 1) * ROW_GAP
h_r = (H_right - ROW_GAP) / 2
axE = ax_in(xR, BOT + h_r + ROW_GAP, W_R, h_r)
axF = ax_in(xR, BOT, W_R, h_r)

zmax_all = 0.0
# ---------------------------------------------------------------- rows
for i, s in enumerate(SUITES):
    zg, true, emu = SC[s]
    zmax = float(zg.max()); zmax_all = max(zmax_all, zmax)
    q5 = Q[s].get(args.scatter_time)
    rec = S.val_block(q5)["emulator_reconstruction"] if q5 else None
    rmse = float(np.sqrt(np.mean((true - emu) ** 2)))
    if rec:
        assert abs(rmse - rec["rmse"]) < 1e-6, f"{s}: reconstruction does not match stored RMSE"
    p95 = rec["per_run_rmse"]["p95"] if rec else float("nan")

    ax = axA[i]
    sc = ax.scatter(true.ravel(), emu.ravel(), c=np.tile(zg, true.shape[0]), cmap="viridis",
                    vmin=0, vmax=zmax, s=2.5, linewidths=0, alpha=0.75, rasterized=True)
    lims = [0, 1200]
    ax.plot(lims, lims, color="0.35", lw=0.8, zorder=1)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xticks(np.arange(0, 1201, 400)); ax.set_yticks(np.arange(0, 1201, 400))
    ax.set_aspect("equal", adjustable="box", anchor="N")
    ax.set_ylabel("Emulated slab-top T (" + S.DEG + "C)")
    if i == n - 1:
        ax.set_xlabel("Modelled slab-top T (" + S.DEG + "C)")
    ax.text(0.05, 0.95, f"{S.SUITE_LABEL.get(s, s)}\n{args.scatter_time:g} Myr, n = {true.shape[0]} held-out\n"
            f"RMSE {rmse:.1f}" + S.DEG + f"C, p95 run {p95:.0f}" + S.DEG + "C",
            transform=ax.transAxes, fontsize=6.5, ha="left", va="top", linespacing=1.25)
    if i == 0:
        cax = ax.inset_axes([0.70, 0.08, 0.05, 0.32])
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label("Depth (km)", fontsize=6.5, labelpad=2)
        cb.set_ticks([0, zmax / 2, zmax]); cb.ax.tick_params(labelsize=6.5, length=2, pad=1.5)
        cb.outline.set_linewidth(0.5)

    ax = axB[i]
    floors = []
    for t in B_TIMES:
        q = Q[s].get(t)
        if q is None:
            print(f"  {s}: no {t:g} Myr model, skipped in RMSE-vs-depth"); continue
        vb = S.val_block(q)
        zq = np.asarray(q["depth_grid_km"], float)
        ax.plot(vb["emulator_reconstruction"]["rmse_by_depth"], zq, color=S.time_color(t), lw=1.3,
                ls="-", label=f"{t:g} Myr")
        floors.append(np.asarray(vb["pca_truncation_baseline"]["rmse_by_depth"], float))
    if floors:
        F = np.vstack(floors)
        ax.fill_betweenx(zq, F.min(0), F.max(0), color="0.55", alpha=0.35, lw=0, label="PCA floor")
    ax.set_ylim(zmax, 0); ax.set_xlim(left=0)
    ax.set_yticks(np.arange(0, zmax + 1, 20))
    ax.set_ylabel("Depth (km)")
    if i == n - 1:
        ax.set_xlabel("Held-out profile RMSE (" + S.DEG + "C)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if i == 0:
        leg = ax.legend(fontsize=6.5, loc="upper right", frameon=True, facecolor="white", edgecolor="black",
                        framealpha=0.9, handlelength=1.6, labelspacing=0.25, borderpad=0.35,
                        title="time since\ninitiation", title_fontsize=6.5)
        leg.get_frame().set_linewidth(0.5)
    ax.text(0.97, 0.03, S.SUITE_LABEL.get(s, s), transform=ax.transAxes, fontsize=6.5, color="0.35",
            ha="right", va="bottom")
    xmax_b = ax.get_xlim()[1]
    ax.set_xlim(0, np.ceil(xmax_b / 5) * 5)

# ---------------------------------------------------------------- (E) RMSE vs time
for s in SUITES:
    ts = np.array(sorted(Q[s]))
    rm = np.array([S.val_block(Q[s][t])["emulator_reconstruction"]["rmse"] for t in ts])
    p95 = np.array([S.val_block(Q[s][t])["emulator_reconstruction"]["per_run_rmse"]["p95"] for t in ts])
    fl = np.array([S.val_block(Q[s][t])["pca_truncation_baseline"]["rmse"] for t in ts])
    ls = S.SUITE_LS.get(s, ":")
    axE.plot(ts, rm, **S.suite_kw(s, "0.15"))
    axE.plot(ts, fl, color="0.65", lw=1.1, ls=ls)
    print(f"{s}: pooled val RMSE {rm.min():.1f}-{rm.max():.1f} C over {ts.min():g}-{ts.max():g} Myr; "
          f"PCA floor {fl.min():.1f}-{fl.max():.1f} C; p95 per-run up to {p95.max():.0f} C")
axE.set_xlim(0, 10.5); axE.set_ylim(bottom=0)
axE.set_xticks([0, 2.5, 5, 7.5, 10])
axE.set_xlabel("Time since initiation (Myr)")
axE.set_ylabel("Held-out profile RMSE (" + S.DEG + "C)")
axE.spines["top"].set_visible(False); axE.spines["right"].set_visible(False)
h_e = [Line2D([0], [0], color="0.15", lw=1.3, label="GP emulator, pooled"),
       Line2D([0], [0], color="0.65", lw=1.1, label="PCA-truncation floor")]
S.outside_legend(axE, h_e, 0.62, handlelength=1.6)
# one suite key for (E) and (F), in the strip between them
S.outside_legend(axE, S.suite_handles(SUITES), -0.02, handlelength=2.2, title="suite", title_fontsize=6.5)

# ---------------------------------------------------------------- (F) dTdt R2 vs depth
for s in SUITES:
    ls = S.SUITE_LS.get(s, ":")
    for w, sub in S.WINDOWS:
        tab = SOB[s][w]
        if not tab:
            print(f"  {s}: window {w} missing ({sub})"); continue
        z = np.array(sorted(tab)); r2 = np.array([tab[d]["val_r2"] for d in z])
        axF.plot(r2, z, **S.suite_kw(s, S.WINDOW_COLOR[w], S.WINDOW_MARKER[w]))
        print(f"{s} {w}: dTdt held-out R2 {r2.min():.3f}-{r2.max():.3f} over {z.min():g}-{z.max():g} km "
              f"(min at {z[np.argmin(r2)]:g} km)")
zF = max(max(max(t) for t in SOB[s].values() if t) for s in SUITES)
axF.axhline(S.DEEP_END_KM, color="0.5", lw=0.6, ls=":")
axF.text(0.02, S.DEEP_END_KM - 1.5, "deep-end validity", fontsize=5.5, color="0.45", ha="left", va="bottom",
         transform=axF.get_yaxis_transform())
axF.set_ylim(zF, 0); axF.set_xlim(0.6, 1.0)
axF.set_yticks(np.arange(0, zF + 1, 20)); axF.set_xticks([0.6, 0.8, 1.0])
axF.set_xlabel(r"Held-out $R^2$, dTdt emulators")
axF.set_ylabel("Depth (km)")
axF.spines["top"].set_visible(False); axF.spines["right"].set_visible(False)
h_w = [Line2D([0], [0], color=S.WINDOW_COLOR[w], lw=1.3, marker=S.WINDOW_MARKER[w], ms=2.4, label=w)
       for w, _ in S.WINDOWS]
S.outside_legend(axF, h_w, 0.30, title="cooling window", title_fontsize=6.5)

labels = []
letters = iter("ABCDEFGHIJ")
for i in range(n):
    labels += [(axA[i], f"({next(letters)})"), (axB[i], f"({next(letters)})")]
labels += [(axE, f"({next(letters)})"), (axF, f"({next(letters)})")]
S.panel_labels(fig, labels)
stem = "emulator_validation" + ("" if SUITES == ["const-vc", "ramped-vc"] else "_" + "_".join(SUITES))
S.save(fig, stem)

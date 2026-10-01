#!/usr/bin/env python3
"""Run-by-run comparison of a pilot suite against the suites it is paired with: the slab-top record of each
pilot run for every suite overlaid, so the ablation can be read without any across-run statistics.

Default triplet: const-vc (no heating, ASPECT 2.5) / const-vc-sh (viscous 1e20 channel, shear heating,
ASPECT 3.0) / const-vc-sh-mu05 (sh with the shear-heating stress limiter at C = 1 MPa, mu' = 0.05).
The 2.5-vs-3.0 version effect is noise-level (const-vc-v3ctrl, 2026-09-24), so differences against
const-vc are the heating effect.

Figure: one row per pilot run (labelled with its design point); columns
  1-3  slab-top T(z) at --times (default 1, 5, 10 Myr), one line per suite
  4    the last suite minus the second (default mu05 - sh: what the limiter takes away) at each time
Runs are kept when the LAST suite has the full 0-10 Myr record on 0-80 km; a missing time for another
suite just leaves that line out.

Table (csv next to the figure): per run, time and depth, T of each suite and dT against the first.

Usage:  env/bin/python src/science-numerical-mods/compare_pilot_triplets.py
            [--suites const-vc const-vc-sh const-vc-sh-mu05] [--runs-file LIST] [--times 1 5 10]
Output: plots/science-numerical-mods/<last suite>/<short>_triplets.{pdf,svg,png} + _triplets_dT.csv
"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "emulator", "science"))
import emu_style as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--suites", nargs="+", default=["const-vc", "const-vc-sh", "const-vc-sh-mu05"])
ap.add_argument("--runs-file", default=None,
                help="run list (default: <last suite>/run-inputs/pilot-list.txt)")
ap.add_argument("--times", nargs="+", type=float, default=[1.0, 5.0, 10.0])
ap.add_argument("--zmax", type=float, default=100.0)
ap.add_argument("--tag", default=None, help="output stem (default: last suite minus 'const-vc-[sh-]')")
args = ap.parse_args()
S.apply_style()

SUITES = args.suites
LAST = SUITES[-1]
SHORT = lambda s: (s[len("const-vc-sh-"):] if s.startswith("const-vc-sh-")
                   else s[len("const-vc-"):] if s.startswith("const-vc-") else s)
TAG = args.tag or SHORT(LAST)
ZG = np.arange(0.0, args.zmax + 0.01, 1.0)
ZVALID = 80.0
COLORS = ["0.25", "#D55E00", "#0072B2", "#009E73", "#CC79A7"]          # Okabe-Ito after the grey reference
OUT_DIR = os.path.join(ROOT, "plots", "science-numerical-mods", LAST)

runs_file = args.runs_file or os.path.join(ROOT, "subd-model-runs", LAST, "run-inputs", "pilot-list.txt")
RUNS = [int(l.split("#")[0]) for l in open(runs_file) if l.split("#")[0].strip()]
PAR = pd.read_csv(os.path.join(ROOT, "data", "params", "params-list.const-vc.csv"))   # paired design


def profile(suite, rid, k):
    f = os.path.join(ROOT, "subd-model-runs", suite, "analysis", f"run_{rid:03d}", f"Tprof_{k}.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f).dropna(subset=["depth_km", "T_C"]).sort_values("depth_km").drop_duplicates("depth_km")
    return np.interp(ZG, d.depth_km, d.T_C, left=np.nan, right=np.nan)


def complete(suite, rid):
    jv = int(np.argmin(np.abs(ZG - ZVALID)))
    for k in range(21):
        p = profile(suite, rid, k)
        if p is None or not np.isfinite(p[:jv + 1]).all():
            return False
    return True


keep = [r for r in RUNS if complete(LAST, r)]
skip = [r for r in RUNS if r not in keep]
print(f"{LAST}: {len(keep)}/{len(RUNS)} pilot runs with the full record"
      + (f" (missing / incomplete: {' '.join(f'{r:03d}' for r in skip)})" if skip else ""))
if not keep:
    sys.exit("nothing to plot -- pull and extract the suite first (extend_profiles_all-mods.sh)")
keep.sort(key=lambda r: PAR.v_conv[r])                                  # rows ordered by convergence rate
KT = [int(round(2 * t)) for t in args.times]
T = {(s, r, k): profile(s, r, k) for s in SUITES for r in keep for k in KT}

# ---------------------------------------------------------------- table + printed summary
rows = []
for r in keep:
    for k, t in zip(KT, args.times):
        ref = T[(SUITES[0], r, k)]
        for z in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
            j = int(np.argmin(np.abs(ZG - z)))
            row = {"run": r, "time_myr": t, "depth_km": z}
            for s in SUITES:
                p = T[(s, r, k)]
                row[f"T_{SHORT(s)}"] = np.nan if p is None else p[j]
                if s != SUITES[0]:
                    row[f"dT_{SHORT(s)}-{SHORT(SUITES[0])}"] = np.nan if p is None or ref is None else p[j] - ref[j]
            if len(SUITES) >= 3:
                a, b = T[(LAST, r, k)], T[(SUITES[1], r, k)]
                row[f"dT_{SHORT(LAST)}-{SHORT(SUITES[1])}"] = np.nan if a is None or b is None else a[j] - b[j]
            rows.append(row)
tab = pd.DataFrame(rows)
os.makedirs(OUT_DIR, exist_ok=True)
tab.to_csv(os.path.join(OUT_DIR, f"{TAG}_triplets_dT.csv"), index=False, float_format="%.1f")
dcols = [c for c in tab.columns if c.startswith("dT_")]
for t in args.times:
    print(f"\n{t:g} Myr: dT (C) per run at 20 / 40 / 60 / 80 km")
    for c in dcols:
        print(f"  {c}")
        for r in keep:
            v = [tab[(tab.run == r) & (tab.time_myr == t) & (tab.depth_km == z)][c].values[0] for z in (20, 40, 60, 80)]
            print(f"    run_{r:03d} (v {PAR.v_conv[r]:.1f}): " + "  ".join(f"{x:+6.1f}" for x in v))

# ---------------------------------------------------------------- figure
nr, nc = len(keep), len(KT) + 1
fig, axs = plt.subplots(nr, nc, figsize=(7.2, 1.55 * nr + 0.7), sharey=True, squeeze=False,
                        gridspec_kw=dict(width_ratios=[1] * len(KT) + [0.85]), facecolor="white")
for i, r in enumerate(keep):
    p = PAR.iloc[r]
    for jc, (k, t) in enumerate(zip(KT, args.times)):
        ax = axs[i, jc]
        for s, col in zip(SUITES, COLORS):
            prof = T[(s, r, k)]
            if prof is not None:
                ax.plot(prof, ZG, color=col, lw=1.0)
        ax.axhspan(80, args.zmax, color="0.92", lw=0, zorder=0)        # deep-end validity band
        ax.set_xlim(0, 1000); ax.set_xticks([0, 250, 500, 750])
        if i == 0:
            ax.set_title(f"{t:g} Myr", fontsize=8)
        if i == nr - 1:
            ax.set_xlabel("Slab-top T (" + S.DEG + "C)")
        else:
            ax.tick_params(labelbottom=False)
    ax = axs[i, -1]
    for k, t in zip(KT, args.times):
        a, b = T[(LAST, r, k)], T[(SUITES[1], r, k)]
        if a is not None and b is not None:
            ax.plot(a - b, ZG, color=S.time_color(t), lw=1.0)
    ax.axvline(0, color="0.6", lw=0.6)
    ax.axhspan(80, args.zmax, color="0.92", lw=0, zorder=0)
    if i == 0:
        ax.set_title(f"{SHORT(LAST)} $-$ {SHORT(SUITES[1])}", fontsize=8)
    if i == nr - 1:
        ax.set_xlabel(r"$\Delta$T (" + S.DEG + "C)")
    else:
        ax.tick_params(labelbottom=False)
    axs[i, 0].set_ylabel(f"run_{r:03d}\nDepth (km)", fontsize=7)
    axs[i, 0].text(0.03, 0.04, f"v {p.v_conv:.1f} cm/yr, dip {p.dip_int:.0f}{S.DEG}\n"
                   f"SP {p.age_SP:.0f} / OP {p.age_OP:.0f} Myr\n" + r"$\eta_{UM}$ " + f"{p.eta_UM:.1e}",
                   transform=axs[i, 0].transAxes, fontsize=5.5, va="bottom", ha="left", color="0.3")
axs[0, 0].set_ylim(args.zmax, 0)
# shared x-range for the difference column, symmetric
lim = 0
for r in keep:
    for k in KT:
        a, b = T[(LAST, r, k)], T[(SUITES[1], r, k)]
        if a is not None and b is not None and np.isfinite(a - b).any():
            lim = max(lim, np.nanmax(np.abs((a - b)[ZG <= ZVALID])))
lim = max(10.0, 1.1 * lim)
for i in range(nr):
    axs[i, -1].set_xlim(-lim, lim)
h = [Line2D([0], [0], color=c, lw=1.2, label=s) for s, c in zip(SUITES, COLORS)]
h += [Line2D([0], [0], color=S.time_color(t), lw=1.2, label=f"{t:g} Myr") for t in args.times]
fig.legend(handles=h, loc="lower center", ncol=len(h), fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.subplots_adjust(left=0.1, right=0.98, top=1 - 0.3 / (1.55 * nr + 0.7), bottom=0.75 / (1.55 * nr + 0.7),
                    hspace=0.12, wspace=0.08)
base = os.path.join(OUT_DIR, f"{TAG}_triplets")
for ext in ("pdf", "svg", "png"):
    fig.savefig(f"{base}.{ext}", facecolor="white", **({"dpi": 300} if ext == "png" else {}))
print("\nwrote", base + ".{pdf,svg,png}", "and", base + "_dT.csv")

#!/usr/bin/env python3
"""Version-control check: const-vc-v3ctrl (ASPECT 3.0, spr/112) vs const-vc (ASPECT 2.5, skx/48).

The 8 const-vc-v3ctrl runs are const-vc physics, no heating, run under the 3.0 binary on the
production const-vc-sh queue.  Their slab-top T(z) difference from the const-vc record of the same
run numbers is the ASPECT-version + decomposition effect that would otherwise sit inside every
const-vc-sh vs const-vc pair.  Decision rule (const-vc-v3ctrl/README.md): a few C -> pair
const-vc-sh against the 2.5 const-vc record; comparable to the heating signal (tens of C at the
slab top) -> re-run const-vc under 3.x or pair const-vc-sh only against these 8.

Inputs: analysis/run_XXX/Tprof_{k}.csv of both suites (written by
src/postproc-numerical-mods/extend_profiles_all-mods.sh), plus run-outputs/run_XXX/log.txt.

Outputs (default --out plots/qc-numerical-mods/const-vc-v3ctrl/):
  v3ctrl_vs_constvc_summary.csv    one row per run x time: mean / rms / max|dT|, dT at fixed depths
  v3ctrl_vs_constvc_by_depth.csv   per depth x time: median, max|dT| across runs
  v3ctrl_vs_constvc_logs.csv       per run: version line, MPI ranks, last timestep, wallclock
  dT_vs_depth.png                  dT(z) = v3ctrl - const-vc, one panel per time, all runs + median
  Tprof_per_run.png                small multiples: both suites' T(z) per run, one colour per time

Usage:
  env/bin/python src/qc-numerical-mods/qc_v3ctrl_vs_constvc.py            # pilot-list runs, 1/5/10 Myr
  env/bin/python src/qc-numerical-mods/qc_v3ctrl_vs_constvc.py --times 0.5 1 5 10 --depth-max 80
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "subd-model-runs"

# Fixed categorical order (one colour per requested time, never per run: 8 runs exceed a safe
# categorical palette, so runs are grey + direct label and the median carries the colour).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GREY = "#9a9a94"
INK = "#1a1a19"

FIXED_DEPTHS = (10, 20, 40, 60, 80, 100)


def read_pilot_list(path: Path) -> list[str]:
    runs = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            runs.append(f"{int(line):03d}")
    return runs


def load_tprofs(suite: str, run: str) -> dict[float, pd.DataFrame]:
    """{time_Myr: DataFrame(depth_km, T_C)} for every Tprof_k.csv of a run."""
    d = RUNS / suite / "analysis" / f"run_{run}"
    out = {}
    for f in sorted(d.glob("Tprof_*.csv")):
        df = pd.read_csv(f)
        if df.empty:
            continue
        out[float(df["time_Myr"].iloc[0])] = df[["depth_km", "T_C"]].reset_index(drop=True)
    return out


def closest(profiles: dict[float, pd.DataFrame], t_req: float, tol: float):
    if not profiles:
        return None, None
    t = min(profiles, key=lambda tt: abs(tt - t_req))
    if abs(t - t_req) > tol:
        return None, None
    return t, profiles[t]


def parse_log(path: Path) -> dict:
    info = dict(version="", ranks=np.nan, last_step=np.nan, last_time_Myr=np.nan, wallclock_s=np.nan)
    if not path.exists():
        return info
    txt = path.read_text(errors="replace")
    m = re.search(r"^--\s+\.\s+version\s+([^\n]+)", txt, re.M)
    if m:
        info["version"] = m.group(1).strip()
    m = re.search(r"running with\s+(\d+)\s+MPI process", txt)
    if m:
        info["ranks"] = int(m.group(1))
    steps = re.findall(r"\*\*\* Timestep\s+(\d+):\s+t=([0-9.eE+-]+)\s+years", txt)
    if steps:
        info["last_step"] = int(steps[-1][0])
        info["last_time_Myr"] = float(steps[-1][1]) / 1e6
    wc = re.findall(r"Total wallclock time elapsed including restarts:\s*([0-9.]+)s", txt)
    if wc:
        info["wallclock_s"] = float(wc[-1])
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a-suite", default="const-vc", help="reference suite (2.5 record)")
    ap.add_argument("--b-suite", default="const-vc-v3ctrl", help="suite under test (3.0 control)")
    ap.add_argument("--runs", nargs="*", default=None, help="run ids; default: b-suite run-inputs/pilot-list.txt")
    ap.add_argument("--times", nargs="*", type=float, default=[1.0, 5.0, 10.0], help="times in Myr")
    ap.add_argument("--tol-myr", type=float, default=0.3)
    ap.add_argument("--depth-max", type=float, default=100.0)
    ap.add_argument("--out", default=None, help="output dir (default plots/qc-numerical-mods/<b-suite>)")
    args = ap.parse_args()

    runs = args.runs or read_pilot_list(RUNS / args.b_suite / "run-inputs" / "pilot-list.txt")
    runs = [r.replace("run_", "").zfill(3) for r in runs]
    out = Path(args.out) if args.out else ROOT / "plots" / "qc-numerical-mods" / args.b_suite
    out.mkdir(parents=True, exist_ok=True)
    times = list(args.times)
    label = f"{args.b_suite} - {args.a_suite}"

    # ---- collect dT(z) -----------------------------------------------------
    rows, by_depth = [], {t: {} for t in times}   # by_depth[t][run] = Series(depth -> dT)
    prof = {}                                      # prof[(run, t)] = (t_a, Ta, t_b, Tb)
    missing = []
    for run in runs:
        pa, pb = load_tprofs(args.a_suite, run), load_tprofs(args.b_suite, run)
        if not pb:
            missing.append(run)
        for t in times:
            ta, dfa = closest(pa, t, args.tol_myr)
            tb, dfb = closest(pb, t, args.tol_myr)
            if dfa is None or dfb is None:
                continue
            m = dfa.merge(dfb, on="depth_km", suffixes=("_a", "_b"))
            m = m[m["depth_km"] <= args.depth_max].dropna()
            if m.empty:
                continue
            dT = (m["T_C_b"] - m["T_C_a"]).to_numpy()
            z = m["depth_km"].to_numpy()
            by_depth[t][run] = pd.Series(dT, index=z)
            prof[(run, t)] = (ta, m[["depth_km", "T_C_a"]], tb, m[["depth_km", "T_C_b"]])
            i = int(np.argmax(np.abs(dT)))
            row = dict(run=run, time_req_Myr=t, t_a_Myr=round(ta, 4), t_b_Myr=round(tb, 4),
                       dt_ab_kyr=round((tb - ta) * 1e3, 2), n_depth=len(z),
                       mean_dT_C=round(float(dT.mean()), 3),
                       rms_dT_C=round(float(np.sqrt((dT**2).mean())), 3),
                       max_abs_dT_C=round(float(np.abs(dT).max()), 3),
                       depth_at_max_km=float(z[i]))
            for d in FIXED_DEPTHS:
                j = np.where(np.isclose(z, d))[0]
                row[f"dT_{d}km_C"] = round(float(dT[j[0]]), 3) if j.size else np.nan
            rows.append(row)

    if not rows:
        print(f"no overlapping profiles for {label}; missing b-suite analysis for runs: {missing}")
        print(f"  -> run: src/postproc-numerical-mods/extend_profiles_all-mods.sh {args.b_suite} 0:20 '1,20;10,20' 8")
        return 1
    if missing:
        print(f"[warn] no {args.b_suite} profiles for runs {missing}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "v3ctrl_vs_constvc_summary.csv", index=False)

    bd_rows = []
    for t in times:
        if not by_depth[t]:
            continue
        M = pd.DataFrame(by_depth[t])           # index depth, columns runs
        for z, r in M.iterrows():
            v = r.dropna()
            bd_rows.append(dict(time_req_Myr=t, depth_km=float(z), n_runs=int(v.size),
                                median_dT_C=round(float(v.median()), 3),
                                mean_dT_C=round(float(v.mean()), 3),
                                max_abs_dT_C=round(float(v.abs().max()), 3)))
    by_depth_df = pd.DataFrame(bd_rows)
    by_depth_df.to_csv(out / "v3ctrl_vs_constvc_by_depth.csv", index=False)

    # ---- logs ---------------------------------------------------------------
    log_rows = []
    for run in runs:
        for suite in (args.a_suite, args.b_suite):
            info = parse_log(RUNS / suite / "run-outputs" / f"run_{run}" / "log.txt")
            log_rows.append(dict(run=run, suite=suite, **info))
    logs = pd.DataFrame(log_rows)
    logs.to_csv(out / "v3ctrl_vs_constvc_logs.csv", index=False)

    # ---- figure 1: dT(z) per time -------------------------------------------
    n = len(times)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 5.2), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, t, col in zip(axes, times, SERIES):
        M = pd.DataFrame(by_depth[t]) if by_depth[t] else pd.DataFrame()
        for run in M.columns:
            ax.plot(M[run], M.index, color=GREY, lw=1.0, alpha=0.85)
            zl = M[run].last_valid_index()
            if zl is not None:
                ax.annotate(run, (M[run][zl], zl), fontsize=7, color=GREY,
                            xytext=(3, 0), textcoords="offset points", va="center")
        if not M.empty:
            ax.plot(M.median(axis=1), M.index, color=col, lw=2.0, label="median")
            ax.legend(frameon=False, loc="lower right", fontsize=8)
        ax.axvline(0, color=INK, lw=0.6)
        ax.set_title(f"{t:g} Myr  (n={M.shape[1]})", fontsize=10)
        ax.set_xlabel("ΔT at slab top (°C)")
        ax.grid(True, lw=0.4, alpha=0.4)
    axes[0].set_ylabel("depth (km)")
    axes[0].invert_yaxis()
    fig.suptitle(f"ΔT(z) = {label}", fontsize=11)
    fig.savefig(out / "dT_vs_depth.png", dpi=180)
    plt.close(fig)

    # ---- figure 2: per-run T(z) small multiples -----------------------------
    ncol = 4
    nrow = int(np.ceil(len(runs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.6 * nrow), sharex=True, sharey=True,
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, run in zip(axes, runs):
        for t, col in zip(times, SERIES):
            if (run, t) not in prof:
                continue
            ta, A, tb, B = prof[(run, t)]
            ax.plot(A["T_C_a"], A["depth_km"], color=col, lw=1.6, label=f"{t:g} Myr {args.a_suite}")
            ax.plot(B["T_C_b"], B["depth_km"], color=col, lw=1.2, ls="--", label=f"{t:g} Myr {args.b_suite}")
        ax.set_title(f"run_{run}", fontsize=10)
        ax.grid(True, lw=0.4, alpha=0.4)
    for ax in axes[len(runs):]:
        ax.axis("off")
    axes[0].invert_yaxis()
    for ax in axes[-ncol:]:
        ax.set_xlabel("T (°C)")
    for ax in axes[::ncol]:
        ax.set_ylabel("depth (km)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"slab-top T(z): solid {args.a_suite}, dashed {args.b_suite}", fontsize=11)
    fig.savefig(out / "Tprof_per_run.png", dpi=180)
    plt.close(fig)

    # ---- stdout ---------------------------------------------------------------
    pd.set_option("display.width", 200)
    print(f"\n== {label}: per time, across runs (depth <= {args.depth_max:g} km)")
    g = summary.groupby("time_req_Myr")
    print(pd.DataFrame({
        "n_runs": g.size(),
        "median_mean_dT_C": g["mean_dT_C"].median().round(2),
        "median_rms_dT_C": g["rms_dT_C"].median().round(2),
        "worst_max_abs_dT_C": g["max_abs_dT_C"].max().round(2),
        "median_|dT|_40km": g["dT_40km_C"].apply(lambda s: s.abs().median()).round(2),
        "median_|dT|_80km": g["dT_80km_C"].apply(lambda s: s.abs().median()).round(2),
    }).to_string())
    print("\n== per run")
    print(summary[["run", "time_req_Myr", "dt_ab_kyr", "mean_dT_C", "rms_dT_C", "max_abs_dT_C",
                   "depth_at_max_km", "dT_20km_C", "dT_40km_C", "dT_60km_C", "dT_80km_C"]].to_string(index=False))
    print("\n== logs")
    print(logs.to_string(index=False))
    print(f"\nwrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

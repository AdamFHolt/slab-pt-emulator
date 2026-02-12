#!/usr/bin/env python3
"""
QC plot: vertical temperature profiles (depth vs T) for all models,
for three requested times in Myr (default: 0, 5, 10).

Input format (per-run file):
    time_Myr, depth_km, T_C
    5.0010,   0.0,     -0.15
    5.0010,   1.0,      3.10
    ...

Typical directory layout:
    ../../subd-model-runs/const-vc/analysis/run_010/Tprof_10.csv
    ../../subd-model-runs/const-vc/analysis/run_011/Tprof_11.csv
    ...

Usage:
  python qc_T_profiles_allmodels_3times.py \
    --glob "../../subd-model-runs/const-vc/analysis/run_*/Tprof_*.csv" \
    --times 0 5 10 \
    --out "../../plots/qc-numerical-mods/const-vc/Tprofiles_t0t5t10"

Notes:
- For each run and each requested time, we choose the profile whose time_Myr
  is closest to the requested time (within --tol-myr).
- If your simulation never reaches a requested time, that run just won't
  contribute to that panel.
"""

import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# font setup
import matplotlib as mpl
import matplotlib.font_manager as fm
font_path = "/home/holt/.local/share/fonts/MYRIADPRO-REGULAR.OTF"
myriad_pro = fm.FontProperties(fname=font_path)
mpl.rcParams['font.family'] = 'Myriad Pro'  
mpl.rcParams['font.size'] = 11.5
mpl.rcParams['axes.labelsize'] = 11.5
mpl.rcParams['axes.labelpad'] = 1.5
mpl.rcParams['xtick.labelsize'] = 9.75
mpl.rcParams['ytick.labelsize'] = 9.75
mpl.rcParams['xtick.major.pad'] = 2
mpl.rcParams['ytick.major.pad'] = 2
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['xtick.minor.size'] = 1.5
mpl.rcParams['ytick.minor.size'] = 1.5

def extract_run_id(path: Path) -> str:
    """
    Try to extract run_id from path like .../run_010/...,
    else fall back to something based on filename stem.
    """
    s = str(path)
    m = re.search(r"/run_(\d+)(/|$)", s)
    if m:
        return m.group(1)  # already zero-padded
    # fallback: find digits in filename
    m2 = re.search(r"(\d+)", path.stem)
    if m2:
        return m2.group(1)
    return path.stem


def read_profile_csv(f: Path) -> pd.DataFrame:
    df = pd.read_csv(f)
    required = {"time_Myr", "depth_km", "T_C"}
    if not required.issubset(df.columns):
        raise ValueError(f"{f} missing required columns {required}. Found {list(df.columns)}")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["time_Myr", "depth_km", "T_C"])
    df["time_Myr"] = df["time_Myr"].astype(float)
    df["depth_km"] = df["depth_km"].astype(float)
    df["T_C"] = df["T_C"].astype(float)

    # If depths are negative (unlikely here), flip sign
    if np.nanmedian(df["depth_km"]) < 0:
        df["depth_km"] = -df["depth_km"]

    return df


def pick_time_slice(df: pd.DataFrame, target_time: float, tol: float) -> pd.DataFrame | None:
    """
    From df with many time_Myr values, pick rows at the time closest to target_time.
    Return None if closest is farther than tol.
    """
    times = df["time_Myr"].to_numpy(float)
    if times.size == 0:
        return None
    uniq = np.unique(times)
    idx = int(np.argmin(np.abs(uniq - target_time)))
    tsel = float(uniq[idx])
    if abs(tsel - target_time) > tol:
        return None
    out = df[np.isclose(df["time_Myr"], tsel)].copy()
    out["time_selected_Myr"] = tsel
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", required=True,
                   help='Glob for per-run profile CSVs, e.g. "../../subd-model-runs/const-vc/analysis/run_*/Tprof_*.csv"')
    p.add_argument("--times", type=float, nargs="+", default=[0.0, 5.0, 10.0],
                   help="Requested times in Myr (default: 0 5 10)")
    p.add_argument("--tol-myr", type=float, default=0.30,
                   help="Tolerance for matching a requested time (default 0.30 Myr)")
    p.add_argument("--out", required=True)
    p.add_argument("--dpi", type=int, default=220)

    p.add_argument("--max-lines", type=int, default=None,
                   help="Optional cap on number of individual profiles per panel (default: plot all)")
    p.add_argument("--alpha", type=float, default=0.2,
                   help="Alpha for individual profiles (default 0.2)")
    p.add_argument("--Tlim", nargs=2, type=float, default=None,
                   help="Optional x-limits: Tmin Tmax (°C)")
    p.add_argument("--zlim", nargs=2, type=float, default=None,
                   help="Optional y-limits: zmin zmax (km)")
    args = p.parse_args()

    out_prefix = Path(args.out).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    files = sorted([Path(x) for x in Path(".").glob(args.glob)] if ("*" not in args.glob and "?" not in args.glob and "[" not in args.glob)
                   else [Path(p) for p in __import__("glob").glob(args.glob)])
    files = [Path(f) for f in files]

    if not files:
        raise SystemExit(f"No files matched --glob: {args.glob}")

    # Collect data: dict[time]-> list of (run_id, depth_array, T_array, time_selected)
    times_req = [float(t) for t in args.times]
    per_time = {t: [] for t in times_req}

    bad = 0
    for f in files:
        try:
            run_id = extract_run_id(f)
            df = read_profile_csv(f)

            for t in times_req:
                sl = pick_time_slice(df, t, args.tol_myr)
                if sl is None:
                    continue
                # enforce sorted depth
                sl = sl.sort_values("depth_km")
                per_time[t].append(
                    (run_id,
                     sl["depth_km"].to_numpy(float),
                     sl["T_C"].to_numpy(float),
                     float(sl["time_selected_Myr"].iloc[0]))
                )
        except Exception:
            bad += 1
            continue

    # Determine global limits if not specified
    all_T = []
    all_z = []
    for t in times_req:
        for _, z, T, _ in per_time[t]:
            all_T.append(T)
            all_z.append(z)

    if all_T:
        all_T = np.concatenate(all_T)
        all_z = np.concatenate(all_z)
        if args.Tlim:
            Tmin, Tmax = args.Tlim
        else:
            Tmin = float(np.nanpercentile(all_T, 1))
            Tmax = float(np.nanpercentile(all_T, 99))
        if args.zlim:
            zmin, zmax = args.zlim
        else:
            zmin = float(np.nanmin(all_z))
            zmax = float(np.nanmax(all_z))
    else:
        Tmin, Tmax = 0.0, 1400.0
        zmin, zmax = 0.0, 200.0

    # Plot
    fig, axes = plt.subplots(1, len(times_req),
                             figsize=(4.6 * len(times_req) + 1.4, 5.2),
                             constrained_layout=True)
    if len(times_req) == 1:
        axes = [axes]

    rng = np.random.default_rng(0)

    for ax, t in zip(axes, times_req):
        entries = per_time[t]
        if not entries:
            ax.set_title(f"time ~ {t:g} Myr (no matches)")
            ax.axis("off")
            continue

        # Choose subset for plotting individual profiles (if huge)
        if args.max_lines is not None and args.max_lines > 0 and len(entries) > args.max_lines:
            idx = rng.choice(len(entries), size=args.max_lines, replace=False)
            plot_entries = [entries[i] for i in idx]
        else:
            plot_entries = entries

        # Build an interpolated matrix onto a common depth grid for stats
        # Use union depth grid from first 1–2 profiles (then refine to min/max)
        # For robustness: define grid by global min/max depth and 1 km spacing if possible.
        z_lo = max(zmin, min(np.nanmin(z) for _, z, _, _ in entries))
        z_hi = min(zmax, max(np.nanmax(z) for _, z, _, _ in entries))
        # 1 km grid if range is sensible
        dz = 1.0
        zgrid = np.arange(z_lo, z_hi + 0.5*dz, dz)

        M = []
        for _, z, T, _tsel in entries:
            # interpolate onto zgrid (skip if too short)
            if len(z) < 5:
                continue
            # remove duplicates in z
            zz, ii = np.unique(z, return_index=True)
            TT = T[ii]
            if len(zz) < 5:
                continue
            Ti = np.interp(zgrid, zz, TT, left=np.nan, right=np.nan)
            M.append(Ti)

        M = np.array(M) if M else None

        # Plot individual profiles
        for _, z, T, tsel in plot_entries:
            ax.plot(T, z, lw=0.8, alpha=args.alpha, color="0.65")

        # Stats overlays
        if M is not None and M.size > 0:
            med = np.nanmedian(M, axis=0)
            ax.plot(med, zgrid, lw=2, alpha=1.0, color="tab:blue")

            p05 = np.nanpercentile(M, 5, axis=0)
            p95 = np.nanpercentile(M, 95, axis=0)
            ax.fill_betweenx(zgrid, p05, p95, alpha=0.22, color="tab:blue")

        # Title with actual selected time (median of selected times)
        t_selecteds = [tsel for *_rest, tsel in entries]
        tmed = float(np.nanmedian(np.array(t_selecteds)))
        ax.set_title(f"time = {tmed:.2f} Myr")

        ax.set_xlabel("Temperature (°C)")
        ax.grid(True, ls=":", alpha=0.35)
        ax.set_xlim(Tmin, Tmax)
        ax.set_ylim(zmin, zmax)
        ax.invert_yaxis()

    axes[0].set_ylabel("Depth (km)")
    fig.savefig(f"{out_prefix}.png", dpi=args.dpi, bbox_inches="tight")
    print("Saved:", f"{out_prefix}.png")
    if bad:
        print(f"Note: skipped {bad} files due to read/parse errors.")


if __name__ == "__main__":
    main()

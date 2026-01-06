#!/usr/bin/env python3
"""
QC histograms: ΔT/Δt (dTdt_C_per_Myr) for multiple depths from the hierarchical master file.

Master format (one file, all depths):
    depth_km,run_id,T1_C,T2_C,dT_C,dt_Myr,dTdt_C_per_Myr

Example:
  python hist_dTdt_3depths.py \
      --params ../../data/params/params-list.const-vc.csv \
      --master ../../subd-model-runs/const-vc/analysis/master_DT1-10.csv \
      --depths 20 50 80 \
      --y dTdt_C_per_Myr \
      --out ../../plots/qc-numerical-mods/const-vc_DT1-10_hist-dTdt_20km50km80km
"""

import argparse
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


def zero_pad_runids(n: int):
    """Return ['000', '001', ...] with minimum width 3 (or larger if n>=1000)."""
    width = max(3, len(str(n - 1)))
    return [f"{i:0{width}d}" for i in range(n)]


def nice_label_y(yname: str) -> str:
    if yname == "dTdt_C_per_Myr":
        return r"$\Delta T / \Delta t$ (°C/Myr)"
    elif yname == "dT_C":
        return r"$\Delta T$ (°C)"
    else:
        return yname


def main():
    p = argparse.ArgumentParser(
        description="Plot histograms of dT or dT/dt for multiple depths "
                    "from the hierarchical master file."
    )
    p.add_argument("--params", required=True,
                   help="Path to params-list.<suite>.csv (for run_id alignment).")
    p.add_argument("--master", required=True,
                   help="Hierarchical master CSV with depth_km column.")
    p.add_argument("--depths", nargs="+", type=float, required=True,
                   help="Depths (km) to plot, e.g. 20 50 80.")
    p.add_argument("--y", default="dTdt_C_per_Myr",
                   choices=["dT_C", "dTdt_C_per_Myr"],
                   help="Quantity to histogram.")
    p.add_argument("--out", required=True,
                   help="Output path prefix (without extension).")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    params_path = Path(args.params).resolve()
    master_path = Path(args.master).resolve()
    out_prefix = Path(args.out).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    depths = args.depths
    yvar = args.y

    # --- Load params (for run_id space) ---
    df_p = pd.read_csv(params_path)
    df_p["run_id"] = zero_pad_runids(len(df_p))

    # --- Load master (hierarchical) ---
    df_m_all = pd.read_csv(master_path, dtype={"run_id": str})
    if "depth_km" not in df_m_all.columns:
        raise ValueError(
            f"Expected a 'depth_km' column in master file {master_path}, "
            "but did not find one."
        )

    # Merge to ensure we only keep runs present in the param LHS design
    df_all = pd.merge(df_p[["run_id"]], df_m_all, on="run_id", how="inner")

    # --- Figure layout: one panel per depth ---
    n = len(depths)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    y_label = nice_label_y(yvar)

    for ax, depth in zip(axes, depths):
        # Filter to this depth (robust float comparison)
        mask = np.isclose(df_all["depth_km"].to_numpy(float), depth)
        df_d = df_all[mask]

        if df_d.empty:
            print(f"[WARN] No rows found in master for depth_km = {depth}")
            ax.text(0.5, 0.5, f"No data @ {depth} km",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue

        yvals = df_d[yvar].to_numpy(float)
        yvals = yvals[np.isfinite(yvals)]

        ax.hist(yvals, bins=30, alpha=0.85, color="skyblue")
        ax.set_title(f"{y_label} @ {depth:.0f} km")
        ax.set_xlabel(y_label)
        ax.set_ylabel("count")
        ax.grid(True, ls=":", alpha=0.4)

    fig.savefig(f"{out_prefix}.png", dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out_prefix}.png")


if __name__ == "__main__":
    main()


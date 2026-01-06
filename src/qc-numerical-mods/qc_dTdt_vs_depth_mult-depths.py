#!/usr/bin/env python3
"""
QC: dT or dT/dt vs *run index* for multiple depths, from the hierarchical master.

Master format:

    depth_km,run_id,T1_C,T2_C,dT_C,dt_Myr,dTdt_C_per_Myr

Example:

  python qc_dt_vs_depth_3depths.py \
      --master ../../subd-model-runs/const-vc/analysis/master_DT1-10.csv \
      --depths 20 50 80 \
      --y dTdt_C_per_Myr \
      --out ../../plots/qc-numerical-mods/const-vc_DT1-10_dTdt-vs-depth-panels
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


def nice_y_label(yvar: str) -> str:
    if yvar == "dTdt_C_per_Myr":
        return r"$\Delta T / \Delta t$ (°C/Myr)"
    elif yvar == "dT_C":
        return r"$\Delta T$ (°C)"
    else:
        return yvar


def main():
    p = argparse.ArgumentParser(
        description="Panel plots of ΔT or dT/dt vs run index for multiple depths "
                    "using the hierarchical master CSV."
    )
    p.add_argument("--master", required=True,
                   help="Hierarchical master CSV with a 'depth_km' column.")
    p.add_argument("--depths", type=float, nargs="+", required=True,
                   help="Depths (km) to extract, e.g. 20 50 80.")
    p.add_argument("--y", default="dTdt_C_per_Myr",
                   choices=["dT_C", "dTdt_C_per_Myr"],
                   help="Quantity to plot on y-axis.")
    p.add_argument("--out", required=True,
                   help="Output path prefix (without extension).")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    master_path = Path(args.master).resolve()
    out_prefix = Path(args.out).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    depths = args.depths
    yvar = args.y
    ylab = nice_y_label(yvar)

    # Load master (one big file)
    df_all = pd.read_csv(master_path, dtype={"run_id": str})
    if "depth_km" not in df_all.columns:
        raise ValueError(
            f"Expected 'depth_km' column in {master_path}, but did not find one."
        )

    n = len(depths)
    fig, axes = plt.subplots(
        1, n,
        figsize=(4.0 * n, 3.6),
        constrained_layout=True
    )
    if n == 1:
        axes = [axes]

    for ax, depth in zip(axes, depths):
        df_d = df_all[np.isclose(df_all["depth_km"], depth)].copy()
        if df_d.empty:
            raise ValueError(
                f"No rows found in {master_path} for depth_km = {depth}."
            )

        y = df_d[yvar].to_numpy(float)
        x = np.arange(len(y))

        ax.scatter(x, y, s=14, alpha=0.8)
        ax.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.set_title(f"{depth:.0f} km", fontsize=11)
        ax.set_xlabel("run #")
        ax.grid(True, ls=":", alpha=0.4)

        # Use common y-limits (as we did before)
        if yvar == "dTdt_C_per_Myr":
            ax.set_ylim(-160.0, 0.0)

    # Put y-label on the leftmost axis only
    axes[0].set_ylabel(ylab)

    fig.savefig(f"{out_prefix}.png", dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out_prefix}.png")


if __name__ == "__main__":
    main()


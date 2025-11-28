#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(
        description="QC: scatter of dT or dT/dt vs run index, for one or more depths "
                    "from the hierarchical master (with depth_km column)."
    )
    p.add_argument(
        "--master",
        required=True,
        help="Hierarchical master CSV with depth_km column "
             "(e.g., ../../subd-model-runs/const-vc/analysis/master_DT1-10.csv)",
    )
    p.add_argument(
        "--depths",
        type=float,
        nargs="*",
        help="Depths (km) to include (e.g. 20 50 80). "
             "If omitted, uses all unique depth_km values in the master.",
    )
    p.add_argument(
        "--y",
        default="dTdt_C_per_Myr",
        choices=["dT_C", "dTdt_C_per_Myr"],
        help="Quantity to plot on y-axis.",
    )
    p.add_argument(
        "--out",
        default="../../plots/qc-numerical-mods/qc_all-DT",
        help="Output path prefix (without extension).",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI.",
    )
    args = p.parse_args()

    master_path = Path(args.master).resolve()
    out_prefix = Path(args.out).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # --- Load master ---
    df = pd.read_csv(master_path, dtype={"run_id": str})

    if "depth_km" not in df.columns:
        raise ValueError(
            f"Expected a 'depth_km' column in master file {master_path}, "
            "but did not find one."
        )

    # Determine which depths to use
    if args.depths and len(args.depths) > 0:
        depths = list(args.depths)
    else:
        depths = sorted(df["depth_km"].unique())

    n = len(depths)
    if n == 0:
        raise ValueError("No depths found / selected in master file.")

    fig, axes = plt.subplots(
        1, n, figsize=(4.0 * n, 3.6), constrained_layout=True
    )
    if n == 1:
        axes = [axes]

    for ax, depth in zip(axes, depths):
        mask = np.isclose(df["depth_km"], depth)
        df_d = df[mask].copy()

        if df_d.empty:
            ax.set_title(f"{args.y} @ {depth:.0f} km (no data)")
            ax.axis("off")
            continue

        y = df_d[args.y].to_numpy(dtype=float)
        idx = np.arange(len(y))

        ax.scatter(idx, y, s=14, c='black', alpha=0.75)
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.set_title(f"{args.y} @ {depth:.0f} km")
        ax.set_xlabel("run index")
        ax.set_ylim(bottom=-160)
        ax.set_ylabel(args.y)
        ax.grid(True, ls=":", alpha=0.4)

    fig.savefig(f"{out_prefix}.png", dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out_prefix}.png")


if __name__ == "__main__":
    main()


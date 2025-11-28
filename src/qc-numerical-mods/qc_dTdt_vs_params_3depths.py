#!/usr/bin/env python3
"""
QC plot: dT or dT/dt vs parameters (+ thermal parameter) for *multiple depths*.

Uses the hierarchical master file format:

    depth_km,run_id,T1_C,T2_C,dT_C,dt_Myr,dTdt_C_per_Myr

Usage example:

  python qc_cooling-rates_all-mods_3depths.py \
      --params ../../data/params/params-list.csv \
      --master ../../subd-model-runs/const-vc/analysis/master_DT1-10.csv \
      --depths 25 50 75 \
      --y dTdt_C_per_Myr \
      --out ./plots/qc_cooling-rates_25-50-75km_DT1-10
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOG_PARAMS = {"eta_int", "eta_UM", "eps_trans"}  # log10 for these


def zero_pad_runids(n: int):
    """Return ['000', '001', ...,] with minimum width 3 (or larger if n>=1000)."""
    width = max(3, len(str(n - 1)))
    return [f"{i:0{width}d}" for i in range(n)]


def nice_label(param: str) -> str:
    labels = {
        "v_conv": r"Convergence rate (cm/yr)",
        "age_SP": r"Age$_{\rm SP}$ (Ma)",
        "age_OP": r"Age$_{\rm OP}$ (Ma)",
        "dip_int": r"Initial dip (°)",
        "eta_int": r"$\eta_{\rm int}$ (Pa·s)",
        "eta_UM": r"$\eta_{\rm UM}$ (Pa·s)",
        "eps_trans": r"$\dot\epsilon_{\rm trans}$ (s$^{-1}$)",
        "thermal_param": r"$v\; \mathrm{age}_{\rm SP}\; {\sin(\mathrm{dip})}$ (km)",
        "dT_C": r"$\Delta T$ (°C)",
        "dTdt_C_per_Myr": r"$\Delta T/\Delta t$ (°C/Myr)",
    }
    return labels.get(param, param)


def compute_thermal_param(df: pd.DataFrame) -> np.ndarray:
    """v_conv * age_SP * sin(dip) in km (arbitrary units)."""
    v = df["v_conv"].to_numpy(float) / 1e3      # cm/yr -> km/yr
    age = df["age_SP"].to_numpy(float) * 1e6    # Myr -> yr
    dip_rad = np.deg2rad(df["dip_int"].to_numpy(float))
    tp = v * np.maximum(age, 0.0) * np.sin(np.clip(dip_rad, 0.0, np.pi / 2))
    return tp


def main():
    p = argparse.ArgumentParser(
        description="Plot ΔT or dT/dt vs parameters (+ thermal parameter) "
                    "for multiple depths from the hierarchical master."
    )
    p.add_argument("--params", required=True,
                   help="Path to params-list.csv (LHS design).")
    p.add_argument("--master", required=True,
                   help="Hierarchical master CSV with depth_km column.")
    p.add_argument("--depths", type=float, nargs="+", required=True,
                   help="Depths (km) to extract (e.g. 25 50 75).")
    p.add_argument("--y", default="dTdt_C_per_Myr",
                   choices=["dT_C", "dTdt_C_per_Myr"],
                   help="Quantity to plot on y-axis.")
    p.add_argument("--out", required=True,
                   help="Output path prefix (without extension).")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    params_path = Path(args.params).resolve()
    master_path = Path(args.master).resolve()
    out_prefix = Path(args.out).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    depths = args.depths

    # 1) Load params and master
    df_p = pd.read_csv(params_path)
    # make a string run_id column consistent with run_000 style
    df_p["run_id"] = zero_pad_runids(len(df_p))

    df_m_all = pd.read_csv(master_path, dtype={"run_id": str})
    if "depth_km" not in df_m_all.columns:
        raise ValueError(
            f"Expected a 'depth_km' column in master file {master_path}, "
            "but did not find one."
        )

    # Prepare per-depth merged dataframes
    depth_dfs = []
    for depth in depths:
        df_m = df_m_all[np.isclose(df_m_all["depth_km"], depth)].copy()
        if df_m.empty:
            raise ValueError(
                f"No rows found in {master_path} for depth_km = {depth}."
            )

        df = pd.merge(df_p, df_m, on="run_id", how="inner")
        if df.empty:
            raise ValueError(
                f"Merge of params and master resulted in empty DataFrame "
                f"for depth_km = {depth}."
            )

        df["thermal_param"] = compute_thermal_param(df)
        df["depth_km"] = depth
        depth_dfs.append(df)

    # 2) Plot
    base_params_all = ["v_conv", "age_SP", "age_OP", "dip_int", "eta_int", "eta_UM", "eps_trans"]
    # only keep parameters that actually exist in df_p
    base_params = [c for c in base_params_all if c in df_p.columns]
    params = base_params + ["thermal_param"]
    n = len(params)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.6 * nrows), constrained_layout=True)
    axes = axes.flatten() if n > 1 else [axes]
    yvar = args.y
    ylab = nice_label(yvar)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    depth_labels = [f"{d:.0f} km" for d in depths]

    for i, pname in enumerate(params):
        ax = axes[i]

        use_logx = pname in LOG_PARAMS

        for d_idx, (depth, df) in enumerate(zip(depths, depth_dfs)):
            color = colors[d_idx % len(colors)]
            label = depth_labels[d_idx]

            x = df[pname].to_numpy(dtype=float)
            yvals = df[yvar].to_numpy(dtype=float)
            msk = np.isfinite(x) & np.isfinite(yvals)

            Xplot = x[msk]
            Yplot = yvals[msk]

            ax.scatter(
                np.log10(Xplot) if use_logx else Xplot,
                Yplot,
                s=18,
                alpha=0.75,
                label=label if i == 0 else None,  # only label once for legend
                color=color,
            )

        # If plotting thermal parameter, use log axis (but not log10 transform)
        if pname == "thermal_param":
            ax.set_xscale("log")

        # Labels and formatting
        xlab = nice_label(pname) + (" (log₁₀)" if use_logx else "")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(True, ls=":", alpha=0.4)
        ax.set_title(f"{pname}", fontsize=10)
        # You can keep or relax this y-limit depending on the quantity
        if yvar == "dTdt_C_per_Myr":
            ax.set_ylim(bottom=-160.0, top=0)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Add legend to the first used axis
    axes[0].legend(frameon=False, fontsize=9, title="Depth")

    # Save
    fig.savefig(f"{out_prefix}.png", dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out_prefix}.png")


if __name__ == "__main__":
    main()

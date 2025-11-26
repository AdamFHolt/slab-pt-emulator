#!/usr/bin/env python3
"""
QC plot: dT or dT/dt vs parameters (+ thermal parameter) for a *single depth*.

Now adapted for the hierarchical master file format:

    depth_km,run_id,T1_C,T2_C,dT_C,dt_Myr,dTdt_C_per_Myr

Usage example:

  python qc_cooling-rates_all-mods.py \
      --params ../../data/params/params-list.csv \
      --master ../../subd-model-runs/const-vc/analysis/master_DT1-10.csv \
      --depth-km 50 \
      --y dTdt_C_per_Myr \
      --out ./plots/qc_cooling-rates_50km_DT1-10

This will:
  - take rows where depth_km == 50,
  - merge with params via run_id,
  - plot y vs each parameter (+ thermal_param).
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
                    "for a single depth from the hierarchical master."
    )
    p.add_argument("--params", required=True,
                   help="Path to params-list.csv (LHS design).")
    p.add_argument("--master", required=True,
                   help="Hierarchical master CSV with depth_km column.")
    p.add_argument("--depth-km", type=float, required=True,
                   help="Depth (km) to extract from master (e.g. 25, 50, 75).")
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

    depth = args.depth_km

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

    # Filter to single depth
    df_m = df_m_all[np.isclose(df_m_all["depth_km"], depth)].copy()
    if df_m.empty:
        raise ValueError(
            f"No rows found in {master_path} for depth_km = {depth}."
        )

    # 2) Merge and add derived thermal parameter
    df = pd.merge(df_p, df_m, on="run_id", how="inner")
    if df.empty:
        raise ValueError("Merge of params and master resulted in empty DataFrame.")

    df["thermal_param"] = compute_thermal_param(df)

    # 3) Plot
    base_params_all = ["v_conv", "age_SP", "age_OP", "dip_int", "eta_int", "eta_UM", "eps_trans"]
    base_params = [c for c in base_params_all if c in df.columns]
    params = base_params + ["thermal_param"]
    n = len(params)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.6 * nrows), constrained_layout=True)
    axes = axes.flatten() if n > 1 else [axes]
    yvar = args.y
    ylab = nice_label(yvar)

    yvals_all = df[yvar].to_numpy(dtype=float)
    run_ids_all = df["run_id"].to_numpy(str)

    for i, pname in enumerate(params):
        ax = axes[i]

        x = df[pname].to_numpy(dtype=float)
        yvals = yvals_all
        run_ids = run_ids_all

        msk = np.isfinite(x) & np.isfinite(yvals)
        use_logx = pname in LOG_PARAMS
        Xplot = x[msk]
        Yplot = yvals[msk]
        RID = run_ids[msk]

        ax.scatter(
            np.log10(Xplot) if use_logx else Xplot,
            Yplot,
            s=18,
            c="black",
            alpha=0.75,
        )

        # If plotting thermal parameter, use log axis (but not log10 transform)
        if pname == "thermal_param":
            ax.set_xscale("log")

        # # Annotate “suspect” points (ΔT/dt > −20 °C/Myr or ΔT > −100 °C, as before)
        # if yvar == "dTdt_C_per_Myr":
        #     thresh = -20.0
        # else:
        #     thresh = -100.0

        # for xi, yi, rid in zip(Xplot, Yplot, RID):
        #     if yi > thresh:
        #         ax.text(
        #             (np.log10(xi) if use_logx else xi),
        #             yi,
        #             rid,
        #             fontsize=7,
        #             ha="left",
        #             va="center",
        #             color="red",
        #         )

        # Labels and formatting
        xlab = nice_label(pname) + (" (log₁₀)" if use_logx else "")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(True, ls=":", alpha=0.4)
        ax.set_title(f"{pname} @ {depth:.0f} km", fontsize=10)
        ax.set_ylim(bottom=-160.0, top=0)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Save
    fig.savefig(f"{out_prefix}.png", dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out_prefix}.png")


if __name__ == "__main__":
    main()

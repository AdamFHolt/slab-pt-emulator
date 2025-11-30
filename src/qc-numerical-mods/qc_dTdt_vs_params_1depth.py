#!/usr/bin/env python3
"""
QC plot: dT or dT/dt vs parameters (+ thermal parameter) for a *single depth*.

Uses the hierarchical master file format:

    depth_km,run_id,T1_C,T2_C,dT_C,dt_Myr,dTdt_C_per_Myr

Example:

  python qc_dTdt_vs_params_1depth.py \
      --params ../../data/params/params-list.const-vc.csv \
      --master ../../subd-model-runs/const-vc/analysis/master_DT1-10.csv \
      --depth-km 50 \
      --y dTdt_C_per_Myr \
      --out ../../plots/qc-numerical-mods/const-vc_DT1-10_dTdt-vs-params_50km

Behavior:
  - If params file includes `t_conv`, we also plot:
      * t_conv
      * v_conv_over_tconv = v_conv / t_conv
    and use a 4-column grid.
  - Otherwise, no extra panels and we use a 3-column grid.
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
        "v_conv_over_tconv": r"$v_{\rm conv}/t_{\rm conv}$ (cm/yr/Myr)",
        "t_conv": r"$t_{\rm conv}$ (Myr)",
        "age_SP": r"Age$_{\rm SP}$ (Ma)",
        "age_OP": r"Age$_{\rm OP}$ (Ma)",
        "dip_int": r"Initial dip (°)",
        "eta_int": r"$\eta_{\rm int}$ (Pa·s)",
        "eta_UM": r"$\eta_{\rm UM}$ (Pa·s)",
        "eps_trans": r"$\dot\epsilon_{\rm trans}$ (s$^{-1}$)",
        "thermal_param": r"$v\; \mathrm{age}_{\rm SP}\; \sin(\mathrm{dip})$ (km)",
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
                   help="Path to params-list.<suite>.csv (LHS design).")
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

    # ------------------------------------------------------------
    # 1) Load params and master
    # ------------------------------------------------------------
    df_p = pd.read_csv(params_path)
    df_p["run_id"] = zero_pad_runids(len(df_p))

    # If t_conv exists, compute v_conv_over_tconv
    ramped = "t_conv" in df_p.columns
    if ramped:
        t = df_p["t_conv"].replace(0, np.nan).to_numpy(float)
        v = df_p["v_conv"].to_numpy(float)
        df_p["v_conv_over_tconv"] = v / t

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

    # Merge and add thermal_param
    df = pd.merge(df_p, df_m, on="run_id", how="inner")
    if df.empty:
        raise ValueError("Merge of params and master resulted in empty DataFrame.")
    df["thermal_param"] = compute_thermal_param(df)

    # ------------------------------------------------------------
    # 2) Build parameter list (x-axes)
    # ------------------------------------------------------------
    base_params = []

    # Always include v_conv if present
    if "v_conv" in df.columns:
        base_params.append("v_conv")

    # If ramped, also include v_conv_over_tconv and t_conv
    if ramped:
        if "v_conv_over_tconv" in df.columns:
            base_params.append("v_conv_over_tconv")
        if "t_conv" in df.columns:
            base_params.append("t_conv")

    # Standard parameters
    for name in ["age_SP", "age_OP", "dip_int", "eta_int", "eta_UM", "eps_trans"]:
        if name in df.columns:
            base_params.append(name)

    params = base_params + ["thermal_param"]

    # Grid: 4 columns if ramped (extra panels), else 3
    if ramped:
        ncols = 4
    else:
        ncols = 3

    n = len(params)
    ncols = min(ncols, n)  # in case very few params
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4*ncols, 3.6*nrows),
        constrained_layout=True
    )
    axes = axes.flatten() if n > 1 else [axes]

    yvar = args.y
    ylab = nice_label(yvar)

    yvals_all = df[yvar].to_numpy(dtype=float)
    run_ids_all = df["run_id"].to_numpy(str)

    # ------------------------------------------------------------
    # 3) Make the panels
    # ------------------------------------------------------------
    for i, pname in enumerate(params):
        ax = axes[i]

        x = df[pname].to_numpy(dtype=float)
        yvals = yvals_all

        msk = np.isfinite(x) & np.isfinite(yvals)
        use_logx = pname in LOG_PARAMS

        Xplot = x[msk]
        Yplot = yvals[msk]

        ax.scatter(
            np.log10(Xplot) if use_logx else Xplot,
            Yplot,
            s=18,
            c="black",
            alpha=0.75,
        )

        if pname == "thermal_param":
            ax.set_xscale("log")

        xlab = nice_label(pname) + (" (log₁₀)" if use_logx else "")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(True, ls=":", alpha=0.4)
        ax.set_title(f"{pname} @ {depth:.0f} km", fontsize=10)

        if yvar == "dTdt_C_per_Myr":
            ax.set_ylim(bottom=-160.0, top=0)

    # Hide unused axes if grid larger than number of params
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.savefig(f"{out_prefix}.png", dpi=args.dpi, bbox_inches="tight")
    print(f"Saved:", f"{out_prefix}.png")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOG_PARAMS = {"eta_int", "eta_UM", "eps_trans"}  # log10 for these

def zero_pad_runids(n):
    width = max(3, len(str(n - 1)))
    return [f"{i:0{width}d}" for i in range(n)]

def nice_label(param):
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

def corr_text(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return "n/a"
    r = np.corrcoef(x[mask], y[mask])[0, 1]
    return f"r = {r:.2f}"

def compute_thermal_param(df):
    """v_conv * age_SP * sin(dip) in km (arbitrary units)."""
    v = df["v_conv"].to_numpy(float) / 1e3     # cm/yr -> km/yr
    age = df["age_SP"].to_numpy(float) * 1e6   # Myr -> yr
    dip_rad = np.deg2rad(df["dip_int"].to_numpy(float))
    tp = v * np.maximum(age, 0.0) * np.sin(np.clip(dip_rad, 0.0, np.pi / 2))
    return tp

def main():
    p = argparse.ArgumentParser(description="Plot ΔT or dT/dt vs parameters (+ thermal parameter).")
    p.add_argument("--params", required=True)
    p.add_argument("--master1", required=True)
    p.add_argument("--master2", required=True)
    p.add_argument("--y", default="dT_C", choices=["dT_C", "dTdt_C_per_Myr"])
    p.add_argument("--out", required=True)
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    params_path = Path(args.params).resolve()
    master1_path = Path(args.master1).resolve()
    master2_path = Path(args.master2).resolve()
    out_prefix = Path(args.out).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    df_p = pd.read_csv(params_path)
    df_p["run_id"] = zero_pad_runids(len(df_p))
    df_m1 = pd.read_csv(master1_path, dtype={"run_id": str})
    df_m2 = pd.read_csv(master2_path, dtype={"run_id": str})


    # 2) Merge and add derived
    df1 = pd.merge(df_p, df_m1, on="run_id", how="inner")
    df2 = pd.merge(df_p, df_m2, on="run_id", how="inner")
    df1["thermal_param"] = compute_thermal_param(df1)
    df2["thermal_param"] = compute_thermal_param(df2)

    # 3) Plot
    base_params = ["v_conv", "age_SP", "age_OP", "dip_int", "eta_int", "eta_UM", "eps_trans"]
    params = base_params + ["thermal_param"]
    n = len(params)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.6 * nrows), constrained_layout=True)
    axes = axes.flatten() if n > 1 else [axes]
    yvar = args.y
    ylab = nice_label(yvar)

    for i, pname in enumerate(params):
        ax = axes[i]

        x1 = df1[pname].to_numpy(dtype=float)
        yvals1 = df1[yvar].to_numpy(dtype=float)
        run_ids1 = df1["run_id"].to_numpy(str)

        x2 = df2[pname].to_numpy(dtype=float)
        yvals2 = df2[yvar].to_numpy(dtype=float)
        run_ids2 = df2["run_id"].to_numpy(str)

        msk1 = np.isfinite(x1) & np.isfinite(yvals1)
        msk2 = np.isfinite(x2) & np.isfinite(yvals2)

        use_logx = pname in LOG_PARAMS

        Xplot1 = x1[msk1]
        Yplot1 = yvals1[msk1]
        RID1 = run_ids1[msk1]

        Xplot2 = x2[msk2]
        Yplot2 = yvals2[msk2]
        RID2 = run_ids2[msk2]
        
        ax.scatter(np.log10(Xplot1) if use_logx else Xplot1, Yplot1, s=18, alpha=0.8, color='blue')
        ax.scatter(np.log10(Xplot2) if use_logx else Xplot2, Yplot2, s=18, alpha=0.8, color='orange')

        # If plotting thermal parameter, use log axis (but not log10 transform)
        if pname == "thermal_param":
            ax.set_xscale("log")

        # Annotate “suspect” points
        for xi, yi, rid in zip(Xplot1, Yplot1, RID1):
            if yi > -20:  
                ax.text(
                    (np.log10(xi) if use_logx else xi),yi,rid,
                    fontsize=7,ha="left",va="center",color="red",
                )
        for xi, yi, rid in zip(Xplot2, Yplot2, RID2):
            if yi > -20:  
                ax.text(
                    (np.log10(xi) if use_logx else xi),yi,rid,
                    fontsize=7,ha="left",va="center",color="red",
                )

        # Labels and formatting
        xlab = nice_label(pname) + (" (log₁₀)" if use_logx else "")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        # rtxt = corr_text(np.log10(Xplot) if use_logx else Xplot, Yplot)
        # ax.text(0.02, 0.95, rtxt, transform=ax.transAxes, ha="left", va="top")
        ax.grid(True, ls=":", alpha=0.4)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # 4) Save
    fig.savefig(f"{out_prefix}.png", dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out_prefix}.png")

    # fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    # print(f"Saved: {out_prefix}.pdf")


if __name__ == "__main__":
    main()

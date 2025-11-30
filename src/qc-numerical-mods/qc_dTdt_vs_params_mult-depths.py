#!/usr/bin/env python3
"""
Multi-depth QC plot: dT or dT/dt vs parameters (+ thermal parameter).

If the params CSV includes `t_conv`, this script adds:
    - v_conv_over_tconv  (v_conv / t_conv)
    - t_conv
and uses a 4-column grid.

Otherwise it uses a 3-column grid (usual case for const-vc suite).
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOG_PARAMS = {"eta_int", "eta_UM", "eps_trans"}  # log10 for these


def zero_pad_runids(n: int):
    """Return ['000','001',...] padded to width >=3."""
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
    """v_conv * age_SP * sin(dip) (arbitrary units, km)."""
    v = df["v_conv"].to_numpy(float) / 1e3     # cm/yr → km/yr
    age = df["age_SP"].to_numpy(float) * 1e6   # Myr → yr
    dip = np.deg2rad(df["dip_int"].to_numpy(float))
    return v * np.maximum(age, 0) * np.sin(np.clip(dip, 0, np.pi/2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True)
    p.add_argument("--master", required=True)
    p.add_argument("--depths", type=float, nargs="+", required=True)
    p.add_argument("--y", default="dTdt_C_per_Myr",
                   choices=["dT_C", "dTdt_C_per_Myr"])
    p.add_argument("--out", required=True)
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    params_path = Path(args.params).resolve()
    master_path = Path(args.master).resolve()
    out_prefix = Path(args.out).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    depths = args.depths

    # ------------------------------------------------------------
    # 1) Load params + master
    # ------------------------------------------------------------
    df_p = pd.read_csv(params_path)
    df_p["run_id"] = zero_pad_runids(len(df_p))

    # If t_conv exists -> compute v_conv_over_tconv
    ramped = "t_conv" in df_p.columns
    if ramped:
        t = df_p["t_conv"].replace(0, np.nan).to_numpy(float)
        v = df_p["v_conv"].to_numpy(float)
        df_p["v_conv_over_tconv"] = v / t

    df_m_all = pd.read_csv(master_path, dtype={"run_id": str})
    if "depth_km" not in df_m_all.columns:
        raise ValueError("Master file missing depth_km column.")

    # Merge per depth
    depth_dfs = []
    for depth in depths:
        df_m = df_m_all[np.isclose(df_m_all["depth_km"], depth)]
        df = pd.merge(df_p, df_m, on="run_id", how="inner")
        if df.empty:
            raise ValueError(f"No matching rows for depth {depth} km")
        df["thermal_param"] = compute_thermal_param(df)
        df["depth_km"] = depth
        depth_dfs.append(df)

    # ------------------------------------------------------------
    # 2) Compose x-parameter list
    # ------------------------------------------------------------
    base_params = []

    # Always include v_conv
    if "v_conv" in df_p.columns:
        base_params.append("v_conv")

    # If ramped: also include v_conv/t_conv and t_conv
    if ramped:
        base_params.append("v_conv_over_tconv")
        base_params.append("t_conv")

    # Standard params
    for name in ["age_SP", "age_OP", "dip_int", "eta_int", "eta_UM", "eps_trans"]:
        if name in df_p.columns:
            base_params.append(name)

    params = base_params + ["thermal_param"]

    # Grid size:
    if ramped:
        ncols = 4   # because of 2 extra ramped-vc panels
    else:
        ncols = 3   # const-vc: 3 per row

    n = len(params)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4*ncols, 3.6*nrows),
        constrained_layout=True
    )
    axes = axes.flatten()

    yvar = args.y
    ylab = nice_label(yvar)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    depth_labels = [f"{d:.0f} km" for d in depths]

    # ------------------------------------------------------------
    # 3) Plot each parameter panel
    # ------------------------------------------------------------
    for i, pname in enumerate(params):
        ax = axes[i]

        use_logx = pname in LOG_PARAMS

        for d_idx, (depth, df) in enumerate(zip(depths, depth_dfs)):
            x = df[pname].to_numpy(float)
            y = df[yvar].to_numpy(float)
            mask = np.isfinite(x) & np.isfinite(y)

            ax.scatter(
                np.log10(x[mask]) if use_logx else x[mask],
                y[mask],
                s=18, alpha=0.75,
                color=colors[d_idx % len(colors)],
                label=depth_labels[d_idx] if i == 0 else None
            )

        if pname == "thermal_param":
            ax.set_xscale("log")

        ax.set_xlabel(nice_label(pname) + (" (log₁₀)" if use_logx else ""))
        ax.set_ylabel(ylab)
        ax.grid(True, ls=":", alpha=0.4)
        ax.set_title(pname)
        if yvar == "dTdt_C_per_Myr":
            ax.set_ylim(-160, 0)

    # hide empty panels
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    axes[0].legend(frameon=False, fontsize=9, title="Depth")

    fig.savefig(f"{out_prefix}.png", dpi=args.dpi, bbox_inches="tight")
    print("Saved:", f"{out_prefix}.png")


if __name__ == "__main__":
    main()


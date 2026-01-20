#!/usr/bin/env python3
"""
Plot emulator misfit (|y_true - y_pred|) vs. each parameter, for both TRAIN and VALIDATION.

Usage example:
  python plot_emulator_misfit_vs_params.py \
      --params ../../data/params-list.csv \
      --data   ./data/50km_dTdt_thermalParam_etaRatio \
      --models ./models \
      --algo   gp_m25 \
      --out    plots/emulator_misfit-vs-params.50km_dTdt_thermalParam_etaRatio.gp_m25.png \
      --label-thresh 8.0
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Axes to display on log10(x); we'll also log-scale thermal_param axis
LOG_PARAMS = {"eta_int", "eta_UM", "eps_trans"}

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
        "thermal_param": r"$v\; \mathrm{age}_{\rm SP}\; \sin(\mathrm{dip})$ (km·Myr/yr)",
        "misfit": r"|Emulator − True| (°C/Myr)",
    }
    return labels.get(param, param)

def compute_thermal_param(df):
    """v_conv * age_SP * sin(dip). Units are arbitrary but consistent with your earlier plots:
       v_conv (cm/yr) -> km/yr, age_SP (Myr) -> yr, dip in radians (sin)."""
    v = df["v_conv"].to_numpy(float) / 1e3     # cm/yr -> km/yr
    age = df["age_SP"].to_numpy(float) * 1e6   # Myr -> yr
    dip_rad = np.deg2rad(df["dip_int"].to_numpy(float))
    tp = v * np.maximum(age, 0.0) * np.sin(np.clip(dip_rad, 0.0, np.pi/2))
    return tp

def main():
    p = argparse.ArgumentParser(description="Plot emulator misfit vs parameters for Train & Validation.")
    p.add_argument("--params", required=True, help="Path to params-list.csv")
    p.add_argument("--data", required=True, help="Path to dataset folder (contains Y_raw.npy, train_idx.npy, val_idx.npy)")
    p.add_argument("--models", required=True, help="Path to models root (contains <name>/<algo>/yhat_*.npy)")
    p.add_argument("--algo",   required=True, help="Model subfolder (e.g., gp_m25)")
    p.add_argument("--out",    required=True, help="Output PNG/PDF path")
    p.add_argument("--label-thresh", type=float, default=8.0,
                   help="Label run_id where |misfit| >= this threshold.")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    params_path = Path(args.params).resolve()
    data_path   = Path(args.data).resolve()
    models_root = Path(args.models).resolve()
    out_path    = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load base data
    df = pd.read_csv(params_path)
    if "run_id" not in df.columns:
        df["run_id"] = zero_pad_runids(len(df))
    # Add derived thermal parameter
    df["thermal_param"] = compute_thermal_param(df)

    # Load arrays from dataset + model predictions
    Y_raw     = np.load(data_path / "Y_raw.npy")
    train_idx = np.load(data_path / "train_idx.npy")
    val_idx   = np.load(data_path / "val_idx.npy")

    # We assume single-target dTdt first column (consistent with your pipeline)
    # yhat_* are shaped to match subsets already
    model_path  = models_root / data_path.name / args.algo
    yhat_train  = np.load(model_path / "yhat_train.npy").reshape(-1)
    yhat_val    = np.load(model_path / "yhat_val.npy").reshape(-1)

    y_true_train = Y_raw[train_idx, 0].reshape(-1)
    y_true_val   = Y_raw[val_idx, 0].reshape(-1)

    misfit_train = np.abs(y_true_train - yhat_train)
    misfit_val   = np.abs(y_true_val   - yhat_val)

    # Grab param columns (existing ones only), then append thermal_param
    base_params = ["v_conv", "age_SP", "age_OP", "dip_int", "eta_int", "eta_UM", "eps_trans"]
    params_to_plot = [c for c in base_params if c in df.columns] + ["thermal_param"]

    n = len(params_to_plot)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.6 * nrows), constrained_layout=True)
    axes = axes.flatten() if n > 1 else [axes]

    # Prepare per-set DataFrames (index-based selection consistent with earlier coverage script)
    df_train = df.iloc[train_idx].copy()
    df_val   = df.iloc[val_idx].copy()

    for i, pname in enumerate(params_to_plot):
        ax = axes[i]
        # X values
        x_train = df_train[pname].to_numpy(dtype=float)
        x_val   = df_val[pname].to_numpy(dtype=float)

        # masks for finite
        mtr = np.isfinite(x_train) & np.isfinite(misfit_train)
        mva = np.isfinite(x_val)   & np.isfinite(misfit_val)

        use_logx = pname in LOG_PARAMS

        # Draw TRAIN first (behind), VALIDATION second (on top)
        ax.scatter(
            (np.log10(x_train[mtr]) if use_logx else x_train[mtr]),
            misfit_train[mtr],
            s=16, alpha=0.35, color="tab:blue", label="Train", zorder=1
        )
        ax.scatter(
            (np.log10(x_val[mva]) if use_logx else x_val[mva]),
            misfit_val[mva],
            s=22, alpha=0.85, color="tab:orange", label="Validation", zorder=2
        )

        # Thermal parameter axis uses true log scale (not log10 transform)
        if pname == "thermal_param":
            ax.set_xscale("log")

        # Label points exceeding threshold (both sets)
        # We’ll label with run_id from the *same* subset rows
        run_ids_train = df_train["run_id"].to_numpy(str)
        run_ids_val   = df_val["run_id"].to_numpy(str)

        for xi, yi, rid in zip(x_train[mtr], misfit_train[mtr], run_ids_train[mtr]):
            if yi >= args.label_thresh:
                ax.text((np.log10(xi) if use_logx else xi), yi, rid,
                        fontsize=7, ha="left", va="center", color="tab:blue", zorder=3)
        for xi, yi, rid in zip(x_val[mva], misfit_val[mva], run_ids_val[mva]):
            if yi >= args.label_thresh:
                ax.text((np.log10(xi) if use_logx else xi), yi, rid,
                        fontsize=7, ha="left", va="center", color="red", zorder=4)

        # Labels/formatting
        xlab = nice_label(pname) + (" (log₁₀)" if use_logx else "")
        ax.set_xlabel(xlab)
        ax.set_ylabel(nice_label("misfit"))
        ax.grid(True, ls=":", alpha=0.4)

    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # One legend
    axes[0].legend(frameon=False, loc="upper right")

    fig.suptitle(f"Emulator misfit vs parameters — {data_path.name} — {args.algo}", fontsize=14)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")

if __name__ == "__main__":
    main()

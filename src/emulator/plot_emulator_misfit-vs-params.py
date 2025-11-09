#!/usr/bin/env python3
"""
Plot emulator misfit (|y_true - y_hat|) vs. each parameter for the VALIDATION set.

Usage:
  python plot_emulator_misfit_vs_params.py \
      --params ../../data/params-list.csv \
      --data ./data/50km_dTdt_thermalParam_etaRatio \
      --models ./models \
      --algo gp_m25 \
      --out plots/emulator_misfit-vs-params.50km_dTdt.gp_m25.png \
      --label-thresh 6.0
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Log-transform these feature axes (log10 on x)
LOG_PARAMS = {"eta_int", "eta_UM", "eps_trans"}
BASE_PARAMS = ["v_conv", "age_SP", "age_OP", "dip_int", "eta_int", "eta_UM", "eps_trans"]

def zero_pad_runids(n: int) -> list[str]:
    width = max(3, len(str(max(0, n - 1)))))
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
        "thermal_param": r"$v\;\mathrm{age}_{\rm SP}\;\sin(\mathrm{dip})$",
        "misfit": r"|Emulator − Truth|",
    }
    return labels.get(param, param)

def compute_thermal_param(df: pd.DataFrame) -> np.ndarray:
    """v_conv(cm/yr) * age_SP(Myr) * sin(dip)  (arbitrary units)."""
    v = pd.to_numeric(df["v_conv"], errors="coerce").to_numpy(float) / 1e3     # cm/yr -> km/yr
    age = pd.to_numeric(df["age_SP"], errors="coerce").to_numpy(float) * 1e6   # Myr -> yr
    dip = pd.to_numeric(df["dip_int"], errors="coerce").to_numpy(float)
    dip_rad = np.deg2rad(dip)
    return v * np.maximum(age, 0.0) * np.sin(np.clip(dip_rad, 0.0, np.pi/2))

def main():
    ap = argparse.ArgumentParser(description="Plot emulator misfit vs. parameters (validation set).")
    ap.add_argument("--params", required=True, help="Path to params-list.csv")
    ap.add_argument("--data",   required=True, help="Path to dataset folder (Y_raw.npy, val_idx.npy, metadata.json)")
    ap.add_argument("--models", required=True, help="Path to models root")
    ap.add_argument("--algo",   default="gp_m25", help="Model subfolder name under models/<name>/")
    ap.add_argument("--out",    required=True, help="Output PNG/PDF path")
    ap.add_argument("--dpi",    type=int, default=200)
    ap.add_argument("--label-thresh", type=float, default=6.0,
                    help="Label run IDs with misfit >= this threshold (in target units).")
    args = ap.parse_args()

    params_path = Path(args.params).resolve()
    data_path   = Path(args.data).resolve()
    models_root = Path(args.models).resolve()
    out_path    = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load params + run_id
    df_p = pd.read_csv(params_path)
    if "run_id" not in df_p.columns:
        df_p["run_id"] = zero_pad_runids(len(df_p))
    else:
        df_p["run_id"] = df_p["run_id"].astype(str).str.zfill(3)

    # Load dataset
    with open(data_path / "metadata.json", "r") as f:
        _meta = json.load(f)

    Y_raw   = np.load(data_path / "Y_raw.npy")     # (N, 1) expected
    val_idx = np.load(data_path / "val_idx.npy")   # (n_val,)

    # Ensure single-target
    if Y_raw.ndim == 1:
        Y_raw = Y_raw.reshape(-1, 1)
    if Y_raw.shape[1] != 1:
        raise ValueError(f"This script assumes a single target; got Y_raw shape {Y_raw.shape}")

    # Load predictions
    name = data_path.name  # e.g., 50km_dTdt_thermalParam_etaRatio
    model_path = models_root / name / args.algo
    yhat_val = np.load(model_path / "yhat_val.npy")  # (n_val, 1) expected
    if yhat_val.ndim == 1:
        yhat_val = yhat_val.reshape(-1, 1)
    if yhat_val.shape[1] != 1:
        raise ValueError(f"Expect single-target predictions; got yhat_val shape {yhat_val.shape}")

    # Build validation frame with misfit
    df_val = df_p.iloc[val_idx].reset_index(drop=True).copy()
    y_true = Y_raw[val_idx, 0]
    y_pred = yhat_val[:, 0]
    misfit = np.abs(y_true - y_pred)

    df_val["__misfit__"] = misfit
    df_val["thermal_param"] = compute_thermal_param(df_val)

    # Parameters to plot
    params = [c for c in BASE_PARAMS if c in df_val.columns] + ["thermal_param"]
    n = len(params)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.6 * nrows), constrained_layout=True)
    axes = axes.flatten() if n > 1 else [axes]

    for i, pname in enumerate(params):
        ax = axes[i]
        x_raw = pd.to_numeric(df_val[pname], errors="coerce").to_numpy(float)
        y_mis = df_val["__misfit__"].to_numpy(float)
        run_id = df_val["run_id"].to_numpy(str)

        m = np.isfinite(x_raw) & np.isfinite(y_mis)
        X = x_raw[m]
        Y = y_mis[m]
        R = run_id[m]

        use_logx = pname in LOG_PARAMS
        if pname == "thermal_param":
            # Plot on true log axis
            good = m & np.isfinite(x_raw) & (x_raw > 0)
            ax.scatter(x_raw[good], y_mis[good], s=22, alpha=0.85, edgecolor="none")
            ax.set_xscale("log")
            # label threshold exceedances
            lbl_mask = (y_mis >= args.label_thresh) & good
            for xi, yi, rid in zip(x_raw[lbl_mask], y_mis[lbl_mask], run_id[lbl_mask]):
                ax.text(xi, yi, rid, fontsize=7, ha="left", va="bottom", color="crimson")
            ax.set_xlabel(nice_label(pname))
        else:
            Xplot = np.log10(X) if use_logx else X
            ax.scatter(Xplot, Y, s=22, alpha=0.85, edgecolor="none")
            # label threshold exceedances
            lbl_mask = (Y >= args.label_thresh)
            for xi, yi, rid in zip(Xplot[lbl_mask], Y[lbl_mask], R[lbl_mask]):
                ax.text(xi, yi, rid, fontsize=7, ha="left", va="bottom", color="crimson")
            xlab = nice_label(pname) + (" (log₁₀)" if use_logx else "")
            ax.set_xlabel(xlab)

        ax.set_ylabel(nice_label("misfit"))
        ax.grid(True, ls=":", alpha=0.4)

    # Hide any unused panels
    for k in range(i + 1, len(axes)):
        axes[k].axis("off")

    fig.suptitle(f"Validation Emulator Misfit vs. Parameters ({name}) • {args.algo}", fontsize=14)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")

if __name__ == "__main__":
    main()

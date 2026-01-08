#!/usr/bin/env python3
"""
Plot how the validation subset covers the parameter space.

Usage (plain coverage; log axes applied to eta_int/eta_UM/eps_trans):
  python plot_emulator_param_coverage.py \
      --params ../../data/params-list.csv \
      --data ./data/50km_dTdt_thermalParam_etaRatio \
      --out plots/emulator_param-coverage_vs_val.50km_dTdt.png

Optional: color validation points by |residual| (needs models/<name>/<algo>/yhat_val.npy):
  python plot_emulator_param_coverage.py \
      --params ../../data/params-list.csv \
      --data ./data/50km_dTdt_thermalParam_etaRatio \
      --out plots/emulator_param-coverage_vs_val.50km_dTdt.residual-gp_m25.png \
      --color-by residual \
      --models ./models \
      --algo gp_m25 \
      --yidx 0
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt

# parameters to show on log axes
LOG_AXES = {"eta_int", "eta_UM", "eps_trans"}

def _log_bins(x, nbins=20):
    """Make log-spaced bins covering positive finite x."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return nbins
    lo, hi = np.min(x), np.max(x)
    if lo <= 0 or hi <= 0 or not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return nbins
    return np.logspace(np.log10(lo), np.log10(hi), nbins)

def main():
    p = argparse.ArgumentParser(description="Plot training vs validation coverage in parameter space.")
    p.add_argument("--params", required=True, help="Path to params-list.csv")
    p.add_argument("--data", required=True, help="Path to dataset folder containing train_idx.npy and val_idx.npy")
    p.add_argument("--out", required=True, help="Output PNG path")

    # optional residual coloring (no target needed otherwise)
    p.add_argument("--color-by", choices=["none", "residual"], default="none",
                   help="Color validation points by absolute residual magnitude (requires model outputs).")
    p.add_argument("--models", default="./models", help="Root dir containing trained model folders")
    p.add_argument("--algo", default="gp_m25", help="Model subfolder name inside models/<name>/")
    p.add_argument("--yidx", type=int, default=0, help="Target index to use for residuals (if multi-target)")
    args = p.parse_args()

    params_path = Path(args.params).resolve()
    data_path = Path(args.data).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load
    df = pd.read_csv(params_path)
    train_idx = np.load(data_path / "train_idx.npy")
    val_idx = np.load(data_path / "val_idx.npy")

    df_train = df.iloc[train_idx].copy()
    df_val   = df.iloc[val_idx].copy()

    # Optional residuals for coloring (aligned to df_val rows)
    residuals = None
    if args.color_by == "residual":
        # data/<name>/Y_raw.npy and models/<name>/<algo>/yhat_val.npy
        name = data_path.name                      # e.g., 50km_dTdt_thermalParam_etaRatio
        Y = np.load(data_path / "Y_raw.npy")       # (N, T) or (N,)
        Y = Y.reshape(Y.shape[0], -1)
        y_true_val = Y[val_idx, args.yidx]

        model_dir = Path(args.models).resolve() / name / args.algo
        yhat_val = np.load(model_dir / "yhat_val.npy")
        yhat_val = yhat_val.reshape(yhat_val.shape[0], -1)[:, args.yidx]

        if yhat_val.shape[0] != y_true_val.shape[0]:
            raise ValueError("yhat_val size does not match number of validation rows.")
        residuals = np.abs(y_true_val - yhat_val)

    # ---- Select key parameters to visualize
    param_cols = ["v_conv", "age_SP", "age_OP", "dip_int", "eta_int", "eta_UM", "eps_trans"]
    existing = [c for c in param_cols if c in df.columns]
    n = len(existing)

    # ---- Pairwise scatter grid
    fig, axes = plt.subplots(n, n, figsize=(2.6*n, 2.6*n), constrained_layout=True)
    scatter_for_cbar = None

    for i, j in itertools.product(range(n), range(n)):
        xname, yname = existing[j], existing[i]
        ax = axes[i, j]

        if i == j:
            # histograms on diagonal
            if xname in LOG_AXES:
                bins = _log_bins(np.concatenate([df_train[xname].values, df_val[xname].values]), nbins=20)
                ax.hist(df_train[xname], bins=bins, color="gray", alpha=0.5, label="Train")
                ax.hist(df_val[xname],   bins=bins, color="orange", alpha=0.7, label="Val")
                ax.set_xscale("log")
            else:
                ax.hist(df_train[xname], bins=20, color="gray", alpha=0.5, label="Train")
                ax.hist(df_val[xname],   bins=20, color="orange", alpha=0.7, label="Val")
            ax.set_xlabel(xname)
            ax.set_ylabel("count")
        else:
            # off-diagonal scatters
            xtr, ytr = df_train[xname].values, df_train[yname].values
            xva, yva = df_val[xname].values,   df_val[yname].values

            ax.scatter(xtr, ytr, s=10, alpha=0.35, color="gray")
            if residuals is None:
                ax.scatter(xva, yva, s=20, alpha=0.75, color="orange", label="Val")
            else:
                sc = ax.scatter(xva, yva, s=24, alpha=0.85, c=residuals, cmap="viridis", label="Val")
                scatter_for_cbar = sc

            # axis labels only on outer edges
            if i == n - 1:
                ax.set_xlabel(xname)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(yname)
            else:
                ax.set_yticklabels([])

            # log axes where appropriate
            if xname in LOG_AXES:
                ax.set_xscale("log")
            if yname in LOG_AXES:
                ax.set_yscale("log")

        ax.grid(True, ls=":", alpha=0.25)

    # ---- Legend + title
    axes[0, 0].legend(loc="upper right", frameon=False)
    fig.suptitle(f"Validation Parameter Coverage ({data_path.name})", fontsize=14)

    # Optional colorbar for residuals
    if scatter_for_cbar is not None:
        cbar = fig.colorbar(scatter_for_cbar, ax=axes.ravel().tolist(), shrink=0.92, pad=0.01)
        cbar.set_label("|residual| (target units)")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")

if __name__ == "__main__":
    main()

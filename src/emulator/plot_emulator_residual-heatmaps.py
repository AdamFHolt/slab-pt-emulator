#!/usr/bin/env python3
"""
Heatmaps of validation residuals (|y_true - y_pred|) over selected parameter pairs.

Consistent interface with your other plotting scripts.

Example:
  python plot_emulator_param_residual_heatmaps.py \
      --params ../../data/params-list.csv \
      --data ./data/50km_dTdt_thermalParam_etaRatio \
      --models ./models \
      --algo gp_m25 \
      --yidx 0 \
      --out plots/emulator_param-residual-heatmaps.50km_dTdt.gp_m25.png

Optionally choose pairs (comma-separated, 3 rows x 2 cols expected):
  --pairs "v_conv,age_SP; age_OP,dip_int; eta_int,eta_UM"
"""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

LOG_AXES = {"eta_int", "eta_UM", "eps_trans"}  # show these on log10 axes

def safe_log10(a):
    a = np.asarray(a, float)
    out = np.full_like(a, np.nan, dtype=float)
    m = a > 0
    out[m] = np.log10(a[m])
    return out

def parse_pairs(pairs_str, available_cols):
    """
    Parse a string like "v_conv,age_SP; age_OP,dip_int; eta_int,eta_UM"
    into a list of (x,y) tuples. Only keep pairs that exist.
    """
    pairs = []
    for chunk in pairs_str.split(";"):
        tok = [t.strip() for t in chunk.split(",") if t.strip()]
        if len(tok) == 2 and all(t in available_cols for t in tok):
            pairs.append((tok[0], tok[1]))
    return pairs

def main():
    ap = argparse.ArgumentParser(description="Residual heatmaps over parameter pairs (validation set).")
    ap.add_argument("--params", required=True, help="Path to params-list.csv")
    ap.add_argument("--data",   required=True, help="Path to dataset folder (has Y_raw.npy, val_idx.npy, metadata.json)")
    ap.add_argument("--models", required=True, help="Path to trained models root")
    ap.add_argument("--algo",   default="gp_m25", help="Model subfolder under models/<name>/")
    ap.add_argument("--yidx",   type=int, default=0, help="Target column index for residuals")
    ap.add_argument("--out",    required=True, help="Output PNG/PDF path")
    ap.add_argument("--pairs",  default="v_conv,age_SP; age_OP,dip_int; eta_int,eta_UM",
                    help="Semicolon-separated param pairs 'x,y; x2,y2; ...' (expects 3x2 = 6 panels)")
    ap.add_argument("--gridsize", type=int, default=35, help="Hexbin grid size")
    ap.add_argument("--dpi",    type=int, default=200)
    args = ap.parse_args()

    params_path = Path(args.params).resolve()
    data_path   = Path(args.data).resolve()
    models_root = Path(args.models).resolve()
    out_path    = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load metadata & arrays
    with open(data_path / "metadata.json", "r") as f:
        meta = json.load(f)
    feat_cols = [c for c in meta.get("feature_cols", [])]

    # pairs to plot (filter to existing)
    pairs = parse_pairs(args.pairs, feat_cols)
    if not pairs:
        # sensible default triad if user gave none/invalid
        candidates = [("v_conv","age_SP"), ("age_OP","dip_int"), ("eta_int","eta_UM")]
        pairs = [(x,y) for x,y in candidates if x in feat_cols and y in feat_cols]
    # pad/trim to 6 panels (3x2)
    while len(pairs) < 6:
        # reuse from beginning if needed
        pairs.append(pairs[len(pairs) % max(1,len(pairs))])
    pairs = pairs[:6]

    # params dataframe, targets, split
    df = pd.read_csv(params_path)
    Y  = np.load(data_path / "Y_raw.npy")
    val_idx = np.load(data_path / "val_idx.npy")

    # predictions for validation set
    name = data_path.name
    yhat_val = np.load(models_root / name / args.algo / "yhat_val.npy")

    # residuals vector for selected target
    y_true_val = Y[val_idx, args.yidx]
    y_pred_val = yhat_val[:, args.yidx]
    resid = np.abs(y_true_val - y_pred_val)

    # slice parameter rows
    df_val   = df.iloc[val_idx].copy()
    # also sample some train points for faint background
    train_idx = np.load(data_path / "train_idx.npy")
    df_train  = df.iloc[train_idx].copy()

    # ---- Figure
    fig, axes = plt.subplots(3, 2, figsize=(12, 14), constrained_layout=True)
    axes = axes.ravel()

    vmin = np.nanpercentile(resid, 5)
    vmax = np.nanpercentile(resid, 95)
    # avoid zero for LogNorm
    vmin = max(vmin, 1e-12)

    for ax, (xcol, ycol) in zip(axes, pairs):
        # get arrays
        x_tr = df_train[xcol].to_numpy(float)
        y_tr = df_train[ycol].to_numpy(float)
        x_va = df_val[xcol].to_numpy(float)
        y_va = df_val[ycol].to_numpy(float)

        # log visuals where appropriate
        if xcol in LOG_AXES:
            x_tr = safe_log10(x_tr)
            x_va = safe_log10(x_va)
            xlab = f"log10({xcol})"
        else:
            xlab = xcol
        if ycol in LOG_AXES:
            y_tr = safe_log10(y_tr)
            y_va = safe_log10(y_va)
            ylab = f"log10({ycol})"
        else:
            ylab = ycol

        # light train background
        ax.scatter(x_tr, y_tr, s=6, alpha=0.15, color="gray", linewidths=0)

        # bin validation points and color by mean residual in each hex
        # We compute weighted mean via two passes: sum(resid) and count
        hb_sum = ax.hexbin(x_va, y_va, C=resid, reduce_C_function=np.mean,
                           gridsize=args.gridsize, cmap="viridis",
                           mincnt=3,  # require at least a few points
                           norm=LogNorm(vmin=vmin, vmax=vmax))

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(True, ls=":", alpha=0.3)

    # single colorbar
    cbar = fig.colorbar(hb_sum, ax=axes.tolist(), shrink=0.95, pad=0.02)
    tcols = meta.get("target", {}).get("target_cols", [])
    tname = tcols[args.yidx] if tcols and args.yidx < len(tcols) else f"target[{args.yidx}]"
    cbar.set_label(f"Mean |residual| for {tname}")

    fig.suptitle(f"Validation Residual Heatmaps ({name}) • {args.algo}", fontsize=15)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot PCA-score prediction diagnostics for one trained model.")
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--split", choices=["val", "train"], default="val")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    ds = Path(args.dataset_dir).resolve()
    md = Path(args.model_dir).resolve()

    y_true = np.load(ds / "Y_raw.npy")
    train_idx = np.load(ds / "train_idx.npy")
    val_idx = np.load(ds / "val_idx.npy")

    if args.split == "val":
        idx = val_idx
        y_pred = np.load(md / "yhat_val.npy")
    else:
        idx = train_idx
        y_pred = np.load(md / "yhat_train.npy")

    y_true_s = y_true[idx]
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    n_targets = y_true_s.shape[1]
    ncols = 4
    nrows = int(np.ceil(n_targets / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.0 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for j in range(nrows * ncols):
        ax = axes.flat[j]
        if j >= n_targets:
            ax.axis("off")
            continue
        yt = y_true_s[:, j]
        yp = y_pred[:, j]
        lo = min(float(np.min(yt)), float(np.min(yp)))
        hi = max(float(np.max(yt)), float(np.max(yp)))
        pad = 0.05 * (hi - lo + 1e-9)
        ax.scatter(yt, yp, s=10, alpha=0.65, color="tab:blue", edgecolors="none")
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", lw=1.0, color="0.4")
        ax.set_title(f"PC{j+1}  R2={_r2(yt, yp):.3f}")
        ax.set_xlabel("true")
        ax.set_ylabel("pred")
        ax.grid(True, ls=":", alpha=0.3)

    out_scatter = Path(f"{args.out_prefix}_{args.split}_scores_scatter.png").resolve()
    out_scatter.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_scatter, dpi=220, bbox_inches="tight")

    # RMSE distribution in score space per run
    rmse_run = np.sqrt(np.mean((y_true_s - y_pred) ** 2, axis=1))
    fig2, ax2 = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    ax2.hist(rmse_run, bins=20, color="tab:orange", alpha=0.8)
    ax2.axvline(float(np.mean(rmse_run)), color="0.2", ls="--", lw=1.5, label=f"mean={np.mean(rmse_run):.2f}")
    ax2.axvline(float(np.percentile(rmse_run, 90)), color="tab:red", ls="--", lw=1.5,
                label=f"P90={np.percentile(rmse_run, 90):.2f}")
    ax2.set_xlabel("Per-run RMSE in PCA score space")
    ax2.set_ylabel("Count")
    ax2.set_title(f"{args.split}: score-space RMSE distribution")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, ls=":", alpha=0.3)

    out_hist = Path(f"{args.out_prefix}_{args.split}_scores_rmse_hist.png").resolve()
    fig2.savefig(out_hist, dpi=220, bbox_inches="tight")

    print(f"[OK] Saved: {out_scatter}")
    print(f"[OK] Saved: {out_hist}")
    print(f"[OK] {args.split} score-space RMSE mean={np.mean(rmse_run):.6f} p90={np.percentile(rmse_run, 90):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

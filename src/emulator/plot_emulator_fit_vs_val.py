#!/usr/bin/env python3
"""
Compare emulator predictions vs. true values (Train & Validation) across depths.

For each listed dataset name (e.g., 25km_dTdt, 50km_dTdt, 75km_dTdt):
- Loads Y_raw.npy, yhat_train.npy, yhat_val.npy, and train/val_idx.npy
- Plots emulator vs true for Train (left) and Validation (right)
- Colors by depth (blue/orange/green)

Usage:
  python plot_emulator_fit_vs_val.py \
    --data-root ./data \
    --models ./models \
    --names 25km_dTdt 50km_dTdt 75km_dTdt \
    --yidx 0 \
    --out plots/emulator_fit_vs_val__base.png
"""

import argparse
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt


def load_bundle(data_root: Path, model_root: Path, name: str):
    """Load true/predicted Y for one dataset name (depth)."""
    data_path = data_root / name
    model_path = model_root / name / "gp_m25"

    with open(data_path / "metadata.json", "r") as f:
        meta = json.load(f)

    Y = np.load(data_path / "Y_raw.npy")
    train_idx = np.load(data_path / "train_idx.npy")
    val_idx = np.load(data_path / "val_idx.npy")

    yhat_train = np.load(model_path / "yhat_train.npy")
    yhat_val = np.load(model_path / "yhat_val.npy")

    target_cols = meta["target"]["target_cols"]
    return (Y[train_idx], yhat_train, Y[val_idx], yhat_val, target_cols)


def main():
    p = argparse.ArgumentParser(description="Plot emulator vs true (Train & Val) across depths.")
    p.add_argument("--data-root", required=True, help="Directory with preprocessed data folders")
    p.add_argument("--models", required=True, help="Directory with trained model folders")
    p.add_argument("--names", nargs="+", required=True,
                   help="Names of subfolders (e.g., 25km_dTdt 50km_dTdt 75km_dTdt)")
    p.add_argument("--yidx", type=int, default=0, help="Which target column to plot (0 = first)")
    p.add_argument("--out", required=True, help="Output plot path (png/pdf)")
    args = p.parse_args()

    data_root = Path(args.data_root).resolve()
    model_root = Path(args.models).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    colors = ["blue", "orange", "green"]
    depth_labels = [n.split("km")[0] + " km" for n in args.names]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    ax_train, ax_val = axes

    all_pts = []
    target_name = None

    for name, color, lbl in zip(args.names, colors, depth_labels):
        Ytr_true, Ytr_pred, Yva_true, Yva_pred, target_cols = load_bundle(data_root, model_root, name)
        target_name = target_cols[args.yidx]

        ytr_true = Ytr_true[:, args.yidx]
        ytr_pred = Ytr_pred[:, args.yidx]
        yva_true = Yva_true[:, args.yidx]
        yva_pred = Yva_pred[:, args.yidx]

        ax_train.scatter(ytr_true, ytr_pred, s=20, alpha=0.7, color=color, label=lbl)
        ax_val.scatter(yva_true, yva_pred, s=20, alpha=0.7, color=color, label=lbl)

        all_pts.append((np.concatenate([ytr_true, yva_true]),
                        np.concatenate([ytr_pred, yva_pred])))

    # Determine global bounds for 1:1 lines
    all_min = min(np.min(y_true) for y_true, _ in all_pts)
    all_max = max(np.max(y_true) for y_true, _ in all_pts)

    for ax, title in zip(axes, ["Train (Fitted)", "Validation (Held-out)"]):
        ax.plot([all_min, all_max], [all_min, all_max], "k--", lw=1)
        ax.set_xlabel(f"True {target_name}")
        ax.set_ylabel(f"Emulator {target_name}")
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Emulator Performance Across Depths", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()

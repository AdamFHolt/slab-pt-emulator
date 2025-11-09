#!/usr/bin/env python3
"""
Compare emulator predictions vs. true values (Train & Validation) across depths,
with summary stats (R², RMSE) for each depth and mean across depths.
"""

import argparse
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt


def load_bundle(data_root: Path, model_root: Path, name: str, algo: str):
    """Load true/predicted Y for one dataset name (depth)."""
    data_path = data_root / name
    model_path = model_root / name / algo

    with open(data_path / "metadata.json", "r") as f:
        meta = json.load(f)

    Y = np.load(data_path / "Y_raw.npy")
    train_idx = np.load(data_path / "train_idx.npy")
    val_idx = np.load(data_path / "val_idx.npy")

    yhat_train = np.load(model_path / "yhat_train.npy")
    yhat_val = np.load(model_path / "yhat_val.npy")

    target_cols = meta["target"]["target_cols"]
    return (Y[train_idx], yhat_train, Y[val_idx], yhat_val, target_cols)


def _read_report_stats(report_path: Path):
    """
    Return dicts for train/val macro metrics, e.g. {"r2": 0.823, "rmse": 3.74}
    """
    try:
        with open(report_path, "r") as f:
            rep = json.load(f)
    except FileNotFoundError:
        return None, None

    train_m = rep.get("metrics", {}).get("train", {}).get("_macro_avg")
    val_m   = rep.get("metrics", {}).get("val",   {}).get("_macro_avg")
    return train_m, val_m


def _fmt_line(lbl, m):
    """Format one compact line."""
    if not m:
        return None
    return f"{lbl}: R²={m['r2']:.3f}, RMSE={m['rmse']:.2f}"


def main():
    p = argparse.ArgumentParser(description="Plot emulator vs true (Train & Val) across depths.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--algo", default="gp_m25")
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--yidx", type=int, default=0)
    p.add_argument("--out", required=True)
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

    # For stats text boxes
    train_lines = []
    val_lines = []
    train_vals, val_vals = [], []

    for name, color, lbl in zip(args.names, colors, depth_labels):
        Ytr_true, Ytr_pred, Yva_true, Yva_pred, target_cols = load_bundle(
            data_root, model_root, name, algo=args.algo
        )
        target_name = target_cols[args.yidx]

        ytr_true = Ytr_true[:, args.yidx]
        ytr_pred = Ytr_pred[:, args.yidx]
        yva_true = Yva_true[:, args.yidx]
        yva_pred = Yva_pred[:, args.yidx]

        ax_train.scatter(ytr_true, ytr_pred, s=20, alpha=0.7, color=color, label=lbl)
        ax_val.scatter(yva_true, yva_pred, s=20, alpha=0.7, color=color, label=lbl)

        all_pts.append((np.concatenate([ytr_true, yva_true]),
                        np.concatenate([ytr_pred, yva_pred])))

        # Read stats
        report_path = model_root / name / args.algo / "report.json"
        tr_m, va_m = _read_report_stats(report_path)
        if tr_m:
            train_lines.append(_fmt_line(lbl, tr_m))
            train_vals.append(tr_m)
        if va_m:
            val_lines.append(_fmt_line(lbl, va_m))
            val_vals.append(va_m)

    # ---- Compute mean metrics over all depths
    def mean_stats(metrics):
        if not metrics:
            return None
        mean_r2 = np.mean([m["r2"] for m in metrics])
        mean_rmse = np.mean([m["rmse"] for m in metrics])
        return {"r2": mean_r2, "rmse": mean_rmse}

    tr_mean = mean_stats(train_vals)
    va_mean = mean_stats(val_vals)
    if tr_mean:
        train_lines.append(f"Mean: R²={tr_mean['r2']:.3f}, RMSE={tr_mean['rmse']:.2f}")
    if va_mean:
        val_lines.append(f"Mean: R²={va_mean['r2']:.3f}, RMSE={va_mean['rmse']:.2f}")

    # ---- Determine global bounds
    all_min = min(np.min(y_true) for y_true, _ in all_pts)
    all_max = max(np.max(y_true) for y_true, _ in all_pts)

    for ax, title in zip(axes, ["Train (Fitted)", "Validation (Held-out)"]):
        ax.plot([all_min, all_max], [all_min, all_max], "k--", lw=1)
        ax.set_xlabel(f"True {target_name}")
        ax.set_ylabel(f"Emulator {target_name}")
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(title, fontsize=11)

    # ---- Add text boxes
    if train_lines:
        ax_train.text(
            0.02, 0.98,
            "\n".join(train_lines),
            transform=ax_train.transAxes, va="top", ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, ec="0.7")
        )
    if val_lines:
        ax_val.text(
            0.02, 0.98,
            "\n".join(val_lines),
            transform=ax_val.transAxes, va="top", ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, ec="0.7")
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()

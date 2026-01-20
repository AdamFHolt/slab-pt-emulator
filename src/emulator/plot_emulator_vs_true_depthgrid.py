#!/usr/bin/env python3
"""
Plot emulator predictions vs. true values across MANY depths, with per-depth stats.

Example:
  python plot_emulator_vs_true_depthgrid.py \
    --suite const-vc \
    --variant dTdt_thermalParam \
    --algo gp_rbf
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


PLOTS_DIR_DEFAULT = Path("/home/holt/Projects/SlabPT-emulator/plots/qc-emulator")


# ------------------------- IO helpers

def load_bundle(data_root: Path, model_root: Path, suite: str, name: str, algo: str):
    """
    Load true/predicted Y for one dataset name (depth folder).
    Expects:
      data_root/<suite>/<name>/metadata.json, Y_raw.npy, train_idx.npy, val_idx.npy
      model_root/<suite>/<name>/<algo>/yhat_train.npy, yhat_val.npy, report.json
    """
    data_path = data_root / suite / name
    model_path = model_root / suite / name / algo

    with open(data_path / "metadata.json", "r") as f:
        meta = json.load(f)

    Y = np.load(data_path / "Y_raw.npy")
    train_idx = np.load(data_path / "train_idx.npy")
    val_idx = np.load(data_path / "val_idx.npy")

    yhat_train = np.load(model_path / "yhat_train.npy")
    yhat_val = np.load(model_path / "yhat_val.npy")

    target_cols = meta["target"]["target_cols"]
    return (Y[train_idx], yhat_train, Y[val_idx], yhat_val, target_cols, model_path)


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


def _depth_from_name(name: str) -> Optional[float]:
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*km_", name)
    if not m:
        return None
    return float(m.group(1))


def _sorted_names(names: List[str]) -> List[str]:
    pairs = []
    for n in names:
        d = _depth_from_name(n)
        pairs.append((d if d is not None else 1e9, n))
    return [n for _, n in sorted(pairs, key=lambda t: t[0])]


def _ensure_2d(y: np.ndarray) -> np.ndarray:
    return y.reshape(-1, 1) if y.ndim == 1 else y


# ------------------------- plotting helpers

def _fmt_line(m: Optional[Dict[str, Any]]) -> str:
    if not m:
        return "stats: (missing report.json)"
    return f"R²={m['r2']:.3f}  RMSE={m['rmse']:.3g}"


def main():
    p = argparse.ArgumentParser(description="Plot emulator vs true across depths (Train & Val).")

    # roots
    p.add_argument("--data-root", default=str(Path(__file__).parent / "data"),
                   help="Root containing suite folders (e.g., ./data)")
    p.add_argument("--models-root", default=str(Path(__file__).parent / "models"),
                   help="Root containing suite folders (e.g., ./models)")

    # which suite/model variant
    p.add_argument("--suite", required=True, choices=["const-vc", "ramped-vc"])
    p.add_argument("--variant", default="dTdt",
                   help="Dataset variant suffix after '<depth>km_' (e.g., dTdt, dTdt_thermalParam, dTdt_thermalParam_etaRatio)")
    p.add_argument("--algo", default="gp_rbf",
                   help="Model subdir name under models/<suite>/<name>/ (e.g., gp_rbf, gp_m15, gp_m25, rf, etc.)")

    # target column index (if multi-target later)
    p.add_argument("--yidx", type=int, default=0)

    # optionally specify exactly which names to plot
    p.add_argument("--names", nargs="*", default=None,
                   help="Optional explicit list of dataset folder names (e.g., 10km_dTdt 20km_dTdt ...). "
                        "If omitted, we auto-discover all '*km_<variant>' folders in data/<suite>/.")

    # output
    p.add_argument("--outdir", default=str(PLOTS_DIR_DEFAULT),
                   help=f"Output directory (default: {PLOTS_DIR_DEFAULT})")
    p.add_argument("--outfile", default=None,
                   help="Optional explicit filename. If omitted, auto-names using suite/variant/algo.")
    p.add_argument("--dpi", type=int, default=220)

    args = p.parse_args()

    data_root = Path(args.data_root).resolve()
    models_root = Path(args.models_root).resolve()
    suite = args.suite

    # Discover names if not provided
    if args.names:
        names = args.names
    else:
        suite_dir = data_root / suite
        if not suite_dir.exists():
            raise SystemExit(f"[ERR] Suite data directory not found: {suite_dir}")
        # match exactly "<depth>km_<variant>" (allow more suffix if you want, but be strict by default)
        pat = re.compile(r"^\d+(?:\.\d+)?km_" + re.escape(args.variant) + r"$")
        names = [p.name for p in suite_dir.iterdir() if p.is_dir() and pat.match(p.name)]

        # If strict match yields nothing, fallback to startswith variant (helpful when variant has extra parts)
        if not names:
            names = [p.name for p in suite_dir.iterdir()
                     if p.is_dir() and p.name.startswith(tuple([f"{i}km_{args.variant}" for i in range(0, 1000)]))]
            # above is ugly; better fallback:
            names = [p.name for p in suite_dir.iterdir()
                     if p.is_dir() and re.match(r"^\d+(?:\.\d+)?km_" + re.escape(args.variant), p.name)]

    if not names:
        raise SystemExit(f"[ERR] No datasets found for suite='{suite}' variant='{args.variant}' under {data_root/suite}")

    names = _sorted_names(names)

    # Load all, compute global bounds
    bundles = []
    all_true = []
    all_pred = []
    target_name = None

    train_stats = []
    val_stats = []

    for name in names:
        Ytr_true, Ytr_pred, Yva_true, Yva_pred, target_cols, model_path = load_bundle(
            data_root, models_root, suite=suite, name=name, algo=args.algo
        )

        Ytr_true = _ensure_2d(Ytr_true)
        Ytr_pred = _ensure_2d(Ytr_pred)
        Yva_true = _ensure_2d(Yva_true)
        Yva_pred = _ensure_2d(Yva_pred)

        if target_name is None:
            target_name = target_cols[args.yidx]

        ytr_true = Ytr_true[:, args.yidx]
        ytr_pred = Ytr_pred[:, args.yidx]
        yva_true = Yva_true[:, args.yidx]
        yva_pred = Yva_pred[:, args.yidx]

        all_true.append(np.concatenate([ytr_true, yva_true]) if yva_true.size else ytr_true)
        all_pred.append(np.concatenate([ytr_pred, yva_pred]) if yva_pred.size else ytr_pred)

        report_path = model_path / "report.json"
        tr_m, va_m = _read_report_stats(report_path)
        if tr_m: train_stats.append(tr_m)
        if va_m: val_stats.append(va_m)

        bundles.append((name, ytr_true, ytr_pred, yva_true, yva_pred, tr_m, va_m))

    all_true_cat = np.concatenate(all_true) if all_true else np.array([0.0])
    all_pred_cat = np.concatenate(all_pred) if all_pred else np.array([0.0])
    global_min = float(np.nanmin(np.concatenate([all_true_cat, all_pred_cat])))
    global_max = float(np.nanmax(np.concatenate([all_true_cat, all_pred_cat])))
    if not np.isfinite(global_min) or not np.isfinite(global_max) or global_min == global_max:
        global_min, global_max = 0.0, 1.0

    # Mean stats
    def _mean_stats(ms: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        if not ms:
            return None
        return {
            "r2": float(np.mean([m["r2"] for m in ms])),
            "rmse": float(np.mean([m["rmse"] for m in ms])),
        }

    tr_mean = _mean_stats(train_stats)
    va_mean = _mean_stats(val_stats)

    n = len(bundles)
    nrows = int(np.ceil(n / 2))
    ncols = 4  # 2 train + 2 val

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12, 3.2 * nrows),
        sharex=True,
        sharey=True
    )

    # Ensure 2D indexing even if nrows == 1
    axes = np.atleast_2d(axes)


    # If n == 1, axes isn't 2D
    if n == 1:
        axes = np.array([axes])

    for i, (name, ytr_true, ytr_pred, yva_true, yva_pred, tr_m, va_m) in enumerate(bundles):
        row = i // 2
        col_offset = (i % 2) * 2  # 0 or 2

        ax_tr = axes[row, col_offset]
        ax_va = axes[row, col_offset + 1]

        d = _depth_from_name(name)
        depth_lbl = f"{d:g} km" if d is not None else name

        # --- Train
        ax_tr.scatter(ytr_true, ytr_pred, s=14, alpha=0.7)
        ax_tr.plot([global_min, global_max], [global_min, global_max], "k--", lw=1)
        ax_tr.set_title(f"{depth_lbl} – Train", fontsize=10)
        ax_tr.grid(True, ls=":", alpha=0.35)

        # --- Val
        ax_va.scatter(yva_true, yva_pred, s=14, alpha=0.7)
        ax_va.plot([global_min, global_max], [global_min, global_max], "k--", lw=1)
        ax_va.set_title(f"{depth_lbl} – Val", fontsize=10)
        ax_va.grid(True, ls=":", alpha=0.35)

        # Small stat boxes
        ax_tr.text(
            0.02, 0.98, _fmt_line(tr_m),
            transform=ax_tr.transAxes, va="top", ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85)
        )
        ax_va.text(
            0.02, 0.98, _fmt_line(va_m),
            transform=ax_va.transAxes, va="top", ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.85)
        )


    # x-labels
    for ax in axes[-1, :]:
        ax.set_xlabel(f"True {target_name}")

    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted")


    # Column titles
    axes[0, 0].set_title("Train (fitted)", fontsize=12)
    axes[0, 1].set_title("Validation (held-out)", fontsize=12)

    # Global limits
    for ax in axes.ravel():
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)

    # Suptitle
    tr_txt = f"Train mean: R²={tr_mean['r2']:.3f}, RMSE={tr_mean['rmse']:.3g}" if tr_mean else "Train mean: (n/a)"
    va_txt = f"Val mean: R²={va_mean['r2']:.3f}, RMSE={va_mean['rmse']:.3g}" if va_mean else "Val mean: (n/a)"
    fig.suptitle(
        f"{suite} | {args.variant} | {args.algo} | target={target_name}\n{tr_txt}   |   {va_txt}",
        fontsize=12, y=0.995
    )

    fig.tight_layout(rect=[0, 0, 1, 0.975])

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.outfile:
        outpath = outdir / args.outfile
    else:
        outpath = outdir / f"emulator_scatter_{suite}_{args.variant}_{args.algo}_{target_name}.png"

    fig.savefig(outpath, dpi=args.dpi, bbox_inches="tight")
    print(f"[OK] Saved: {outpath}")


if __name__ == "__main__":
    main()

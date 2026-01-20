#!/usr/bin/env python3
"""
Plot how the validation subset covers parameter space (train vs val), optionally colored by |residual|.

Example (plain coverage):
  python plot_emulator_param-coverage.py --suite const-vc --variant dTdt_thermalParam --algo gp_rbf

Color validation points by |residual|:
  python plot_emulator_param-coverage.py --suite const-vc --variant dTdt_thermalParam --algo gp_rbf --color-by residual
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PLOTS_DIR_DEFAULT = Path("/home/holt/Projects/SlabPT-emulator/plots/qc-emulator")

# parameters to show on log axes
LOG_AXES = {"eta_int", "eta_UM", "eps_trans"}


# ------------------------- small helpers

def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _zfill_run_id_series(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.number):
        return s.astype(int).map(lambda i: f"{i:03d}")
    return s.astype(str).str.zfill(3)


def _load_params_csv(params_path: Path) -> pd.DataFrame:
    df = pd.read_csv(params_path)
    if "run_id" not in df.columns:
        df = df.copy()
        df["run_id"] = [f"{i:03d}" for i in range(len(df))]
    df["run_id"] = _zfill_run_id_series(df["run_id"])
    return df


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


def _depth_from_name(name: str) -> Optional[float]:
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*km_", name)
    return float(m.group(1)) if m else None


def _sorted_names(names: List[str]) -> List[str]:
    pairs = []
    for n in names:
        d = _depth_from_name(n)
        pairs.append((d if d is not None else 1e9, n))
    return [n for _, n in sorted(pairs, key=lambda t: t[0])]


# ------------------------- core plotting

def plot_one_dataset(
    *,
    suite: str,
    name: str,
    data_path: Path,
    model_path: Path,
    df_params_full: pd.DataFrame,
    outpath: Path,
    color_by: str,
    yidx: int,
    dpi: int,
):
    """
    One N×N grid (pairwise scatter + diagonal hists) comparing train vs val coverage.
    """
    meta = _load_json(data_path / "metadata.json")
    run_ids = meta.get("run_ids", None)
    if not run_ids:
        raise SystemExit(f"[ERR] metadata.json missing 'run_ids' for dataset: {suite}/{name}")

    # Align params to the dataset order (rows of X/Y arrays) via run_ids
    dfp = df_params_full.set_index("run_id", drop=False)
    missing = [rid for rid in run_ids if rid not in dfp.index]
    if missing:
        raise SystemExit(
            f"[ERR] params CSV missing {len(missing)} run_ids referenced by metadata for {suite}/{name}. "
            f"First few: {missing[:10]}"
        )
    df_aligned = dfp.loc[run_ids].reset_index(drop=True)

    # indices refer to aligned rows (post-drop/clean)
    train_idx = np.load(data_path / "train_idx.npy")
    val_idx = np.load(data_path / "val_idx.npy")

    df_train = df_aligned.iloc[train_idx].copy()
    df_val = df_aligned.iloc[val_idx].copy()

    # Optional residuals for coloring (aligned to df_val rows)
    residuals = None
    if color_by == "residual":
        Y = np.load(data_path / "Y_raw.npy")
        Y = Y.reshape(Y.shape[0], -1)
        y_true_val = Y[val_idx, yidx]

        yhat_val = np.load(model_path / "yhat_val.npy")
        yhat_val = np.asarray(yhat_val).reshape(-1, 1) if np.asarray(yhat_val).ndim == 1 else np.asarray(yhat_val)
        yhat_val = yhat_val[:, yidx]

        if yhat_val.shape[0] != y_true_val.shape[0]:
            raise ValueError(
                f"yhat_val size does not match number of validation rows for {suite}/{name}: "
                f"{yhat_val.shape[0]} vs {y_true_val.shape[0]}"
            )
        residuals = np.abs(y_true_val - yhat_val)

    # ---- Select key parameters to visualize (only those present)
    param_cols = ["v_conv", "age_SP", "age_OP", "dip_int", "eta_int", "eta_UM", "eps_trans", "t_conv"]
    existing = [c for c in param_cols if c in df_aligned.columns]
    n = len(existing)
    if n == 0:
        raise SystemExit(f"[ERR] No expected param columns found in params CSV for {suite}/{name}")

    fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.6 * n), constrained_layout=True)
    scatter_for_cbar = None

    for i, j in itertools.product(range(n), range(n)):
        xname, yname = existing[j], existing[i]
        ax = axes[i, j]

        if i == j:
            # histograms on diagonal
            xtr = pd.to_numeric(df_train[xname], errors="coerce").to_numpy(float)
            xva = pd.to_numeric(df_val[xname], errors="coerce").to_numpy(float)

            if xname in LOG_AXES:
                bins = _log_bins(np.concatenate([xtr, xva]), nbins=20)
                ax.hist(xtr, bins=bins, alpha=0.5, label="Train")
                ax.hist(xva, bins=bins, alpha=0.7, label="Val")
                ax.set_xscale("log")
            else:
                ax.hist(xtr, bins=20, alpha=0.5, label="Train")
                ax.hist(xva, bins=20, alpha=0.7, label="Val")

            ax.set_ylabel("count")
            if i == n - 1:
                ax.set_xlabel(xname)
            else:
                ax.set_xticklabels([])

        else:
            # off-diagonal scatters
            xtr = pd.to_numeric(df_train[xname], errors="coerce").to_numpy(float)
            ytr = pd.to_numeric(df_train[yname], errors="coerce").to_numpy(float)
            xva = pd.to_numeric(df_val[xname], errors="coerce").to_numpy(float)
            yva = pd.to_numeric(df_val[yname], errors="coerce").to_numpy(float)

            mtr = np.isfinite(xtr) & np.isfinite(ytr)
            mva = np.isfinite(xva) & np.isfinite(yva)

            ax.scatter(xtr[mtr], ytr[mtr], s=10, alpha=0.35, label=None)

            if residuals is None:
                ax.scatter(xva[mva], yva[mva], s=20, alpha=0.75, label="Val")
            else:
                sc = ax.scatter(xva[mva], yva[mva], s=24, alpha=0.85, c=residuals[mva], cmap="viridis", label="Val")
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

    axes[0, 0].legend(loc="upper right", frameon=False)
    title = f"Validation Parameter Coverage — {suite}/{name}"
    if residuals is not None:
        title += " (colored by |residual|)"
    fig.suptitle(title, fontsize=14)

    if scatter_for_cbar is not None:
        cbar = fig.colorbar(scatter_for_cbar, ax=axes.ravel().tolist(), shrink=0.92, pad=0.01)
        cbar.set_label("|residual| (target units)")

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {outpath}")


def main():
    p = argparse.ArgumentParser(description="Plot training vs validation coverage in parameter space.")

    p.add_argument("--data-root", default=str(Path(__file__).parent / "data"),
                   help="Root containing suite folders (e.g., ./data)")
    p.add_argument("--models-root", default=str(Path(__file__).parent / "models"),
                   help="Root containing suite folders (e.g., ./models)")

    p.add_argument("--suite", required=True, choices=["const-vc", "ramped-vc"])
    p.add_argument("--variant", required=True,
                   help="Dataset variant suffix after '<depth>km_' (e.g., dTdt, dTdt_thermalParam)")
    p.add_argument("--algo", default="gp_rbf",
                   help="Model subdir name under models/<suite>/<name>/ (e.g., gp_rbf, gp_m15, gp_m25, rf, etc.)")

    p.add_argument("--names", nargs="*", default=None,
                   help="Optional explicit dataset names. If omitted, auto-discovers all '*km_<variant>' under data/<suite>/.")

    p.add_argument("--params", default=None,
                   help="Optional params-list.<suite>.csv override. If omitted, uses metadata.json['params_path'] from the first dataset.")

    p.add_argument("--color-by", choices=["none", "residual"], default="none",
                   help="Color validation points by absolute residual magnitude (requires model outputs).")
    p.add_argument("--yidx", type=int, default=0)

    p.add_argument("--outdir", default=str(PLOTS_DIR_DEFAULT),
                   help=f"Output directory (default: {PLOTS_DIR_DEFAULT})")
    p.add_argument("--dpi", type=int, default=220)

    args = p.parse_args()

    data_root = Path(args.data_root).resolve()
    models_root = Path(args.models_root).resolve()
    suite_dir = data_root / args.suite

    if not suite_dir.exists():
        raise SystemExit(f"[ERR] Suite data directory not found: {suite_dir}")

    if args.names:
        names = args.names
    else:
        pat = re.compile(r"^\d+(?:\.\d+)?km_" + re.escape(args.variant) + r"$")
        names = [p.name for p in suite_dir.iterdir() if p.is_dir() and pat.match(p.name)]
        names = _sorted_names(names)

    if not names:
        raise SystemExit(f"[ERR] No datasets found for suite='{args.suite}' variant='{args.variant}' in {suite_dir}")

    # Load params CSV once
    if args.params:
        params_path = Path(args.params).expanduser().resolve()
    else:
        meta0 = _load_json(suite_dir / names[0] / "metadata.json")
        params_path = Path(meta0["params_path"]).resolve()

    if not params_path.exists():
        raise SystemExit(f"[ERR] params-list CSV not found: {params_path}")

    df_params_full = _load_params_csv(params_path)

    outdir = Path(args.outdir).resolve()
    plotdir = outdir.joinpath(args.suite, "param-coverage").resolve()
    plotdir.mkdir(parents=True, exist_ok=True)

    for name in names:
        data_path = data_root / args.suite / name
        model_path = models_root / args.suite / name / args.algo

        outpath = plotdir / f"{args.suite}_{name}_{args.algo}_coverage.png"
        plot_one_dataset(
            suite=args.suite,
            name=name,
            data_path=data_path,
            model_path=model_path,
            df_params_full=df_params_full,
            outpath=outpath,
            color_by=args.color_by,
            yidx=args.yidx,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()

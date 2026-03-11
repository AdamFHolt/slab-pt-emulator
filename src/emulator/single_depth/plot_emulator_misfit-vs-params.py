#!/usr/bin/env python3
"""
Plot emulator misfit (|y_true - y_pred|) vs. each FEATURE (from metadata["feature_cols"]),
for both TRAIN and VALIDATION. One figure per dataset (depth).

Example:
  python plot_emulator_misfit-vs-params.py --suite const-vc --variant dTdt_thermalParam --algo gp_rbf
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PLOTS_DIR_DEFAULT = Path("/home/holt/Projects/SlabPT-emulator/plots/qc-emulator")

# Params we want to display on a log-x axis (raw values)
LOGX_PARAMS = {"eta_int", "eta_UM", "eps_trans"}

# ------------------------- labels

def nice_label(param: str) -> str:
    labels = {
        "v_conv": r"Convergence rate (cm/yr)",
        "t_conv": r"$t_{\rm conv}$ (Myr)",
        "v_conv_over_tconv": r"$v_{\rm conv}/t_{\rm conv}$",
        "age_SP": r"Age$_{\rm SP}$ (Ma)",
        "age_OP": r"Age$_{\rm OP}$ (Ma)",
        "dip_int": r"Initial dip (°)",
        "eta_int": r"$\eta_{\rm int}$ (Pa·s)",
        "eta_UM": r"$\eta_{\rm UM}$ (Pa·s)",
        "eta_ratio": r"$\eta_{\rm int}/\eta_{\rm UM}$",
        "eps_trans": r"$\dot\epsilon_{\rm trans}$ (s$^{-1}$)",
        "thermal_param": r"$v\,\mathrm{age}_{\rm SP}\,\sin(\mathrm{dip})$",
        "misfit": r"|Emulator − True| (°C/Myr)",
    }
    return labels.get(param, param)


# ------------------------- dataset discovery / parsing

def _depth_from_name(name: str) -> Optional[float]:
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*km_", name)
    return float(m.group(1)) if m else None


def _sorted_names(names: List[str]) -> List[str]:
    pairs = []
    for n in names:
        d = _depth_from_name(n)
        pairs.append((d if d is not None else 1e9, n))
    return [n for _, n in sorted(pairs, key=lambda t: t[0])]


# ------------------------- IO helpers

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


def _compute_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived cols if inputs exist.
    Keep consistent with your preprocess logic:
      thermal_param = v_conv * age_SP * sin(dip_int)
      eta_ratio = eta_int / eta_UM
      v_conv_over_tconv = v_conv / t_conv
    """
    out = df.copy()

    if "thermal_param" not in out.columns:
        if all(c in out.columns for c in ["v_conv", "age_SP", "dip_int"]):
            out["thermal_param"] = (
                pd.to_numeric(out["v_conv"], errors="coerce")
                * pd.to_numeric(out["age_SP"], errors="coerce")
                * np.sin(np.radians(pd.to_numeric(out["dip_int"], errors="coerce")))
            )

    if "eta_ratio" not in out.columns:
        if all(c in out.columns for c in ["eta_int", "eta_UM"]):
            ei = pd.to_numeric(out["eta_int"], errors="coerce")
            eu = pd.to_numeric(out["eta_UM"], errors="coerce")
            out["eta_ratio"] = ei / eu

    if "v_conv_over_tconv" not in out.columns:
        if all(c in out.columns for c in ["v_conv", "t_conv"]):
            v = pd.to_numeric(out["v_conv"], errors="coerce")
            t = pd.to_numeric(out["t_conv"], errors="coerce").replace(0, np.nan)
            out["v_conv_over_tconv"] = v / t

    return out


def load_bundle(data_root: Path, models_root: Path, suite: str, name: str, algo: str, yidx: int):
    """
    Load Y_true and predictions, plus metadata and model dir.
    """
    data_path = data_root / suite / name
    model_path = models_root / suite / name / algo

    meta = _load_json(data_path / "metadata.json")

    Y_raw = np.load(data_path / "Y_raw.npy")
    train_idx = np.load(data_path / "train_idx.npy")
    val_idx = np.load(data_path / "val_idx.npy")

    yhat_train = np.load(model_path / "yhat_train.npy")
    yhat_val = np.load(model_path / "yhat_val.npy")

    # Force 2D
    if Y_raw.ndim == 1:
        Y_raw = Y_raw.reshape(-1, 1)
    yhat_train = np.asarray(yhat_train).reshape(-1)
    yhat_val = np.asarray(yhat_val).reshape(-1)

    target_cols = meta["target"]["target_cols"]
    target_name = target_cols[yidx]

    y_true_train = Y_raw[train_idx, yidx].reshape(-1)
    y_true_val = Y_raw[val_idx, yidx].reshape(-1)

    return dict(
        meta=meta,
        data_path=data_path,
        model_path=model_path,
        train_idx=train_idx,
        val_idx=val_idx,
        y_true_train=y_true_train,
        y_true_val=y_true_val,
        yhat_train=yhat_train,
        yhat_val=yhat_val,
        target_name=target_name,
    )


# ------------------------- plotting

def plot_one_dataset(
    bundle: Dict[str, Any],
    df_params_full: pd.DataFrame,
    outpath: Path,
    label_thresh: float,
    dpi: int,
):
    meta = bundle["meta"]
    name = bundle["data_path"].name

    # --- Align params to the dataset rows via metadata["run_ids"]
    run_ids = meta.get("run_ids", None)
    if not run_ids:
        raise SystemExit(f"[ERR] metadata.json missing 'run_ids' for dataset: {name}")

    dfp = df_params_full.set_index("run_id", drop=False)
    missing = [rid for rid in run_ids if rid not in dfp.index]
    if missing:
        raise SystemExit(
            f"[ERR] params CSV missing {len(missing)} run_ids referenced by metadata for {name}. "
            f"First few: {missing[:10]}"
        )

    df_aligned = dfp.loc[run_ids].reset_index(drop=True)
    df_aligned = _compute_derived_columns(df_aligned)

    tr = bundle["train_idx"]
    va = bundle["val_idx"]

    df_train = df_aligned.iloc[tr].copy()
    df_val = df_aligned.iloc[va].copy()

    y_true_train = bundle["y_true_train"]
    y_true_val = bundle["y_true_val"]
    yhat_train = bundle["yhat_train"]
    yhat_val = bundle["yhat_val"]

    misfit_train = np.abs(y_true_train - yhat_train)
    misfit_val = np.abs(y_true_val - yhat_val)

    # --- plot ONLY the features used for training, but in raw param space where possible
    feat_cols: List[str] = meta.get("feature_cols", [])
    if not feat_cols:
        raise SystemExit(f"[ERR] metadata.json missing 'feature_cols' for dataset: {name}")

    # Some features may be log-transformed during preprocessing (e.g. eta_UM),
    # but we still plot the raw param on a log-x axis for interpretability.
    # If a feature is not present in df (e.g. computed), try derived-compute already done above.
    params_to_plot = [c for c in feat_cols if c in df_aligned.columns]

    # In case you trained with a derived feature but params CSV doesn't have it (we compute it),
    # keep it if present now.
    for extra in ["thermal_param", "eta_ratio", "v_conv_over_tconv"]:
        if extra in feat_cols and extra not in params_to_plot and extra in df_aligned.columns:
            params_to_plot.append(extra)

    if not params_to_plot:
        raise SystemExit(
            f"[ERR] None of metadata feature_cols exist in params table for {name}.\n"
            f"feature_cols={feat_cols}\nparams columns={list(df_aligned.columns)}"
        )

    n = len(params_to_plot)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.6 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for i, pname in enumerate(params_to_plot):
        ax = axes[i]

        x_train = pd.to_numeric(df_train[pname], errors="coerce").to_numpy(float)
        x_val = pd.to_numeric(df_val[pname], errors="coerce").to_numpy(float)

        mtr = np.isfinite(x_train) & np.isfinite(misfit_train)
        mva = np.isfinite(x_val) & np.isfinite(misfit_val)

        use_logx = pname in LOGX_PARAMS or pname in {"thermal_param", "eta_ratio"}

        # Scatter
        ax.scatter(x_train[mtr], misfit_train[mtr], s=16, alpha=0.35, label="Train", zorder=1)
        ax.scatter(x_val[mva], misfit_val[mva], s=22, alpha=0.85, label="Validation", zorder=2)

        if use_logx:
            ax.set_xscale("log")

        # Labels for outliers
        run_ids_train = df_train["run_id"].astype(str).to_numpy()
        run_ids_val = df_val["run_id"].astype(str).to_numpy()

        for xi, yi, rid in zip(x_train[mtr], misfit_train[mtr], run_ids_train[mtr]):
            if yi >= label_thresh and np.isfinite(xi) and xi > 0:
                ax.text(xi, yi, rid, fontsize=7, ha="left", va="center", zorder=3)
        for xi, yi, rid in zip(x_val[mva], misfit_val[mva], run_ids_val[mva]):
            if yi >= label_thresh and np.isfinite(xi) and xi > 0:
                ax.text(xi, yi, rid, fontsize=7, ha="left", va="center", color="red", zorder=4)

        ax.set_xlabel(nice_label(pname) + (" (log x)" if use_logx else ""))
        ax.set_ylabel(nice_label("misfit"))
        ax.grid(True, ls=":", alpha=0.4)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    axes[0].legend(frameon=False, loc="upper right")

    fig.suptitle(
        f"Emulator misfit vs features — {bundle['data_path'].name} — {bundle['model_path'].name}\n"
        f"target={bundle['target_name']}",
        fontsize=14
    )

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {outpath}")


def main():
    p = argparse.ArgumentParser(description="Plot emulator misfit vs feature columns (Train & Val).")

    p.add_argument("--data-root", default=str(Path(__file__).parent.parent / "data"),
                   help="Root containing suite folders (e.g., ./data)")
    p.add_argument("--models-root", default=str(Path(__file__).parent.parent / "models"),
                   help="Root containing suite folders (e.g., ./models)")

    p.add_argument("--suite", required=True, choices=["const-vc", "ramped-vc"])
    p.add_argument("--variant", required=True,
                   help="Dataset variant suffix after '<depth>km_' (e.g., dTdt, dTdt_thermalParam)")
    p.add_argument("--algo", default="gp_rbf",
                   help="Model subdir name under models/<suite>/<name>/ (e.g., gp_rbf, gp_m15, gp_m25, rf, etc.)")
    p.add_argument("--yidx", type=int, default=0)

    p.add_argument("--names", nargs="*", default=None,
                   help="Optional explicit dataset names (e.g., 10km_dTdt 20km_dTdt ...). "
                        "If omitted, auto-discovers all '*km_<variant>' under data/<suite>/.")
    p.add_argument("--params", default=None,
                   help="Optional params-list.<suite>.csv override. If omitted, uses metadata.json['params_path'] per dataset.")

    p.add_argument("--outdir", default=str(PLOTS_DIR_DEFAULT),
                   help=f"Output directory (default: {PLOTS_DIR_DEFAULT})")
    p.add_argument("--label-thresh", type=float, default=8.0,
                   help="Label run_id where |misfit| >= this threshold.")
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

    # Load params CSV once (prefer CLI override; else read from first dataset metadata)
    if args.params:
        params_path = Path(args.params).expanduser().resolve()
    else:
        # Use metadata params_path from first dataset (they should all match within suite)
        meta0 = _load_json(suite_dir / names[0] / "metadata.json")
        params_path = Path(meta0["params_path"]).resolve()

    if not params_path.exists():
        raise SystemExit(f"[ERR] params-list CSV not found: {params_path}")

    df_params_full = _load_params_csv(params_path)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    plotdir = outdir.joinpath(args.suite, "misfits-vs-params").resolve()
    plotdir.mkdir(parents=True, exist_ok=True)


    for name in names:
        bundle = load_bundle(
            data_root=data_root,
            models_root=models_root,
            suite=args.suite,
            name=name,
            algo=args.algo,
            yidx=args.yidx,
        )

        outpath = plotdir / f"{args.suite}_{name}_{args.algo}.png"
        plot_one_dataset(
            bundle=bundle,
            df_params_full=df_params_full,
            outpath=outpath,
            label_thresh=args.label_thresh,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()

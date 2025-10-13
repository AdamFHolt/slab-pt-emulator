#!/usr/bin/env python3
"""
Train the Slab P–T emulator on preprocessed arrays.

Inputs (produced by preprocess_training.py):
- X_std.npy, Y_std.npy : standardized features/targets (float32/64)
- X_raw.npy, Y_raw.npy : raw (unstandardized) features/targets
- train_idx.npy, val_idx.npy : index arrays for split
- metadata.json : includes scalers (X.mean/std, Y.mean/std), feature/target names, etc.

Outputs:
- model.joblib : trained scikit-learn model (MultiOutput wrapper if multi-target)
- report.json : metrics on train/val in RAW units (RMSE, MAE, R²), kernel params if GP
- yhat_train.npy, yhat_val.npy : predictions in RAW units (same shape as Y_raw subsets)
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np

# ---- Model choices
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ---------- utils

def _inverse_standardize(arr_std: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Undo z-score: arr = arr_std * std + mean. Broadcasts over columns."""
    return arr_std * std + mean

def _load_data(data_dir: Path) -> Dict[str, Any]:
    with open(data_dir / "metadata.json", "r") as f:
        meta = json.load(f)

    X_std = np.load(data_dir / "X_std.npy")
    Y_std = np.load(data_dir / "Y_std.npy")
    X_raw = np.load(data_dir / "X_raw.npy")
    Y_raw = np.load(data_dir / "Y_raw.npy")

    train_idx_path = data_dir / "train_idx.npy"
    val_idx_path   = data_dir / "val_idx.npy"

    if train_idx_path.exists() and val_idx_path.exists():
        train_idx = np.load(train_idx_path)
        val_idx   = np.load(val_idx_path)
    else:
        # Fallback: no split files (shouldn't happen if you used the preprocessor)
        n = X_std.shape[0]
        train_idx = np.arange(n, dtype=int)
        val_idx   = np.array([], dtype=int)
        print("[WARN] train_idx.npy / val_idx.npy not found; using all rows for training.")

    # scalers (lists) -> np.ndarray
    X_mu = np.asarray(meta["scalers"]["X"]["mean"], dtype=float)
    X_sd = np.asarray(meta["scalers"]["X"]["std"],  dtype=float)
    Y_mu = np.asarray(meta["scalers"]["Y"]["mean"], dtype=float)
    Y_sd = np.asarray(meta["scalers"]["Y"]["std"],  dtype=float)

    return dict(
        X_std=X_std, Y_std=Y_std, X_raw=X_raw, Y_raw=Y_raw,
        train_idx=train_idx, val_idx=val_idx,
        X_mu=X_mu, X_sd=X_sd, Y_mu=Y_mu, Y_sd=Y_sd,
        meta=meta
    )

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }

def _ensure_2d(y: np.ndarray) -> np.ndarray:
    return y.reshape(-1, 1) if y.ndim == 1 else y


# ---------- model builders

def build_gp(n_features: int, lengthscale_init: float, lengthscale_bounds: Tuple[float, float],
             noise_level_init: float, noise_bounds: Tuple[float, float],
             n_restarts: int, alpha: float, random_state: int) -> GaussianProcessRegressor:
    """
    Returns a single-target GP regressor. We'll wrap per-target with MultiOutputRegressor.
    NOTE: We train on standardized X/Y, so set normalize_y=False here.
    """
    kernel = C(1.0, (1e-3, 1e3)) * RBF(
        length_scale=np.full(n_features, lengthscale_init),
        length_scale_bounds=lengthscale_bounds
    ) + WhiteKernel(noise_level=noise_level_init, noise_level_bounds=noise_bounds)

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,                    # add'l nugget on the diagonal
        normalize_y=False,              # we already standardized Y
        n_restarts_optimizer=n_restarts,
        random_state=random_state
    )
    return gp

def build_rf(n_estimators: int, max_depth: int | None, random_state: int, n_jobs: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs
    )


# ---------- training entry

def main():
    p = argparse.ArgumentParser(description="Train GP (or RF) emulator on preprocessed data.")
    p.add_argument("--data-dir", default=str(Path(__file__).parent / "data"),
                   help="Directory containing X_std.npy, Y_std.npy, train_idx.npy, val_idx.npy, metadata.json")
    p.add_argument("--model", choices=["gp", "rf"], default="gp",
                   help="Model type: Gaussian Process (gp) or Random Forest (rf).")

    # GP hyperparams
    p.add_argument("--ls-init", type=float, default=1.0, help="RBF lengthscale initial value (in standardized X units).")
    p.add_argument("--ls-bounds", type=float, nargs=2, default=[1e-2, 1e2], help="RBF lengthscale bounds (min max).")
    p.add_argument("--noise-init", type=float, default=1e-3, help="WhiteKernel noise init (std^2) in standardized Y units.")
    p.add_argument("--noise-bounds", type=float, nargs=2, default=[1e-6, 1e-1], help="WhiteKernel noise bounds (min max).")
    p.add_argument("--alpha", type=float, default=1e-6, help="Jitter on K-diagonal for numerical stability.")
    p.add_argument("--gp-restarts", type=int, default=5, help="Optimizer restarts for kernel hyperparameters.")

    # RF hyperparams
    p.add_argument("--rf-trees", type=int, default=400, help="n_estimators for RandomForest.")
    p.add_argument("--rf-max-depth", type=int, default=None, help="Max depth for RandomForest (None = unlimited).")
    p.add_argument("--rf-jobs", type=int, default=-1, help="n_jobs for RandomForest (-1 = all cores).")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="models", help="Output directory for model + report.")
    args = p.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir  = (Path(args.out) if Path(args.out).is_absolute() else Path.cwd() / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = _load_data(data_dir)
    Xs = bundle["X_std"]; Ys = _ensure_2d(bundle["Y_std"])      # standardized
    Xr = bundle["X_raw"]; Yr = _ensure_2d(bundle["Y_raw"])      # raw for metric reporting
    tr = bundle["train_idx"]; va = bundle["val_idx"]
    X_mu, X_sd, Y_mu, Y_sd = bundle["X_mu"], bundle["X_sd"], bundle["Y_mu"], bundle["Y_sd"]
    meta = bundle["meta"]
    n_features = Xs.shape[1]
    n_targets  = Ys.shape[1]

    # Build model
    if args.model == "gp":
        base_gp = build_gp(
            n_features=n_features,
            lengthscale_init=args.ls_init,
            lengthscale_bounds=tuple(args.ls_bounds),
            noise_level_init=args.noise_init,
            noise_bounds=tuple(args.noise_bounds),
            n_restarts=args.gp_restarts,
            alpha=args.alpha,
            random_state=args.seed
        )
        model = MultiOutputRegressor(base_gp) if n_targets > 1 else base_gp
    else:
        base_rf = build_rf(
            n_estimators=args.rf_trees,
            max_depth=args.rf_max_depth,
            random_state=args.seed,
            n_jobs=args.rf_jobs
        )
        model = MultiOutputRegressor(base_rf) if n_targets > 1 else base_rf

    # Train
    Xtr, Ytr = Xs[tr], Ys[tr]
    model.fit(Xtr, Ytr.ravel() if Ytr.shape[1] == 1 else Ytr)

    # Predict (standardized space) then invert to RAW units for reporting
    Yhat_tr_std = model.predict(Xs[tr])
    Yhat_va_std = model.predict(Xs[va]) if va.size else np.empty((0, Ys.shape[1]))

    # Ensure 2D for broadcasting inverse transform
    Yhat_tr_std = _ensure_2d(np.asarray(Yhat_tr_std))
    Yhat_va_std = _ensure_2d(np.asarray(Yhat_va_std))

    Yhat_tr = _inverse_standardize(Yhat_tr_std, Y_mu, Y_sd)
    Yhat_va = _inverse_standardize(Yhat_va_std, Y_mu, Y_sd)

    # Metrics in RAW units (per-target and averaged)
    metrics: Dict[str, Any] = {"target_cols": meta["target"]["target_cols"]}
    def per_target_metrics(y_true, y_pred, prefix):
        out = {}
        for j, name in enumerate(metrics["target_cols"]):
            m = _metrics(y_true[:, j], y_pred[:, j])
            out[name] = m
        # macro-averages
        out["_macro_avg"] = {
            "rmse": float(np.mean([out[name]["rmse"] for name in metrics["target_cols"]])),
            "mae":  float(np.mean([out[name]["mae"]  for name in metrics["target_cols"]])),
            "r2":   float(np.mean([out[name]["r2"]   for name in metrics["target_cols"]])),
        }
        metrics[prefix] = out

    per_target_metrics(Yr[tr], Yhat_tr, "train")
    if va.size:
        per_target_metrics(Yr[va], Yhat_va, "val")

    # Extract GP kernel params (lengthscales etc.) if using GP
    if args.model == "gp":
        def dump_gp_params(estimator) -> Dict[str, Any]:
            k = estimator.kernel_
            return {"kernel": str(k)}
        if isinstance(model, MultiOutputRegressor):
            metrics["gp_kernels"] = [dump_gp_params(est) for est in model.estimators_]
        else:
            metrics["gp_kernels"] = [dump_gp_params(model)]

    # Save artifacts
    joblib.dump(model, out_dir / "model.joblib")
    np.save(out_dir / "yhat_train.npy", Yhat_tr)
    if va.size:
        np.save(out_dir / "yhat_val.npy", Yhat_va)

    report = {
        "model_type": args.model,
        "n_features": n_features,
        "n_targets": n_targets,
        "train_size": int(tr.size),
        "val_size": int(va.size),
        "feature_cols": meta["feature_cols"],
        "target_cols": metrics["target_cols"],
        "metrics": metrics,
        "data_dir": str(data_dir),
        "seed": args.seed,
        "gp_hparams": {
            "ls_init": args.ls_init, "ls_bounds": args.ls_bounds,
            "noise_init": args.noise_init, "noise_bounds": args.noise_bounds,
            "alpha": args.alpha, "restarts": args.gp_restarts
        } if args.model == "gp" else None,
        "rf_hparams": {
            "n_estimators": args.rf_trees,
            "max_depth": args.rf_max_depth
        } if args.model == "rf" else None,
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("[OK] Trained model saved to:", out_dir / "model.joblib")
    if "val" in metrics:
        print(f"[OK] Val macro RMSE: {metrics['val']['_macro_avg']['rmse']:.3f} "
              f"R²: {metrics['val']['_macro_avg']['r2']:.3f}")
    else:
        print(f"[OK] Train macro RMSE: {metrics['train']['_macro_avg']['rmse']:.3f} "
              f"R²: {metrics['train']['_macro_avg']['r2']:.3f}")


if __name__ == "__main__":
    main()

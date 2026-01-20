#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np

from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C


def _inverse_standardize(arr_std: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return arr_std * std + mean

def _ensure_2d(y: np.ndarray) -> np.ndarray:
    return y.reshape(-1, 1) if y.ndim == 1 else y

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }

def _load_data(data_root: Path, data_name: str) -> Dict[str, Any]:
    data_path = (data_root / data_name).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_path}")

    meta_path = data_path / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in: {data_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    X_std = np.load(data_path / "X_std.npy")
    Y_std = np.load(data_path / "Y_std.npy")
    X_raw = np.load(data_path / "X_raw.npy")
    Y_raw = np.load(data_path / "Y_raw.npy")

    train_idx_path = data_path / "train_idx.npy"
    val_idx_path   = data_path / "val_idx.npy"
    if train_idx_path.exists() and val_idx_path.exists():
        train_idx = np.load(train_idx_path)
        val_idx   = np.load(val_idx_path)
    else:
        n = X_std.shape[0]
        train_idx = np.arange(n, dtype=int)
        val_idx   = np.array([], dtype=int)
        print("[WARN] train_idx.npy / val_idx.npy not found; using all rows for training.")

    Y_mu = np.asarray(meta["scalers"]["Y"]["mean"], dtype=float)
    Y_sd = np.asarray(meta["scalers"]["Y"]["std"],  dtype=float)

    return dict(
        X_std=X_std, Y_std=Y_std, X_raw=X_raw, Y_raw=Y_raw,
        train_idx=train_idx, val_idx=val_idx,
        Y_mu=Y_mu, Y_sd=Y_sd,
        meta=meta,
        data_path=str(data_path),
        meta_path=str(meta_path),
    )


def build_gp(n_features, lengthscale_init, lengthscale_bounds, noise_level_init, noise_bounds,
             n_restarts, alpha, random_state, kernel_name):

    if kernel_name == "rbf":
        base = RBF(length_scale=np.full(n_features, lengthscale_init),
                   length_scale_bounds=lengthscale_bounds)
    elif kernel_name == "matern25":
        base = Matern(length_scale=np.full(n_features, lengthscale_init),
                      length_scale_bounds=lengthscale_bounds, nu=2.5)
    else:  # matern15
        base = Matern(length_scale=np.full(n_features, lengthscale_init),
                      length_scale_bounds=lengthscale_bounds, nu=1.5)

    kernel = C(1.0, (1e-3, 1e3)) * base + WhiteKernel(
        noise_level=noise_level_init, noise_level_bounds=noise_bounds
    )
    return GaussianProcessRegressor(
        kernel=kernel, alpha=alpha, normalize_y=False,
        n_restarts_optimizer=n_restarts, random_state=random_state
    )


def build_rf(n_estimators: int, max_depth: int | None, random_state: int, n_jobs: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs
    )


def main():
    p = argparse.ArgumentParser(description="Train GP (or RF) emulator on preprocessed data.")
    p.add_argument("--data-root", default=str(Path(__file__).parent / "data"),
                   help="Directory containing dataset folders (e.g., ./data/const-vc).")
    p.add_argument("--data-name", type=str, required=True,
                   help="Dataset folder name within --data-root (e.g., 50km_dTdt_thermalParam).")
    p.add_argument("--model", choices=["gp", "rf"], default="gp")

    # GP hyperparams
    p.add_argument("--ls-init", type=float, default=1.0)
    p.add_argument("--ls-bounds", type=float, nargs=2, default=[1e-2, 1e2])
    p.add_argument("--noise-init", type=float, default=1e-3)
    p.add_argument("--noise-bounds", type=float, nargs=2, default=[1e-6, 1e-1])
    p.add_argument("--alpha", type=float, default=1e-6)
    p.add_argument("--gp-restarts", type=int, default=5)
    p.add_argument("--kernel", choices=["rbf","matern15","matern25"], default="matern15")

    # RF hyperparams
    p.add_argument("--rf-trees", type=int, default=400)
    p.add_argument("--rf-max-depth", type=int, default=None)
    p.add_argument("--rf-jobs", type=int, default=-1)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="models", help="Output directory for model + report.")
    args = p.parse_args()

    kernel_suffix = ""
    if args.model == "gp":
        kernel_suffix = {"rbf": "_rbf", "matern15": "_m15", "matern25": "_m25"}[args.kernel]

    data_root = Path(args.data_root).resolve()
    bundle = _load_data(data_root, args.data_name)

    Xs = bundle["X_std"]
    Ys = _ensure_2d(bundle["Y_std"])
    Yr = _ensure_2d(bundle["Y_raw"])
    tr = bundle["train_idx"]
    va = bundle["val_idx"]
    Y_mu, Y_sd = bundle["Y_mu"], bundle["Y_sd"]
    meta = bundle["meta"]

    n_features = Xs.shape[1]
    n_targets  = Ys.shape[1]

    # Output dir
    out_root = Path(args.out).resolve()
    model_dir_name = args.model + kernel_suffix
    out_dir = out_root / args.data_name / model_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    if args.model == "gp":
        base = build_gp(
            n_features=n_features,
            lengthscale_init=args.ls_init,
            lengthscale_bounds=tuple(args.ls_bounds),
            noise_level_init=args.noise_init,
            noise_bounds=tuple(args.noise_bounds),
            n_restarts=args.gp_restarts,
            alpha=args.alpha,
            random_state=args.seed,
            kernel_name=args.kernel
        )
        model = MultiOutputRegressor(base) if n_targets > 1 else base
    else:
        base = build_rf(
            n_estimators=args.rf_trees,
            max_depth=args.rf_max_depth,
            random_state=args.seed,
            n_jobs=args.rf_jobs
        )
        model = MultiOutputRegressor(base) if n_targets > 1 else base

    # Train
    Xtr, Ytr = Xs[tr], Ys[tr]
    model.fit(Xtr, Ytr.ravel() if Ytr.shape[1] == 1 else Ytr)

    # Predict in standardized space then invert to RAW units
    Yhat_tr_std = _ensure_2d(np.asarray(model.predict(Xs[tr])))
    Yhat_va_std = _ensure_2d(np.asarray(model.predict(Xs[va]))) if va.size else np.empty((0, Ys.shape[1]))

    Yhat_tr = _inverse_standardize(Yhat_tr_std, Y_mu, Y_sd)
    Yhat_va = _inverse_standardize(Yhat_va_std, Y_mu, Y_sd) if va.size else Yhat_va_std

    # Metrics in RAW units
    metrics: Dict[str, Any] = {"target_cols": meta["target"]["target_cols"]}

    def per_target_metrics(y_true, y_pred, prefix):
        out = {}
        for j, name in enumerate(metrics["target_cols"]):
            out[name] = _metrics(y_true[:, j], y_pred[:, j])
        out["_macro_avg"] = {
            "rmse": float(np.mean([out[name]["rmse"] for name in metrics["target_cols"]])),
            "mae":  float(np.mean([out[name]["mae"]  for name in metrics["target_cols"]])),
            "r2":   float(np.mean([out[name]["r2"]   for name in metrics["target_cols"]])),
        }
        metrics[prefix] = out

    per_target_metrics(Yr[tr], Yhat_tr, "train")
    if va.size:
        per_target_metrics(Yr[va], Yhat_va, "val")

    # GP kernel dump
    if args.model == "gp":
        def dump_gp_params(est) -> Dict[str, Any]:
            return {"kernel": str(est.kernel_)}
        if isinstance(model, MultiOutputRegressor):
            metrics["gp_kernels"] = [dump_gp_params(est) for est in model.estimators_]
        else:
            metrics["gp_kernels"] = [dump_gp_params(model)]

    # Save artifacts
    joblib.dump(model, out_dir / "model.joblib")
    np.save(out_dir / "yhat_train.npy", Yhat_tr)
    if va.size:
        np.save(out_dir / "yhat_val.npy", Yhat_va)

    # Copy metadata alongside model (helps later)
    shutil.copy2(bundle["meta_path"], out_dir / "metadata.json")

    report = {
        "model_type": args.model,
        "kernel": (args.kernel if args.model == "gp" else None),
        "n_features": n_features,
        "n_targets": n_targets,
        "train_size": int(tr.size),
        "val_size": int(va.size),
        "feature_cols": meta["feature_cols"],
        "target_cols": metrics["target_cols"],
        "metrics": metrics,
        "data_dir": bundle["data_path"],
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

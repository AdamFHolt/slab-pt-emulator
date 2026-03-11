#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return obj


def _load_profile_on_grid(csv_path: Path, depth_grid: np.ndarray) -> np.ndarray:
    # Reuse the same interpolation assumptions as the plotting scripts:
    # - read one profile CSV
    # - sort by depth
    # - drop duplicate depths
    # - interpolate onto the common PCA depth grid
    df = pd.read_csv(csv_path)
    need = {"depth_km", "T_C"}
    if not need.issubset(df.columns):
        raise ValueError(f"{csv_path} missing required columns {need}")

    z = pd.to_numeric(df["depth_km"], errors="coerce").to_numpy(float)
    t = pd.to_numeric(df["T_C"], errors="coerce").to_numpy(float)
    mask = np.isfinite(z) & np.isfinite(t)
    z = z[mask]
    t = t[mask]

    order = np.argsort(z)
    z = z[order]
    t = t[order]

    z_unique, idx = np.unique(z, return_index=True)
    t_unique = t[idx]

    out = np.interp(depth_grid, z_unique, t_unique, left=np.nan, right=np.nan)
    if not np.isfinite(out).all():
        raise ValueError(f"{csv_path} does not fully cover depth grid")
    return out


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # We use the usual coefficient of determination on flattened arrays.
    # This gives one interpretable summary for a whole split even when the
    # target is multi-output (multiple PCs or full depth profiles).
    yt = np.ravel(np.asarray(y_true, dtype=float))
    yp = np.ravel(np.asarray(y_pred, dtype=float))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _per_target_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_cols: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for j, name in enumerate(target_cols):
        out[name] = {
            "rmse": _rmse(y_true[:, j], y_pred[:, j]),
            "mae": _mae(y_true[:, j], y_pred[:, j]),
            "r2": _r2(y_true[:, j], y_pred[:, j]),
        }

    out["_macro_avg"] = {
        "rmse": float(np.mean([out[name]["rmse"] for name in target_cols])),
        "mae": float(np.mean([out[name]["mae"] for name in target_cols])),
        "r2": float(np.mean([out[name]["r2"] for name in target_cols])),
    }
    return out


def _profile_metrics(true_profiles: np.ndarray, pred_profiles: np.ndarray) -> dict[str, Any]:
    # These metrics are in physical profile space (temperature vs depth), which
    # is the most interpretable place to judge whether the PCA emulator is good
    # enough for scientific use.
    err = true_profiles - pred_profiles
    per_run_rmse = np.sqrt(np.mean(err ** 2, axis=1))
    per_run_mae = np.mean(np.abs(err), axis=1)
    rmse_by_depth = np.sqrt(np.mean(err ** 2, axis=0))

    return {
        "rmse": _rmse(true_profiles, pred_profiles),
        "mae": _mae(true_profiles, pred_profiles),
        "r2": _r2(true_profiles, pred_profiles),
        "per_run_rmse": _distribution_summary(per_run_rmse),
        "per_run_mae": _distribution_summary(per_run_mae),
        "rmse_by_depth": rmse_by_depth.tolist(),
    }


def _load_split_predictions(model_dir: Path, split: str) -> np.ndarray:
    path = model_dir / f"yhat_{split}.npy"
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compute score-space and reconstructed-profile quality metrics for one profile-PCA model."
    )
    ap.add_argument("--dataset-dir", required=True, help="Path to one profile-PCA dataset folder.")
    ap.add_argument("--model-dir", required=True, help="Path to one trained model artifact dir.")
    ap.add_argument(
        "--json-out",
        default=None,
        help="Optional explicit output path. Default: <model-dir>/profile_pca_quality.json",
    )
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    meta = _load_json(dataset_dir / "metadata.json")
    model_report = _load_json(model_dir / "report.json")

    depth_grid = np.asarray(meta["profile"]["depth_grid_km"], dtype=float)
    source_paths = [Path(p) for p in meta["profile"]["source_paths"]]
    train_idx = np.asarray(np.load(dataset_dir / "train_idx.npy"), dtype=int)
    val_idx = np.asarray(np.load(dataset_dir / "val_idx.npy"), dtype=int)

    y_true_scores = np.load(dataset_dir / "Y_raw.npy")
    scores_raw = np.load(dataset_dir / "scores_raw.npy")
    mean_profile = np.load(dataset_dir / "pca_mean_profile.npy")
    components = np.load(dataset_dir / "pca_components.npy")
    score_scale = np.load(dataset_dir / "pca_score_scale.npy")
    score_space = str(meta.get("target", {}).get("score_space", "raw")).strip().lower()

    # Load the original raw temperature profiles from the source CSV files so
    # reconstruction error is measured in physical units, not only in PCA space.
    true_profiles = np.vstack([_load_profile_on_grid(path, depth_grid) for path in source_paths])

    # This is the "best possible" reconstruction using the retained PCs only.
    # It isolates truncation error from emulator prediction error.
    pca_recon_all = mean_profile[None, :] + scores_raw @ components

    splits: dict[str, np.ndarray] = {"train": train_idx, "val": val_idx}
    metrics: dict[str, Any] = {}

    for split_name, split_idx in splits.items():
        if split_idx.size == 0:
            continue

        y_pred_scores = _load_split_predictions(model_dir, split_name)
        if y_pred_scores.shape[0] != split_idx.size:
            raise RuntimeError(f"{split_name} prediction row count does not match split size.")

        y_true_split = y_true_scores[split_idx]
        score_metrics = _per_target_metrics(
            y_true_split,
            y_pred_scores,
            list(meta["target"]["target_cols"]),
        )
        score_metrics["per_run_rmse"] = _distribution_summary(
            np.sqrt(np.mean((y_true_split - y_pred_scores) ** 2, axis=1))
        )

        # Models may predict raw PCA scores or whitened scores depending on the
        # dataset configuration. We convert back to raw score space before
        # reconstructing temperature profiles.
        if score_space == "whitened":
            pred_scores_raw = y_pred_scores * score_scale[None, :]
        else:
            pred_scores_raw = y_pred_scores

        emu_recon_split = mean_profile[None, :] + pred_scores_raw @ components
        true_profiles_split = true_profiles[split_idx]
        pca_recon_split = pca_recon_all[split_idx]

        metrics[split_name] = {
            "n_rows": int(split_idx.size),
            "score_space": score_metrics,
            "profile_space": {
                "emulator_reconstruction": _profile_metrics(true_profiles_split, emu_recon_split),
                "pca_truncation_baseline": _profile_metrics(true_profiles_split, pca_recon_split),
            },
        }

    out_path = Path(args.json_out).resolve() if args.json_out else (model_dir / "profile_pca_quality.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 1,
        "dataset_mode": "profile-pca",
        "dataset_dir": dataset_dir.as_posix(),
        "model_dir": model_dir.as_posix(),
        "suite": meta["suite"],
        "dataset_name": meta["dataset_name"],
        "model_type": model_report.get("model_type"),
        "model_kernel": model_report.get("kernel"),
        "score_space": score_space,
        "target_cols": meta["target"]["target_cols"],
        "depth_grid_km": depth_grid.tolist(),
        "metrics": metrics,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[OK] wrote quality report: {out_path}")
    for split_name in ("train", "val"):
        split_metrics = metrics.get(split_name)
        if not split_metrics:
            continue
        score_rmse = split_metrics["score_space"]["_macro_avg"]["rmse"]
        recon_rmse = split_metrics["profile_space"]["emulator_reconstruction"]["rmse"]
        pca_rmse = split_metrics["profile_space"]["pca_truncation_baseline"]["rmse"]
        print(
            f"[OK] split={split_name} "
            f"score_rmse={score_rmse:.6f} "
            f"profile_rmse={recon_rmse:.6f} "
            f"pca_only_rmse={pca_rmse:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

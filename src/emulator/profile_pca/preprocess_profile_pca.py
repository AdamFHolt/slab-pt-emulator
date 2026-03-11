#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]


@dataclass(frozen=True)
class ProfileSlice:
    run_id: str
    profile_path: Path
    time_myr: float
    depth_km: np.ndarray
    temp_c: np.ndarray


def _load_params(params_path: Path) -> pd.DataFrame:
    df = pd.read_csv(params_path)
    if "run_id" not in df.columns:
        df = df.copy()
        df["run_id"] = [f"{i:03d}" for i in range(len(df))]
    else:
        if np.issubdtype(df["run_id"].dtype, np.number):
            df["run_id"] = df["run_id"].astype(int).map(lambda i: f"{i:03d}")
        else:
            df["run_id"] = df["run_id"].astype(str).str.zfill(3)
    return df


def _default_feature_cols(suite: str) -> list[str]:
    if suite == "const-vc":
        return ["v_conv", "age_SP", "age_OP", "dip_int", "eta_UM"]
    return ["v_conv", "t_conv", "age_SP", "age_OP", "dip_int", "eta_UM"]


def _read_profile_file(path: Path) -> tuple[float, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    need = {"time_Myr", "depth_km", "T_C"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} missing columns {need}")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["time_Myr", "depth_km", "T_C"]).copy()
    if df.empty:
        raise ValueError(f"{path} has no finite profile rows")

    df["time_Myr"] = pd.to_numeric(df["time_Myr"], errors="coerce")
    df["depth_km"] = pd.to_numeric(df["depth_km"], errors="coerce")
    df["T_C"] = pd.to_numeric(df["T_C"], errors="coerce")
    df = df.dropna(subset=["time_Myr", "depth_km", "T_C"]) 

    tvals = df["time_Myr"].to_numpy(float)
    tuniq = np.unique(tvals)
    if tuniq.size != 1:
        raise ValueError(f"{path} expected single-time profile, got {tuniq.size} times")

    depth = df["depth_km"].to_numpy(float)
    temp = df["T_C"].to_numpy(float)
    order = np.argsort(depth)
    depth = depth[order]
    temp = temp[order]

    depth_unique, idx = np.unique(depth, return_index=True)
    temp_unique = temp[idx]

    return float(tuniq[0]), depth_unique, temp_unique


def _discover_profile_slices(analysis_dir: Path, target_time_myr: float, tol_myr: float) -> list[ProfileSlice]:
    out: list[ProfileSlice] = []
    for run_dir in sorted(analysis_dir.glob("run_*")):
        if not run_dir.is_dir():
            continue

        m = re.match(r"run_(\d+)$", run_dir.name)
        if not m:
            continue
        run_id = m.group(1).zfill(3)

        best: tuple[float, Path, float, np.ndarray, np.ndarray] | None = None
        for p in sorted(run_dir.glob("Tprof_*.csv")):
            try:
                t, z, T = _read_profile_file(p)
            except Exception:
                continue
            dt = abs(t - target_time_myr)
            if best is None or dt < best[0]:
                best = (dt, p, t, z, T)

        if best is None:
            continue

        dt, p, t, z, T = best
        if dt <= tol_myr:
            out.append(ProfileSlice(run_id=run_id, profile_path=p, time_myr=t, depth_km=z, temp_c=T))

    return out


def _build_common_depth_grid(slices: list[ProfileSlice], depth_min_km: float | None, depth_max_km: float | None,
                             depth_step_km: float) -> np.ndarray:
    if not slices:
        raise ValueError("No profile slices available to build depth grid.")

    z_min_data = max(float(np.min(s.depth_km)) for s in slices)
    z_max_data = min(float(np.max(s.depth_km)) for s in slices)

    z_lo = z_min_data if depth_min_km is None else max(z_min_data, float(depth_min_km))
    z_hi = z_max_data if depth_max_km is None else min(z_max_data, float(depth_max_km))

    if not np.isfinite(z_lo) or not np.isfinite(z_hi) or z_hi <= z_lo:
        raise ValueError("Invalid overlapping depth range across runs.")

    n = int(np.floor((z_hi - z_lo) / depth_step_km)) + 1
    if n < 2:
        raise ValueError("Depth grid has fewer than 2 points; adjust depth range/step.")

    return z_lo + np.arange(n, dtype=float) * float(depth_step_km)


def _interpolate_profiles(slices: list[ProfileSlice], depth_grid: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    run_ids: list[str] = []
    times: list[float] = []
    mats: list[np.ndarray] = []
    src_paths: list[str] = []

    for s in slices:
        ti = np.interp(depth_grid, s.depth_km, s.temp_c, left=np.nan, right=np.nan)
        if not np.isfinite(ti).all():
            continue
        run_ids.append(s.run_id)
        times.append(s.time_myr)
        mats.append(ti)
        src_paths.append(s.profile_path.as_posix())

    if not mats:
        raise ValueError("No valid interpolated profiles after depth-grid intersection.")

    return run_ids, np.asarray(times, dtype=float), np.vstack(mats), src_paths


def _standardize(arr: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    mu = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0, ddof=0)
    sd_safe = np.where(sd == 0.0, 1.0, sd)
    arr_std = (arr - mu) / sd_safe
    return arr_std, {"mean": mu.tolist(), "std": sd_safe.tolist()}


def _split_indices(n_rows: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n_rows, dtype=int)
    rng.shuffle(idx)
    n_val = int(round(val_frac * n_rows))
    val_idx = np.sort(idx[:n_val])
    train_idx = np.sort(idx[n_val:])
    return train_idx, val_idx


def main() -> int:
    ap = argparse.ArgumentParser(description="Build profile-PCA emulator datasets from Tprof per-run outputs.")
    ap.add_argument("--suite", choices=["const-vc", "ramped-vc"], required=True)
    ap.add_argument("--analysis-root", default=None,
                    help="Path to suite analysis dir containing run_*/Tprof_*.csv. Defaults to subd-model-runs/<suite>/analysis.")
    ap.add_argument("--params", default=None,
                    help="Path to params-list.<suite>.csv. Defaults to data/params/params-list.<suite>.csv.")
    ap.add_argument("--target-time-myr", type=float, default=5.0,
                    help="Requested profile time (Myr). Nearest Tprof per run is used within tolerance.")
    ap.add_argument("--time-tol-myr", type=float, default=0.025,
                    help="Max |time-selected - target-time| per run.")
    ap.add_argument("--depth-min-km", type=float, default=0.0)
    ap.add_argument("--depth-max-km", type=float, default=80.0)
    ap.add_argument("--depth-step-km", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=5, help="Number of retained PCA components.")
    ap.add_argument("--score-space", choices=["raw", "whitened"], default="raw",
                    help="Training target space for PCA scores: raw or variance-whitened.")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset-name", default=None,
                    help="Output dataset folder name. Default: profileT_pca_t<target-time>Myr")
    ap.add_argument("--outdir", default=None,
                    help="Root for suite datasets. Default: src/emulator/data/<suite>")
    args = ap.parse_args()

    if args.k < 1:
        raise ValueError("--k must be >= 1")
    if not (0.0 <= args.val_frac < 1.0):
        raise ValueError("--val-frac must satisfy 0 <= val-frac < 1")
    if args.depth_step_km <= 0:
        raise ValueError("--depth-step-km must be > 0")

    analysis_root = (Path(args.analysis_root).resolve() if args.analysis_root
                     else (REPO_ROOT / "subd-model-runs" / args.suite / "analysis").resolve())
    params_path = (Path(args.params).resolve() if args.params
                   else (REPO_ROOT / "data" / "params" / f"params-list.{args.suite}.csv").resolve())

    if not analysis_root.exists():
        raise FileNotFoundError(f"Analysis directory not found: {analysis_root}")
    if not params_path.exists():
        raise FileNotFoundError(f"Params file not found: {params_path}")

    dataset_name = args.dataset_name
    if not dataset_name:
        tlabel = str(args.target_time_myr).replace(".", "p")
        dataset_name = f"profileT_pca_t{tlabel}Myr"

    out_root = (Path(args.outdir).resolve() if args.outdir
                else (REPO_ROOT / "src" / "emulator" / "data" / args.suite).resolve())
    out_dir = out_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    params_df = _load_params(params_path)
    feat_cols = [c for c in _default_feature_cols(args.suite) if c in params_df.columns]
    if not feat_cols:
        raise ValueError("No expected feature columns found in params file.")

    slices = _discover_profile_slices(analysis_root, args.target_time_myr, args.time_tol_myr)
    if not slices:
        raise RuntimeError("No profile slices matched target time within tolerance.")

    depth_grid = _build_common_depth_grid(
        slices,
        depth_min_km=args.depth_min_km,
        depth_max_km=args.depth_max_km,
        depth_step_km=args.depth_step_km,
    )

    run_ids, times_sel, temp_matrix, source_paths = _interpolate_profiles(slices, depth_grid)

    params_df = params_df.copy()
    params_df["run_id"] = params_df["run_id"].astype(str).str.zfill(3)

    prof_df = pd.DataFrame({"run_id": run_ids, "time_selected_myr": times_sel, "source_path": source_paths})
    merged = pd.merge(prof_df, params_df, on="run_id", how="inner").drop_duplicates(subset=["run_id"]).copy()
    if merged.empty:
        raise RuntimeError("No overlap between discovered run profiles and parameter rows.")

    temp_by_run = {rid: temp_matrix[i] for i, rid in enumerate(run_ids)}
    keep_run_ids = merged["run_id"].tolist()

    X_raw = merged[feat_cols].to_numpy(dtype=float)
    Y_profile = np.vstack([temp_by_run[rid] for rid in keep_run_ids])

    finite_mask = np.isfinite(X_raw).all(axis=1) & np.isfinite(Y_profile).all(axis=1)
    dropped = int((~finite_mask).sum())
    merged = merged.loc[finite_mask].reset_index(drop=True)
    X_raw = X_raw[finite_mask]
    Y_profile = Y_profile[finite_mask]

    if X_raw.shape[0] < 3:
        raise RuntimeError("Too few valid rows after filtering; need at least 3.")

    train_idx, val_idx = _split_indices(X_raw.shape[0], args.val_frac, args.seed)
    if train_idx.size < 2:
        raise RuntimeError("Too few train rows after split; increase dataset size or reduce val-frac.")

    profile_mean = np.mean(Y_profile[train_idx], axis=0)
    Y_centered = Y_profile - profile_mean

    k_eff = min(args.k, train_idx.size, Y_profile.shape[1])
    if k_eff < 1:
        raise RuntimeError("Effective PCA component count is < 1.")

    pca = PCA(n_components=k_eff, svd_solver="full", random_state=args.seed)
    pca.fit(Y_centered[train_idx])
    scores_raw = pca.transform(Y_centered)
    score_scale = np.sqrt(np.maximum(pca.explained_variance_, 1e-12))
    scores_whitened = scores_raw / score_scale

    y_target = scores_raw if args.score_space == "raw" else scores_whitened

    X_std, X_scaler = _standardize(X_raw)
    if args.score_space == "whitened":
        # Keep whitened targets in their natural unit-variance space for training.
        Y_std = y_target.copy()
        Y_scaler = {
            "mean": np.zeros(k_eff, dtype=float).tolist(),
            "std": np.ones(k_eff, dtype=float).tolist(),
        }
    else:
        Y_std, Y_scaler = _standardize(y_target)

    np.save(out_dir / "X_raw.npy", X_raw)
    np.save(out_dir / "X_std.npy", X_std)
    np.save(out_dir / "Y_raw.npy", y_target)
    np.save(out_dir / "Y_std.npy", Y_std)
    np.save(out_dir / "scores_raw.npy", scores_raw)
    np.save(out_dir / "scores_whitened.npy", scores_whitened)
    np.save(out_dir / "pca_score_scale.npy", score_scale)
    np.save(out_dir / "train_idx.npy", train_idx)
    np.save(out_dir / "val_idx.npy", val_idx)

    np.save(out_dir / "pca_mean_profile.npy", profile_mean)
    np.save(out_dir / "pca_components.npy", pca.components_)
    np.save(out_dir / "pca_explained_variance_ratio.npy", pca.explained_variance_ratio_)

    meta: dict[str, Any] = {
        "suite": args.suite,
        "dataset_mode": "profile-pca",
        "dataset_name": dataset_name,
        "feature_cols": feat_cols,
        "target": {
            "target_kind": "profile_pca_scores",
            "target_cols": [f"PC{i+1}" for i in range(k_eff)],
            "source_variable": "T_C",
            "profile_space": "T(depth_km)",
            "score_space": args.score_space,
        },
        "profile": {
            "analysis_root": analysis_root.as_posix(),
            "target_time_myr": float(args.target_time_myr),
            "time_tolerance_myr": float(args.time_tol_myr),
            "time_selected_myr": merged["time_selected_myr"].astype(float).tolist(),
            "run_ids": merged["run_id"].astype(str).tolist(),
            "source_paths": merged["source_path"].astype(str).tolist(),
            "depth_grid_km": depth_grid.tolist(),
            "n_depth": int(depth_grid.size),
            "dropped_nonfinite_rows": dropped,
        },
        "pca": {
            "k": int(k_eff),
            "fit_on": "train_split_only",
            "centered_only": True,
            "score_scale_file": "pca_score_scale.npy",
            "mean_profile_file": "pca_mean_profile.npy",
            "components_file": "pca_components.npy",
            "explained_variance_ratio_file": "pca_explained_variance_ratio.npy",
        },
        "scalers": {
            "X": X_scaler,
            "Y": Y_scaler,
        },
        "split": {
            "val_frac": float(args.val_frac),
            "seed": int(args.seed),
            "n_train": int(train_idx.size),
            "n_val": int(val_idx.size),
        },
        "cli_args": vars(args),
    }

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Built profile-PCA dataset: {out_dir}")
    print(f"[OK] Rows: {X_raw.shape[0]} (train={train_idx.size}, val={val_idx.size})")
    print(f"[OK] Depth grid: {depth_grid[0]:.1f}..{depth_grid[-1]:.1f} km, n={depth_grid.size}")
    print(f"[OK] PCA components: k={k_eff}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_profile_on_grid(csv_path: Path, depth_grid: np.ndarray) -> np.ndarray:
    df = pd.read_csv(csv_path)
    need = {"depth_km", "T_C"}
    if not need.issubset(df.columns):
        raise ValueError(f"{csv_path} missing required columns {need}")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["depth_km", "T_C"]).copy()
    if df.empty:
        raise ValueError(f"{csv_path} has no valid profile rows")

    z = pd.to_numeric(df["depth_km"], errors="coerce").to_numpy(float)
    T = pd.to_numeric(df["T_C"], errors="coerce").to_numpy(float)
    mask = np.isfinite(z) & np.isfinite(T)
    z = z[mask]
    T = T[mask]
    order = np.argsort(z)
    z = z[order]
    T = T[order]

    z_u, idx = np.unique(z, return_index=True)
    T_u = T[idx]
    if z_u.size < 2:
        raise ValueError(f"{csv_path} has insufficient depth points")

    Ti = np.interp(depth_grid, z_u, T_u, left=np.nan, right=np.nan)
    if not np.isfinite(Ti).all():
        raise ValueError(f"{csv_path} does not fully cover depth grid")
    return Ti


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot true profiles vs PCA reconstructions for a preprocessed profile-PCA dataset.")
    ap.add_argument("--dataset-dir", required=True, help="Path to one dataset folder (contains metadata + PCA files).")
    ap.add_argument("--split", choices=["val", "train", "all"], default="val",
                    help="Which rows to sample for overlay plot (default: val).")
    ap.add_argument("--n-samples", type=int, default=20, help="Number of profiles to overlay.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="Output PNG path.")
    args = ap.parse_args()

    ds = Path(args.dataset_dir).resolve()
    with open(ds / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    depth_grid = np.asarray(meta["profile"]["depth_grid_km"], dtype=float)
    source_paths = [Path(p) for p in meta["profile"]["source_paths"]]

    scores = np.load(ds / "scores_raw.npy")
    mean_profile = np.load(ds / "pca_mean_profile.npy")
    components = np.load(ds / "pca_components.npy")

    true_profiles = np.vstack([_load_profile_on_grid(p, depth_grid) for p in source_paths])
    recon_profiles = mean_profile[None, :] + scores @ components

    if true_profiles.shape != recon_profiles.shape:
        raise RuntimeError("Profile shape mismatch between true and reconstructed arrays.")

    train_idx = np.load(ds / "train_idx.npy") if (ds / "train_idx.npy").exists() else np.arange(true_profiles.shape[0])
    val_idx = np.load(ds / "val_idx.npy") if (ds / "val_idx.npy").exists() else np.array([], dtype=int)

    if args.split == "val" and val_idx.size > 0:
        pool = val_idx
    elif args.split == "train":
        pool = train_idx
    else:
        pool = np.arange(true_profiles.shape[0], dtype=int)

    rng = np.random.default_rng(args.seed)
    if args.n_samples > 0 and pool.size > args.n_samples:
        sample_idx = np.sort(rng.choice(pool, size=args.n_samples, replace=False))
    else:
        sample_idx = np.sort(pool)

    true_s = true_profiles[sample_idx]
    recon_s = recon_profiles[sample_idx]

    rmse_global = float(np.sqrt(np.mean((true_profiles - recon_profiles) ** 2)))
    rmse_depth = np.sqrt(np.mean((true_profiles - recon_profiles) ** 2, axis=0))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True)

    for i in range(true_s.shape[0]):
        ax0.plot(true_s[i], depth_grid, color="0.65", lw=0.9, alpha=0.55)
        ax0.plot(recon_s[i], depth_grid, color="tab:blue", lw=1.0, alpha=0.75)

    ax0.set_xlabel("Temperature (C)")
    ax0.set_ylabel("Depth (km)")
    ax0.set_title(f"True (gray) vs PCA recon (blue) | split={args.split}, n={true_s.shape[0]}")
    ax0.grid(True, ls=":", alpha=0.35)
    ax0.invert_yaxis()

    med_true = np.median(true_profiles, axis=0)
    med_recon = np.median(recon_profiles, axis=0)
    p05_true = np.percentile(true_profiles, 5, axis=0)
    p95_true = np.percentile(true_profiles, 95, axis=0)
    p05_recon = np.percentile(recon_profiles, 5, axis=0)
    p95_recon = np.percentile(recon_profiles, 95, axis=0)

    ax1.plot(med_true, depth_grid, color="0.2", lw=2.0, label="true median")
    ax1.plot(med_recon, depth_grid, color="tab:blue", lw=2.0, label="recon median")
    ax1.fill_betweenx(depth_grid, p05_true, p95_true, color="0.6", alpha=0.20, label="true 5-95%")
    ax1.fill_betweenx(depth_grid, p05_recon, p95_recon, color="tab:blue", alpha=0.18, label="recon 5-95%")

    ax1_t = ax1.twiny()
    ax1_t.plot(rmse_depth, depth_grid, color="tab:red", lw=1.7, alpha=0.85)
    ax1_t.set_xlabel("RMSE by depth (C)")

    ax1.set_xlabel("Temperature (C)")
    ax1.set_ylabel("Depth (km)")
    ax1.set_title(f"Distribution + depth RMSE | global RMSE={rmse_global:.3f} C")
    ax1.grid(True, ls=":", alpha=0.35)
    ax1.invert_yaxis()
    ax1.legend(loc="lower right", fontsize=8)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"[OK] Saved: {out_path}")
    print(f"[OK] global_rmse_C={rmse_global:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

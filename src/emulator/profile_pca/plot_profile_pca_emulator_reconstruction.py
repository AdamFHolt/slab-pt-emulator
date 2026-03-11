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
    z = pd.to_numeric(df["depth_km"], errors="coerce").to_numpy(float)
    t = pd.to_numeric(df["T_C"], errors="coerce").to_numpy(float)
    mask = np.isfinite(z) & np.isfinite(t)
    z = z[mask]
    t = t[mask]
    order = np.argsort(z)
    z = z[order]
    t = t[order]
    zu, idx = np.unique(z, return_index=True)
    tu = t[idx]
    ti = np.interp(depth_grid, zu, tu, left=np.nan, right=np.nan)
    if not np.isfinite(ti).all():
        raise ValueError(f"{csv_path} does not fully cover depth grid")
    return ti


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot raw profile vs PCA vs emulator-predicted reconstruction.")
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--model-dir", required=True, help="Model artifact dir with yhat_train.npy/yhat_val.npy")
    ap.add_argument("--split", choices=["val", "train"], default="val")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds = Path(args.dataset_dir).resolve()
    md = Path(args.model_dir).resolve()

    with open(ds / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    depth_grid = np.asarray(meta["profile"]["depth_grid_km"], dtype=float)
    source_paths = [Path(p) for p in meta["profile"]["source_paths"]]

    scores_raw = np.load(ds / "scores_raw.npy")
    score_scale = np.load(ds / "pca_score_scale.npy") if (ds / "pca_score_scale.npy").exists() else np.ones(scores_raw.shape[1], dtype=float)
    score_space = str(meta.get("target", {}).get("score_space", "raw")).strip().lower()
    mean_profile = np.load(ds / "pca_mean_profile.npy")
    components = np.load(ds / "pca_components.npy")
    train_idx = np.load(ds / "train_idx.npy")
    val_idx = np.load(ds / "val_idx.npy")

    true_raw = np.vstack([_load_profile_on_grid(p, depth_grid) for p in source_paths])
    recon_pca = mean_profile[None, :] + scores_raw @ components

    if args.split == "val":
        idx = val_idx
        pred_scores = np.load(md / "yhat_val.npy")
    else:
        idx = train_idx
        pred_scores = np.load(md / "yhat_train.npy")

    if pred_scores.ndim == 1:
        pred_scores = pred_scores.reshape(-1, 1)

    if pred_scores.shape[0] != idx.size:
        raise RuntimeError("Prediction rows do not match selected split indices.")

    if score_space == "whitened":
        pred_scores_raw = pred_scores * score_scale[None, :]
    else:
        pred_scores_raw = pred_scores

    recon_pred_split = mean_profile[None, :] + pred_scores_raw @ components
    true_raw_split = true_raw[idx]
    recon_pca_split = recon_pca[idx]

    err_pca = true_raw_split - recon_pca_split
    err_pred = true_raw_split - recon_pred_split

    rmse_pca = float(np.sqrt(np.mean(err_pca ** 2)))
    rmse_pred = float(np.sqrt(np.mean(err_pred ** 2)))
    rmse_depth_pca = np.sqrt(np.mean(err_pca ** 2, axis=0))
    rmse_depth_pred = np.sqrt(np.mean(err_pred ** 2, axis=0))

    rng = np.random.default_rng(args.seed)
    if args.n_samples > 0 and idx.size > args.n_samples:
        pick_local = np.sort(rng.choice(np.arange(idx.size), size=args.n_samples, replace=False))
    else:
        pick_local = np.arange(idx.size)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.2, 5.4), constrained_layout=True)

    for i in pick_local:
        ax0.plot(true_raw_split[i], depth_grid, color="0.65", lw=0.9, alpha=0.45)
        ax0.plot(recon_pca_split[i], depth_grid, color="tab:blue", lw=1.0, alpha=0.65)
        ax0.plot(recon_pred_split[i], depth_grid, color="tab:orange", lw=1.0, alpha=0.65)

    ax0.set_xlabel("Temperature (C)")
    ax0.set_ylabel("Depth (km)")
    ax0.invert_yaxis()
    ax0.grid(True, ls=":", alpha=0.35)
    ax0.set_title(f"{args.split}: raw(gray), PCA(blue), emulator(orange)")

    med_true = np.median(true_raw_split, axis=0)
    med_pca = np.median(recon_pca_split, axis=0)
    med_pred = np.median(recon_pred_split, axis=0)

    ax1.plot(med_true, depth_grid, color="0.15", lw=2.2, label="raw median")
    ax1.plot(med_pca, depth_grid, color="tab:blue", lw=2.0, label=f"PCA median (RMSE={rmse_pca:.2f} C)")
    ax1.plot(med_pred, depth_grid, color="tab:orange", lw=2.0, label=f"Emu median (RMSE={rmse_pred:.2f} C)")

    ax1_t = ax1.twiny()
    ax1_t.plot(rmse_depth_pca, depth_grid, color="tab:blue", ls="--", lw=1.6, alpha=0.85)
    ax1_t.plot(rmse_depth_pred, depth_grid, color="tab:orange", ls="-", lw=1.6, alpha=0.85)
    ax1_t.set_xlabel("RMSE by depth (C)")

    ax1.set_xlabel("Temperature (C)")
    ax1.set_ylabel("Depth (km)")
    ax1.invert_yaxis()
    ax1.grid(True, ls=":", alpha=0.35)
    ax1.set_title("Median profiles + depthwise RMSE")
    ax1.legend(loc="lower right", fontsize=8)

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")

    print(f"[OK] Saved: {out}")
    print(f"[OK] split={args.split} rmse_pca_only_C={rmse_pca:.6f}")
    print(f"[OK] split={args.split} rmse_emu_recon_C={rmse_pred:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

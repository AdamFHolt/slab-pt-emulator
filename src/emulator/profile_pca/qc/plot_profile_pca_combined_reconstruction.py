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
    ap = argparse.ArgumentParser(description="Plot raw vs PCA and raw vs emulator reconstructions with depthwise RMSE strips.")
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--split", choices=["val", "train"], default="val")
    ap.add_argument("--n-samples", type=int, default=25)
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

    pred_scores_raw = pred_scores * score_scale[None, :] if score_space == "whitened" else pred_scores
    recon_pred_split = mean_profile[None, :] + pred_scores_raw @ components
    true_raw_split = true_raw[idx]
    recon_pca_split = recon_pca[idx]

    err_pca = true_raw_split - recon_pca_split
    err_pred = true_raw_split - recon_pred_split
    rmse_depth_pca = np.sqrt(np.mean(err_pca ** 2, axis=0))
    rmse_depth_pred = np.sqrt(np.mean(err_pred ** 2, axis=0))
    rmse_pca = float(np.sqrt(np.mean(err_pca ** 2)))
    rmse_pred = float(np.sqrt(np.mean(err_pred ** 2)))

    rng = np.random.default_rng(args.seed)
    if args.n_samples > 0 and idx.size > args.n_samples:
        pick_local = np.sort(rng.choice(np.arange(idx.size), size=args.n_samples, replace=False))
    else:
        pick_local = np.arange(idx.size)

    true_s = true_raw_split[pick_local]
    pca_s = recon_pca_split[pick_local]
    pred_s = recon_pred_split[pick_local]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(13.2, 5.6),
        constrained_layout=True,
        sharey=True,
        gridspec_kw={"width_ratios": [1.55, 0.58, 1.55, 0.58]},
    )
    ax0, ax1, ax2, ax3 = axes

    for i in range(true_s.shape[0]):
        ax0.plot(true_s[i], depth_grid, color="0.7", lw=1.0, alpha=0.45)
        ax0.plot(pca_s[i], depth_grid, color="tab:blue", lw=1.1, alpha=0.75)

        ax2.plot(true_s[i], depth_grid, color="0.7", lw=1.0, alpha=0.45)
        ax2.plot(pred_s[i], depth_grid, color="tab:orange", lw=1.1, alpha=0.75)

    ax0.set_xlabel("Temperature ($^\\circ$C)")
    ax0.set_ylabel("Depth (km)")
    ax0.set_title("Raw vs PCA reconstruction")
    ax0.grid(True, ls=":", alpha=0.35)
    ax0.invert_yaxis()

    ax1.plot(rmse_depth_pca, depth_grid, color="tab:blue", lw=2.0)
    ax1.set_xlabel("RMSE ($^\\circ$C)")
    ax1.set_title("")
    ax1.grid(True, ls=":", alpha=0.35)
    ax1.invert_yaxis()
    ax1.text(
        0.50,
        0.975,
        f"mean = {rmse_pca:.2f} $^\\circ$C",
        transform=ax1.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )

    ax2.set_xlabel("Temperature ($^\\circ$C)")
    ax2.set_title("Raw vs emulator reconstruction")
    ax2.grid(True, ls=":", alpha=0.35)
    ax2.invert_yaxis()

    ax3.plot(rmse_depth_pred, depth_grid, color="tab:orange", lw=2.0)
    ax3.set_xlabel("RMSE ($^\\circ$C)")
    ax3.set_title("")
    ax3.grid(True, ls=":", alpha=0.35)
    ax3.invert_yaxis()
    ax3.text(
        0.50,
        0.975,
        f"mean = {rmse_pred:.2f} $^\\circ$C",
        transform=ax3.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )

    xmin = float(min(np.nanmin(true_s), np.nanmin(pca_s), np.nanmin(pred_s)))
    xmax = float(max(np.nanmax(true_s), np.nanmax(pca_s), np.nanmax(pred_s)))
    xpad = 0.03 * max(1.0, xmax - xmin)
    for ax in (ax0, ax2):
        ax.set_xlim(xmin - xpad, xmax + xpad)

    rmse_max = float(max(np.nanmax(rmse_depth_pca), np.nanmax(rmse_depth_pred)))
    for ax in (ax1, ax3):
        ax.set_xlim(0.0, rmse_max * 1.05)

    y0 = float(np.nanmin(depth_grid))
    y1 = max(85.0, float(np.nanmax(depth_grid)))
    for ax in (ax0, ax1, ax2, ax3):
        ax.set_ylim(y1, y0)

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"[OK] Saved: {out}")
    print(f"[OK] split={args.split} rmse_pca_only_C={rmse_pca:.6f}")
    print(f"[OK] split={args.split} rmse_emu_recon_C={rmse_pred:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
PLOTS_ROOT_DEFAULT = REPO_ROOT / "plots" / "science-emulator" / "profile-pca"


def _time_from_name(name: str) -> float:
    m = re.search(r"_t([0-9p]+)Myr_", name)
    if not m:
        raise ValueError(f"Could not parse time from dataset name: {name}")
    return float(m.group(1).replace("p", "."))


def _time_label(value: float) -> str:
    return f"{value:g} Myr"


def _time_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


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


def _predict_raw_scores(model, X_std: np.ndarray, y_mu: np.ndarray, y_sd: np.ndarray, score_space: str, score_scale: np.ndarray) -> np.ndarray:
    y_pred_std = np.asarray(model.predict(X_std))
    if y_pred_std.ndim == 1:
        y_pred_std = y_pred_std.reshape(-1, 1)
    y_pred = y_pred_std * y_sd[None, :] + y_mu[None, :]
    if score_space == "whitened":
        y_pred = y_pred * score_scale[None, :]
    return y_pred


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize profile-PCA time evolution for one suite.")
    ap.add_argument("--suite", required=True, choices=["const-vc", "ramped-vc"])
    ap.add_argument("--data-root", default=str(REPO_ROOT / "src" / "emulator" / "data" / "profile_pca"))
    ap.add_argument("--models-root", default=str(REPO_ROOT / "src" / "emulator" / "models" / "profile_pca"))
    ap.add_argument("--model-tag", default="gp_m25")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--times", default="0.5 1 2 3 4 5", help="Space-separated time list in Myr.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve() / args.suite / "runs"
    models_root = Path(args.models_root).resolve() / args.suite / "runs"
    times = [float(x) for x in args.times.replace(",", " ").split() if x.strip()]

    rows: list[dict[str, object]] = []
    for t in times:
        tlabel = _time_tag(t)
        dname = f"profileT_pca_t{tlabel}Myr_k{args.k}"
        ds = data_root / dname
        md = models_root / dname / args.model_tag
        if not ds.exists() or not md.exists():
            continue

        with open(ds / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        depth_grid = np.asarray(meta["profile"]["depth_grid_km"], dtype=float)
        source_paths = [Path(p) for p in meta["profile"]["source_paths"]]
        val_idx = np.load(ds / "val_idx.npy")
        true_raw = np.vstack([_load_profile_on_grid(p, depth_grid) for p in source_paths])
        true_val = true_raw[val_idx]

        mean_profile = np.load(ds / "pca_mean_profile.npy")
        components = np.load(ds / "pca_components.npy")
        score_scale = np.load(ds / "pca_score_scale.npy") if (ds / "pca_score_scale.npy").exists() else np.ones(components.shape[0], dtype=float)
        score_space = str(meta.get("target", {}).get("score_space", "raw")).strip().lower()
        y_mu = np.asarray(meta["scalers"]["Y"]["mean"], dtype=float)
        y_sd = np.asarray(meta["scalers"]["Y"]["std"], dtype=float)
        X_std = np.load(ds / "X_std.npy")[val_idx]
        model = joblib.load(md / "model.joblib")
        pred_scores_raw = _predict_raw_scores(model, X_std, y_mu, y_sd, score_space, score_scale)
        recon_pred = mean_profile[None, :] + pred_scores_raw @ components

        quality_path = md / "profile_pca_quality.json"
        if quality_path.exists():
            rep = json.load(open(quality_path, "r", encoding="utf-8"))
            rmse_emu = float(rep["metrics"]["val"]["profile_space"]["emulator_reconstruction"]["rmse"])
            rmse_pca = float(rep["metrics"]["val"]["profile_space"]["pca_truncation_baseline"]["rmse"])
        else:
            scores_raw = np.load(ds / "scores_raw.npy")[val_idx]
            recon_pca = mean_profile[None, :] + scores_raw @ components
            rmse_emu = float(np.sqrt(np.mean((true_val - recon_pred) ** 2)))
            rmse_pca = float(np.sqrt(np.mean((true_val - recon_pca) ** 2)))

        rows.append(
            {
                "time": t,
                "depth_grid": depth_grid,
                "true_median": np.median(true_val, axis=0),
                "emu_median": np.median(recon_pred, axis=0),
                "rmse_depth_emu": np.sqrt(np.mean((true_val - recon_pred) ** 2, axis=0)),
                "rmse_emu": rmse_emu,
                "rmse_pca": rmse_pca,
            }
        )

    if not rows:
        raise SystemExit("[ERR] No matching profile-PCA runs found.")

    rows.sort(key=lambda r: float(r["time"]))
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.15, 0.92, len(rows)))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.4, 5.5), constrained_layout=True)

    for color, row in zip(colors, rows):
        depth_grid = np.asarray(row["depth_grid"])
        t = float(row["time"])
        label = _time_label(t)
        ax0.plot(np.asarray(row["true_median"]), depth_grid, color=color, ls="--", lw=1.6, alpha=0.8)
        ax0.plot(np.asarray(row["emu_median"]), depth_grid, color=color, lw=2.1, label=label)

    ax0.set_xlabel("Temperature ($^\\circ$C)")
    ax0.set_ylabel("Depth (km)")
    ax0.invert_yaxis()
    ax0.grid(True, ls=":", alpha=0.35)
    ax0.set_title("Median profiles by time")
    ax0.legend(title="Time", loc="upper right", fontsize=8, title_fontsize=9)
    ax0.text(
        0.02, 0.02,
        "Dashed = raw median\nSolid = emulator median",
        transform=ax0.transAxes,
        va="bottom",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9),
    )

    for color, row in zip(colors, rows):
        depth_grid = np.asarray(row["depth_grid"])
        t = float(row["time"])
        label = _time_label(t)
        ax1.plot(np.asarray(row["rmse_depth_emu"]), depth_grid, color=color, lw=2.0, label=label)

    ax1.set_xlabel("Validation profile RMSE ($^\\circ$C)")
    ax1.set_ylabel("Depth (km)")
    ax1.invert_yaxis()
    ax1.set_title("Reconstruction RMSE by depth")
    ax1.grid(True, ls=":", alpha=0.35)

    out = Path(args.out).resolve() if args.out else (PLOTS_ROOT_DEFAULT / args.suite / f"{args.suite}_profile_pca_time_summary.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"[OK] Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

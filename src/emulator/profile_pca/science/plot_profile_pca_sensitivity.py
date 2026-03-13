#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
PLOTS_ROOT_DEFAULT = REPO_ROOT / "plots" / "science-emulator" / "profile-pca"


def _time_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def nice_label(param: str) -> str:
    labels = {
        "v_conv": r"$v_{\rm conv}$",
        "t_conv": r"$t_{\rm conv}$",
        "age_SP": r"$\mathrm{age}_{\rm SP}$",
        "age_OP": r"$\mathrm{age}_{\rm OP}$",
        "dip_int": r"$\theta_{\rm init}$",
        "eta_UM": r"$\eta_{\rm UM}$",
    }
    return labels.get(param, param)


def _predict_raw_scores(model, X_std: np.ndarray, y_mu: np.ndarray, y_sd: np.ndarray, score_space: str, score_scale: np.ndarray) -> np.ndarray:
    y_pred_std = np.asarray(model.predict(X_std))
    if y_pred_std.ndim == 1:
        y_pred_std = y_pred_std.reshape(-1, 1)
    y_pred = y_pred_std * y_sd[None, :] + y_mu[None, :]
    if score_space == "whitened":
        y_pred = y_pred * score_scale[None, :]
    return y_pred


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot depth-dependent profile-PCA sensitivity for one trained dataset.")
    ap.add_argument("--suite", required=True, choices=["const-vc", "ramped-vc"])
    ap.add_argument("--time", type=float, default=3.0, help="Target time in Myr.")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--model-tag", default="gp_m25")
    ap.add_argument("--data-root", default=str(REPO_ROOT / "src" / "emulator" / "data" / "profile_pca"))
    ap.add_argument("--models-root", default=str(REPO_ROOT / "src" / "emulator" / "models" / "profile_pca"))
    ap.add_argument("--n-baselines", type=int, default=64)
    ap.add_argument("--grid-size", type=int, default=21)
    ap.add_argument("--percentiles", default="5,95", help="Low/high percentiles for parameter sweep.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tlabel = _time_tag(args.time)
    dname = f"profileT_pca_t{tlabel}Myr_k{args.k}"
    ds = Path(args.data_root).resolve() / args.suite / "runs" / dname
    md = Path(args.models_root).resolve() / args.suite / "runs" / dname / args.model_tag
    if not ds.exists():
        raise SystemExit(f"[ERR] Missing dataset dir: {ds}")
    if not md.exists():
        raise SystemExit(f"[ERR] Missing model dir: {md}")

    with open(ds / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = list(meta["feature_cols"])
    depth_grid = np.asarray(meta["profile"]["depth_grid_km"], dtype=float)
    X_raw = np.load(ds / "X_raw.npy")
    X_std_full = np.load(ds / "X_std.npy")
    train_idx = np.load(ds / "train_idx.npy")
    mean_profile = np.load(ds / "pca_mean_profile.npy")
    components = np.load(ds / "pca_components.npy")
    score_scale = np.load(ds / "pca_score_scale.npy") if (ds / "pca_score_scale.npy").exists() else np.ones(components.shape[0], dtype=float)
    score_space = str(meta.get("target", {}).get("score_space", "raw")).strip().lower()
    x_mu = np.asarray(meta["scalers"]["X"]["mean"], dtype=float)
    x_sd = np.asarray(meta["scalers"]["X"]["std"], dtype=float)
    y_mu = np.asarray(meta["scalers"]["Y"]["mean"], dtype=float)
    y_sd = np.asarray(meta["scalers"]["Y"]["std"], dtype=float)

    rng = np.random.default_rng(args.seed)
    if train_idx.size > args.n_baselines:
        idx = np.sort(rng.choice(train_idx, size=args.n_baselines, replace=False))
    else:
        idx = train_idx
    baselines = X_raw[idx]

    p_lo, p_hi = [float(x) for x in args.percentiles.split(",")]
    model = joblib.load(md / "model.joblib")

    fig, ax = plt.subplots(figsize=(7.3, 6.0), constrained_layout=True)
    cmap = plt.cm.tab10

    for i, fname in enumerate(feature_cols):
        fidx = i
        lo = float(np.percentile(X_raw[:, fidx], p_lo))
        hi = float(np.percentile(X_raw[:, fidx], p_hi))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            continue
        grid = np.linspace(lo, hi, args.grid_size)

        mean_profiles = []
        for value in grid:
            X_probe = baselines.copy()
            X_probe[:, fidx] = value
            X_probe_std = (X_probe - x_mu[None, :]) / x_sd[None, :]
            pred_scores_raw = _predict_raw_scores(model, X_probe_std, y_mu, y_sd, score_space, score_scale)
            recon = mean_profile[None, :] + pred_scores_raw @ components
            mean_profiles.append(np.mean(recon, axis=0))

        mean_profiles_arr = np.asarray(mean_profiles)
        delta_profile = mean_profiles_arr[-1] - mean_profiles_arr[0]
        ax.plot(delta_profile, depth_grid, lw=2.0, color=cmap(i % 10), label=nice_label(fname))

    ax.axvline(0.0, color="0.35", ls="--", lw=1.0)
    ax.set_xlabel("Temperature response, high - low ($^\\circ$C)")
    ax.set_ylabel("Depth (km)")
    ax.invert_yaxis()
    ax.grid(True, ls=":", alpha=0.35)
    ax.legend(loc="lower right", fontsize=9)
    ax.text(
        0.02, 0.02,
        f"Parameter sweep uses {int(p_lo)}-{int(p_hi)}th percentile range\nAveraged over {baselines.shape[0]} training baselines",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.9),
    )

    out = Path(args.out).resolve() if args.out else (PLOTS_ROOT_DEFAULT / args.suite / f"{args.suite}_{dname}_{args.model_tag}_sensitivity.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"[OK] Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def _predict_raw_scores(
    model,
    X_std: np.ndarray,
    y_mu: np.ndarray,
    y_sd: np.ndarray,
    score_space: str,
    score_scale: np.ndarray,
) -> np.ndarray:
    y_pred_std = np.asarray(model.predict(X_std))
    if y_pred_std.ndim == 1:
        y_pred_std = y_pred_std.reshape(-1, 1)
    y_pred = y_pred_std * y_sd[None, :] + y_mu[None, :]
    if score_space == "whitened":
        y_pred = y_pred * score_scale[None, :]
    return y_pred


def _load_profiles(
    *,
    suite: str,
    time_myr: float,
    k: int,
    model_tag: str,
    feature: str,
    profile_percentiles: list[float],
    n_baselines: int,
    seed: int,
    data_root: Path,
    models_root: Path,
) -> tuple[np.ndarray, dict[str, tuple[float, np.ndarray]], int]:
    tlabel = _time_tag(time_myr)
    dname = f"profileT_pca_t{tlabel}Myr_k{k}"
    ds = data_root / suite / "runs" / dname
    md = models_root / suite / "runs" / dname / model_tag
    if not ds.exists():
        raise FileNotFoundError(f"Missing dataset dir: {ds}")
    if not md.exists():
        raise FileNotFoundError(f"Missing model dir: {md}")

    with open(ds / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = list(meta["feature_cols"])
    if feature not in feature_cols:
        raise KeyError(f"Feature {feature!r} not present in {dname}; available: {feature_cols}")
    fidx = feature_cols.index(feature)

    depth_grid = np.asarray(meta["profile"]["depth_grid_km"], dtype=float)
    X_raw = np.load(ds / "X_raw.npy")
    train_idx = np.load(ds / "train_idx.npy")
    mean_profile = np.load(ds / "pca_mean_profile.npy")
    components = np.load(ds / "pca_components.npy")
    score_scale = np.load(ds / "pca_score_scale.npy") if (ds / "pca_score_scale.npy").exists() else np.ones(components.shape[0], dtype=float)
    score_space = str(meta.get("target", {}).get("score_space", "raw")).strip().lower()
    x_mu = np.asarray(meta["scalers"]["X"]["mean"], dtype=float)
    x_sd = np.asarray(meta["scalers"]["X"]["std"], dtype=float)
    y_mu = np.asarray(meta["scalers"]["Y"]["mean"], dtype=float)
    y_sd = np.asarray(meta["scalers"]["Y"]["std"], dtype=float)

    rng = np.random.default_rng(seed)
    if train_idx.size > n_baselines:
        idx = np.sort(rng.choice(train_idx, size=n_baselines, replace=False))
    else:
        idx = train_idx
    baselines = X_raw[idx]

    model = joblib.load(md / "model.joblib")

    results: dict[str, tuple[float, np.ndarray]] = {}
    labels = ["low", "median", "high"]
    values = [float(np.percentile(X_raw[:, fidx], p)) for p in profile_percentiles]
    for label, value in zip(labels, values):
        X_probe = baselines.copy()
        X_probe[:, fidx] = value
        X_probe_std = (X_probe - x_mu[None, :]) / x_sd[None, :]
        pred_scores_raw = _predict_raw_scores(model, X_probe_std, y_mu, y_sd, score_space, score_scale)
        recon = mean_profile[None, :] + pred_scores_raw @ components
        results[label] = (value, np.mean(recon, axis=0))

    return depth_grid, results, baselines.shape[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot low/median/high profile families for one key parameter across profile-PCA times.")
    ap.add_argument("--suite", required=True, choices=["const-vc", "ramped-vc"])
    ap.add_argument("--times", default="0.5 3 5", help="Space- or comma-separated times in Myr.")
    ap.add_argument("--feature", default="v_conv")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--model-tag", default="gp_m25")
    ap.add_argument("--profile-percentiles", default="5,50,95", help="Low, median, high percentiles for the chosen parameter.")
    ap.add_argument("--n-baselines", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default=str(REPO_ROOT / "src" / "emulator" / "data" / "profile_pca"))
    ap.add_argument("--models-root", default=str(REPO_ROOT / "src" / "emulator" / "models" / "profile_pca"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    times = [float(x) for x in args.times.replace(",", " ").split() if x.strip()]
    pvals = [float(x) for x in args.profile_percentiles.split(",")]
    if len(pvals) != 3:
        raise SystemExit("[ERR] --profile-percentiles must have exactly three values, e.g. 5,50,95")

    data_root = Path(args.data_root).resolve()
    models_root = Path(args.models_root).resolve()

    fig, axes = plt.subplots(1, len(times), figsize=(3.8 * len(times), 6.0), constrained_layout=True, sharey=True)
    axes = np.atleast_1d(axes)

    line_specs = {
        "low": ("#2c7fb8", 2.1, "--"),
        "median": ("#111111", 2.5, "-"),
        "high": ("#d95f0e", 2.1, "--"),
    }

    baselines_n = None
    xlim = (-50.0, 1150.0)
    for ax, time_myr in zip(axes, times):
        depth_grid, profiles, n_base = _load_profiles(
            suite=args.suite,
            time_myr=time_myr,
            k=args.k,
            model_tag=args.model_tag,
            feature=args.feature,
            profile_percentiles=pvals,
            n_baselines=args.n_baselines,
            seed=args.seed,
            data_root=data_root,
            models_root=models_root,
        )
        baselines_n = baselines_n or n_base

        for label in ["low", "median", "high"]:
            value, profile = profiles[label]
            color, lw, ls = line_specs[label]
            ax.plot(profile, depth_grid, color=color, lw=lw, ls=ls, label=f"{label.capitalize()} ({value:g})")

        ax.set_xlabel("Temperature ($^\\circ$C)")
        ax.set_xlim(*xlim)
        ax.grid(True, ls=":", alpha=0.35)
        ax.invert_yaxis()
        ax.text(50.0, 0.0, f"{time_myr:g} Myr", ha="left", va="center", fontsize=13)

    axes[0].set_ylabel("Depth (km)")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0.0, 0.07), fontsize=9, title=nice_label(args.feature), title_fontsize=10)
    axes[0].text(
        0.02,
        0.02,
        f"{int(pvals[0])}-{int(pvals[1])}-{int(pvals[2])}% values, mean over {baselines_n}",
        transform=axes[0].transAxes,
        va="bottom",
        ha="left",
        fontsize=8,
    )

    times_tag = "_".join(_time_tag(t) for t in times)
    out = Path(args.out).resolve() if args.out else (
        PLOTS_ROOT_DEFAULT / args.suite / f"{args.suite}_profile_pca_{times_tag}Myr_{args.feature}_{args.model_tag}_profile_family.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"[OK] Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

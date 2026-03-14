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


def _load_time_panel(
    *,
    suite: str,
    time_myr: float,
    k: int,
    model_tag: str,
    data_root: Path,
    models_root: Path,
    n_baselines: int,
    grid_size: int,
    p_lo: float,
    p_hi: float,
    seed: int,
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray], int]:
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

    deltas: dict[str, np.ndarray] = {}
    for fidx, fname in enumerate(feature_cols):
        lo = float(np.percentile(X_raw[:, fidx], p_lo))
        hi = float(np.percentile(X_raw[:, fidx], p_hi))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            continue
        grid = np.linspace(lo, hi, grid_size)

        mean_profiles = []
        for value in grid:
            X_probe = baselines.copy()
            X_probe[:, fidx] = value
            X_probe_std = (X_probe - x_mu[None, :]) / x_sd[None, :]
            pred_scores_raw = _predict_raw_scores(model, X_probe_std, y_mu, y_sd, score_space, score_scale)
            recon = mean_profile[None, :] + pred_scores_raw @ components
            mean_profiles.append(np.mean(recon, axis=0))

        mean_profiles_arr = np.asarray(mean_profiles)
        deltas[fname] = mean_profiles_arr[-1] - mean_profiles_arr[0]

    return feature_cols, depth_grid, deltas, baselines.shape[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot depth-dependent profile-PCA sensitivity for one trained dataset.")
    ap.add_argument("--suite", required=True, choices=["const-vc", "ramped-vc"])
    ap.add_argument("--time", type=float, default=3.0, help="Target time in Myr.")
    ap.add_argument("--times", default=None, help="Optional space- or comma-separated times in Myr for multi-panel output.")
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

    times = [float(x) for x in (args.times.replace(",", " ").split() if args.times else [args.time]) if str(x).strip()]
    p_lo, p_hi = [float(x) for x in args.percentiles.split(",")]
    data_root = Path(args.data_root).resolve()
    models_root = Path(args.models_root).resolve()

    ncols = len(times)
    panel_width = 3.15 if ncols >= 6 else 3.8
    fig, axes = plt.subplots(1, ncols, figsize=(panel_width * ncols, 6.0), constrained_layout=True, sharey=True)
    axes = np.atleast_1d(axes)
    cmap = plt.cm.tab10
    feature_cols_ref: list[str] | None = None
    baselines_n = None
    panel_data: list[tuple[np.ndarray, dict[str, np.ndarray], float]] = []
    xmins: list[float] = []
    xmaxs: list[float] = []

    for time_myr in times:
        feature_cols, depth_grid, deltas, n_base = _load_time_panel(
            suite=args.suite,
            time_myr=time_myr,
            k=args.k,
            model_tag=args.model_tag,
            data_root=data_root,
            models_root=models_root,
            n_baselines=args.n_baselines,
            grid_size=args.grid_size,
            p_lo=p_lo,
            p_hi=p_hi,
            seed=args.seed,
        )
        feature_cols_ref = feature_cols_ref or feature_cols
        baselines_n = baselines_n or n_base
        panel_data.append((depth_grid, deltas, time_myr))
        if deltas:
            all_vals = np.concatenate([np.asarray(v) for v in deltas.values()])
            xmins.append(float(np.nanmin(all_vals)))
            xmaxs.append(float(np.nanmax(all_vals)))

    if xmins and xmaxs:
        xmin = min(xmins)
        xmax = max(xmaxs)
        pad = 0.05 * max(1e-9, xmax - xmin)
        xlim = (xmin - pad, xmax + pad)
    else:
        xlim = None

    for ax, (depth_grid, deltas, time_myr) in zip(axes, panel_data):
        for i, fname in enumerate(feature_cols_ref):
            if fname not in deltas:
                continue
            ax.plot(deltas[fname], depth_grid, lw=2.0, color=cmap(i % 10), label=nice_label(fname))

        ax.axvline(0.0, color="0.35", ls="--", lw=1.0)
        ax.set_xlabel("Temperature response, high - low ($^\\circ$C)")
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.grid(True, ls=":", alpha=0.35)
        ax.invert_yaxis()
        ax.text(
            0.03,
            0.97,
            f"{time_myr:g} Myr",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
        )

    axes[0].set_ylabel("Depth (km)")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].text(
        0.02, 0.02,
        f"{int(p_lo)}-{int(p_hi)}% sweep, mean over {baselines_n} baselines",
        transform=axes[0].transAxes,
        va="bottom",
        ha="left",
        fontsize=8,
    )

    if len(times) == 1:
        tlabel = _time_tag(times[0])
        out = Path(args.out).resolve() if args.out else (PLOTS_ROOT_DEFAULT / args.suite / f"{args.suite}_profileT_pca_t{tlabel}Myr_k{args.k}_{args.model_tag}_sensitivity.png")
    else:
        times_tag = "_".join(_time_tag(t) for t in times)
        out = Path(args.out).resolve() if args.out else (PLOTS_ROOT_DEFAULT / args.suite / f"{args.suite}_profile_pca_{times_tag}Myr_{args.model_tag}_sensitivity.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"[OK] Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

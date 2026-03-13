#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[4]


def _inverse_standardize(arr_std: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return arr_std * std + mean


def _load_dataset_bundle(data_dir: Path) -> dict[str, object]:
    meta = json.loads((data_dir / "metadata.json").read_text())
    return {
        "meta": meta,
        "x_raw": np.load(data_dir / "X_raw.npy"),
        "train_idx": np.load(data_dir / "train_idx.npy"),
        "x_mean": np.asarray(meta["scalers"]["X"]["mean"], dtype=float),
        "x_std": np.asarray(meta["scalers"]["X"]["std"], dtype=float),
        "y_mean": np.asarray(meta["scalers"]["Y"]["mean"], dtype=float),
        "y_std": np.asarray(meta["scalers"]["Y"]["std"], dtype=float),
    }


def _predict_raw(model: object, x_raw: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray,
                 y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    x_std_space = (x_raw - x_mean) / x_std
    yhat_std = np.asarray(model.predict(x_std_space))
    if yhat_std.ndim == 1:
        yhat_std = yhat_std.reshape(-1, 1)
    return _inverse_standardize(yhat_std, y_mean, y_std)


def _feature_label_map() -> dict[str, str]:
    return {
        "v_conv": r"$v_{\mathrm{conv}}$ (cm/yr)",
        "age_SP": r"$\mathrm{age}_{\mathrm{SP}}$ (Myr)",
        "age_OP": r"$\mathrm{age}_{\mathrm{OP}}$ (Myr)",
        "dip_int": r"$\theta_{\mathrm{slab}}$ ($^\circ$)",
        "eta_UM": r"$\eta_{\mathrm{UM}}$ ($\log_{10}$ Pa s)",
        "t_conv": r"$t_{\mathrm{conv}}$ (Myr)",
    }


def _target_label(target_col: str) -> str:
    if target_col == "dTdt_C_per_Myr":
        return r"Cooling rate ($^\circ$C / Myr)"
    return target_col


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot a two-parameter single-depth emulator response surface.")
    p.add_argument("--suite", required=True)
    p.add_argument("--data-name", required=True)
    p.add_argument("--model-tag", default="gp_m25")
    p.add_argument("--x-feature", default="v_conv")
    p.add_argument("--y-feature", default="age_SP")
    p.add_argument("--grid-size", type=int, default=81)
    p.add_argument("--range-quantiles", type=float, nargs=2, default=(0.05, 0.95), metavar=("QLOW", "QHIGH"))
    p.add_argument(
        "--data-root",
        default=str(REPO_ROOT / "src" / "emulator" / "data" / "single_depth"),
    )
    p.add_argument(
        "--models-root",
        default=str(REPO_ROOT / "src" / "emulator" / "models" / "single_depth"),
    )
    p.add_argument(
        "--outdir",
        default=None,
        help="Defaults to plots/science-emulator/<suite>.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    q_low, q_high = args.range_quantiles
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError("--range-quantiles must satisfy 0 <= low < high <= 1")

    data_dir = Path(args.data_root).resolve() / args.suite / "runs" / args.data_name
    model_dir = Path(args.models_root).resolve() / args.suite / "runs" / args.data_name / args.model_tag
    outdir = Path(args.outdir).resolve() if args.outdir else REPO_ROOT / "plots" / "science-emulator" / args.suite
    outdir.mkdir(parents=True, exist_ok=True)

    bundle = _load_dataset_bundle(data_dir)
    model = joblib.load(model_dir / "model.joblib")
    report = json.loads((model_dir / "report.json").read_text())

    feature_cols = list(bundle["meta"]["feature_cols"])
    if args.x_feature not in feature_cols or args.y_feature not in feature_cols:
        raise ValueError(f"Requested features must be in dataset feature_cols={feature_cols}")

    x_train = np.asarray(bundle["x_raw"], dtype=float)[np.asarray(bundle["train_idx"], dtype=int)]
    baseline = np.median(x_train, axis=0)
    x_idx = feature_cols.index(args.x_feature)
    y_idx = feature_cols.index(args.y_feature)

    x_vals = np.linspace(*np.quantile(x_train[:, x_idx], [q_low, q_high]), args.grid_size)
    y_vals = np.linspace(*np.quantile(x_train[:, y_idx], [q_low, q_high]), args.grid_size)

    xx, yy = np.meshgrid(x_vals, y_vals)
    x_eval = np.repeat(baseline.reshape(1, -1), xx.size, axis=0)
    x_eval[:, x_idx] = xx.ravel()
    x_eval[:, y_idx] = yy.ravel()
    z = _predict_raw(
        model,
        x_eval,
        np.asarray(bundle["x_mean"], dtype=float),
        np.asarray(bundle["x_std"], dtype=float),
        np.asarray(bundle["y_mean"], dtype=float),
        np.asarray(bundle["y_std"], dtype=float),
    )[:, 0].reshape(xx.shape)

    labels = _feature_label_map()
    target_label = _target_label(report["target_cols"][0])

    fig, ax = plt.subplots(figsize=(7.4, 5.8), constrained_layout=True)
    mesh = ax.contourf(xx, yy, z, levels=18, cmap="viridis")
    contour = ax.contour(xx, yy, z, levels=8, colors="white", linewidths=0.65, alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8, fmt="%.0f")
    ax.scatter([baseline[x_idx]], [baseline[y_idx]], marker="x", s=70, c="red", linewidths=2)
    ax.set_xlabel(labels.get(args.x_feature, args.x_feature), fontsize=11)
    ax.set_ylabel(labels.get(args.y_feature, args.y_feature), fontsize=11)
    ax.tick_params(labelsize=10)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(target_label, fontsize=11)
    ax.text(
        0.985,
        0.97,
        f"Emulator val.\n$R^2$ = {report['metrics']['val']['_macro_avg']['r2']:.3f}\nRMSE = {report['metrics']['val']['_macro_avg']['rmse']:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
    )

    out_path = outdir / f"{args.suite}_{args.data_name}_{args.model_tag}_surface_{args.x_feature}_vs_{args.y_feature}.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()

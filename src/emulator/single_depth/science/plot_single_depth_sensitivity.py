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
    x_raw = np.load(data_dir / "X_raw.npy")
    train_idx = np.load(data_dir / "train_idx.npy")
    val_idx = np.load(data_dir / "val_idx.npy")

    x_mean = np.asarray(meta["scalers"]["X"]["mean"], dtype=float)
    x_std = np.asarray(meta["scalers"]["X"]["std"], dtype=float)
    y_mean = np.asarray(meta["scalers"]["Y"]["mean"], dtype=float)
    y_std = np.asarray(meta["scalers"]["Y"]["std"], dtype=float)

    return {
        "meta": meta,
        "x_raw": x_raw,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }


def _predict_raw(model: object, x_raw: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray,
                 y_mean: np.ndarray, y_std: np.ndarray) -> np.ndarray:
    x_std_space = (x_raw - x_mean) / x_std
    yhat_std = np.asarray(model.predict(x_std_space))
    if yhat_std.ndim == 1:
        yhat_std = yhat_std.reshape(-1, 1)
    return _inverse_standardize(yhat_std, y_mean, y_std)


def _default_label_map() -> dict[str, str]:
    return {
        "v_conv": r"$v_{\mathrm{conv}}$ (cm/yr)",
        "age_SP": r"$\mathrm{age}_{\mathrm{SP}}$ (Myr)",
        "age_OP": r"$\mathrm{age}_{\mathrm{OP}}$ (Myr)",
        "dip_int": r"$\theta_{\mathrm{slab}}$ ($^\circ$)",
        "eta_UM": r"$\eta_{\mathrm{UM}}$ ($\log_{10}$ Pa s)",
        "t_conv": r"$t_{\mathrm{conv}}$ (Myr)",
    }


def _default_feature_order() -> list[str]:
    return ["v_conv", "age_SP", "age_OP", "dip_int", "eta_UM", "t_conv"]


def _target_label(target_col: str) -> str:
    if target_col == "dTdt_C_per_Myr":
        return r"Cooling rate ($^\circ$C / Myr)"
    return target_col


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot one-at-a-time single-depth emulator sensitivity curves and a ranked "
            "effect-size summary."
        )
    )
    p.add_argument("--suite", required=True, help="Suite name, e.g. const-vc or ramped-vc.")
    p.add_argument("--data-name", required=True, help="Dataset name, e.g. 40km_dTdt.")
    p.add_argument("--model-tag", default="gp_m25", help="Model subdirectory tag.")
    p.add_argument(
        "--data-root",
        default=str(REPO_ROOT / "src" / "emulator" / "data" / "single_depth"),
        help="Root containing single-depth dataset bundles grouped by suite.",
    )
    p.add_argument(
        "--models-root",
        default=str(REPO_ROOT / "src" / "emulator" / "models" / "single_depth"),
        help="Root containing single-depth trained models grouped by suite.",
    )
    p.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Defaults to plots/science-emulator/<suite>.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of highest-effect parameters to show as response curves.",
    )
    p.add_argument(
        "--grid-size",
        type=int,
        default=121,
        help="Number of evaluation points for each one-at-a-time curve.",
    )
    p.add_argument(
        "--sample-size",
        type=int,
        default=64,
        help="Number of baseline samples used to average partial-dependence curves.",
    )
    p.add_argument(
        "--range-quantiles",
        type=float,
        nargs=2,
        default=(0.05, 0.95),
        metavar=("QLOW", "QHIGH"),
        help="Quantile range used for one-at-a-time sweeps and effect-size ranking.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    q_low, q_high = args.range_quantiles
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError("--range-quantiles must satisfy 0 <= low < high <= 1")
    if args.grid_size < 3:
        raise ValueError("--grid-size must be at least 3")

    data_dir = Path(args.data_root).resolve() / args.suite / "runs" / args.data_name
    model_dir = Path(args.models_root).resolve() / args.suite / "runs" / args.data_name / args.model_tag
    outdir = Path(args.outdir).resolve() if args.outdir else (
        REPO_ROOT / "plots" / "science-emulator" / args.suite
    )
    outdir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    bundle = _load_dataset_bundle(data_dir)
    report = json.loads((model_dir / "report.json").read_text())
    model = joblib.load(model_dir / "model.joblib")

    x_raw = bundle["x_raw"]
    train_idx = bundle["train_idx"]
    x_train = x_raw[train_idx]
    x_mean = bundle["x_mean"]
    x_std = bundle["x_std"]
    y_mean = bundle["y_mean"]
    y_std = bundle["y_std"]

    feature_cols = list(bundle["meta"]["feature_cols"])
    target_col = report["target_cols"][0]
    target_label = _target_label(target_col)
    label_map = _default_label_map()

    n_baselines = min(args.sample_size, x_train.shape[0])
    sample_idx = np.linspace(0, x_train.shape[0] - 1, n_baselines, dtype=int)
    x_baselines = x_train[sample_idx]

    sweep_specs: list[dict[str, object]] = []
    for j, feat in enumerate(feature_cols):
        qmin, qmax = np.quantile(x_train[:, j], [q_low, q_high])
        x_axis = np.linspace(float(qmin), float(qmax), args.grid_size)

        curves = []
        for baseline in x_baselines:
            x_eval = np.repeat(baseline.reshape(1, -1), args.grid_size, axis=0)
            x_eval[:, j] = x_axis
            yhat = _predict_raw(model, x_eval, x_mean, x_std, y_mean, y_std)[:, 0]
            curves.append(yhat)
        curves_arr = np.vstack(curves)
        mean_curve = np.mean(curves_arr, axis=0)
        q10_curve = np.quantile(curves_arr, 0.10, axis=0)
        q90_curve = np.quantile(curves_arr, 0.90, axis=0)

        sweep_specs.append(
            {
                "feature": feat,
                "label": label_map.get(feat, feat),
                "x_axis": x_axis,
                "mean_curve": mean_curve,
                "q10_curve": q10_curve,
                "q90_curve": q90_curve,
                "effect_size": float(np.max(mean_curve) - np.min(mean_curve)),
            }
        )

    feature_rank = {name: idx for idx, name in enumerate(_default_feature_order())}
    ranked = sorted(
        sweep_specs,
        key=lambda item: (-item["effect_size"], feature_rank.get(str(item["feature"]), 999)),
    )
    top_specs = ranked[: max(1, min(args.top_k, len(ranked)))]

    n_cols = 2
    n_curve_rows = int(np.ceil(len(top_specs) / n_cols))
    fig = plt.figure(figsize=(12, 3.8 + 2.6 * n_curve_rows), constrained_layout=True)
    gs = fig.add_gridspec(n_curve_rows + 1, n_cols, height_ratios=[1.15] + [1.0] * n_curve_rows)

    ax_bar = fig.add_subplot(gs[0, :])
    bar_labels = [item["label"] for item in ranked]
    bar_vals = [item["effect_size"] for item in ranked]
    ax_bar.set_axisbelow(True)
    ax_bar.bar(bar_labels, bar_vals, color="#4477AA", zorder=3)
    ax_bar.set_ylabel(f"Predicted {target_label} range", fontsize=11)
    ax_bar.set_title(
        f"{args.suite} {args.data_name}",
        fontsize=14,
    )
    ax_bar.tick_params(axis="x", rotation=20, labelsize=10)
    ax_bar.tick_params(axis="y", labelsize=10)
    ax_bar.grid(axis="y", alpha=0.25)
    val_text = (
        "Emulator val.\n"
        f"$R^2$ = {report['metrics']['val']['_macro_avg']['r2']:.3f}\n"
        f"RMSE = {report['metrics']['val']['_macro_avg']['rmse']:.3f}"
    )
    ax_bar.text(
        0.985,
        0.97,
        val_text,
        transform=ax_bar.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
    )

    y_lo = min(float(np.min(np.asarray(spec["q10_curve"], dtype=float))) for spec in top_specs)
    y_hi = max(float(np.max(np.asarray(spec["q90_curve"], dtype=float))) for spec in top_specs)
    y_pad = 0.05 * max(y_hi - y_lo, 1.0)

    for idx, spec in enumerate(top_specs):
        row = 1 + idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        x_axis = spec["x_axis"]
        ax.fill_between(x_axis, spec["q10_curve"], spec["q90_curve"], color="#99CCEE", alpha=0.45)
        ax.plot(x_axis, spec["mean_curve"], color="#117733", lw=2.0)

        ax.set_xlabel(spec["label"], fontsize=11)
        ax.set_ylabel(target_label, fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
        ax.grid(alpha=0.25)
        ax.text(
            0.98,
            0.96,
            rf"$\Delta$={spec['effect_size']:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.9},
        )

    # Hide any unused subplot slots in the last row.
    total_slots = n_curve_rows * n_cols
    for idx in range(len(top_specs), total_slots):
        row = 1 + idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.axis("off")

    out_path = outdir / f"{args.suite}_{args.data_name}_{args.model_tag}_sensitivity.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()

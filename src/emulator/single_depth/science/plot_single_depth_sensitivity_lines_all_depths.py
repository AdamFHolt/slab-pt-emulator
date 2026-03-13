#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[4]


def _inverse_standardize(arr_std: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return arr_std * std + mean


def _feature_order() -> list[str]:
    return ["v_conv", "age_SP", "age_OP", "dip_int", "eta_UM", "t_conv"]


def _feature_labels() -> dict[str, str]:
    return {
        "v_conv": r"$v_{\mathrm{conv}}$",
        "age_SP": r"$\mathrm{age}_{\mathrm{SP}}$",
        "age_OP": r"$\mathrm{age}_{\mathrm{OP}}$",
        "dip_int": r"$\theta_{\mathrm{slab}}$",
        "eta_UM": r"$\eta_{\mathrm{UM}}$",
        "t_conv": r"$t_{\mathrm{conv}}$",
    }


def _target_label(target_col: str) -> str:
    if target_col == "dTdt_C_per_Myr":
        return r"$\Delta$ cooling rate ($^\circ$C / Myr)"
    return target_col


def _load_bundle(data_dir: Path) -> dict[str, object]:
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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot signed one-at-a-time sensitivity as lines over depth.")
    p.add_argument("--suite", required=True)
    p.add_argument("--variant", default="dTdt")
    p.add_argument("--model-tag", default="gp_m25")
    p.add_argument("--grid-size", type=int, default=121)
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

    data_root = Path(args.data_root).resolve() / args.suite / "runs"
    models_root = Path(args.models_root).resolve() / args.suite / "runs"
    outdir = Path(args.outdir).resolve() if args.outdir else REPO_ROOT / "plots" / "science-emulator" / args.suite
    outdir.mkdir(parents=True, exist_ok=True)

    ordered_features = _feature_order()
    labels = _feature_labels()
    present_features: list[str] | None = None
    target_col: str | None = None
    rows: list[tuple[int, dict[str, float]]] = []

    for dataset_dir in sorted(data_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        m = re.match(r"(?P<depth>\d+)km_(?P<variant>.+)$", dataset_dir.name)
        if not m or m.group("variant") != args.variant:
            continue
        depth_km = int(m.group("depth"))
        model_dir = models_root / dataset_dir.name / args.model_tag
        if not (model_dir / "model.joblib").exists() or not (model_dir / "report.json").exists():
            continue

        bundle = _load_bundle(dataset_dir)
        model = joblib.load(model_dir / "model.joblib")
        report = json.loads((model_dir / "report.json").read_text())
        target_col = report["target_cols"][0]

        x_train = np.asarray(bundle["x_raw"], dtype=float)[np.asarray(bundle["train_idx"], dtype=int)]
        feature_cols = list(bundle["meta"]["feature_cols"])
        if present_features is None:
            present_features = [feat for feat in ordered_features if feat in feature_cols]

        baseline = np.median(x_train, axis=0)
        by_feature: dict[str, float] = {}
        for j, feat in enumerate(feature_cols):
            qmin, qmax = np.quantile(x_train[:, j], [q_low, q_high])
            x_axis = np.linspace(float(qmin), float(qmax), args.grid_size)
            x_eval = np.repeat(baseline.reshape(1, -1), args.grid_size, axis=0)
            x_eval[:, j] = x_axis
            yhat = _predict_raw(
                model,
                x_eval,
                np.asarray(bundle["x_mean"], dtype=float),
                np.asarray(bundle["x_std"], dtype=float),
                np.asarray(bundle["y_mean"], dtype=float),
                np.asarray(bundle["y_std"], dtype=float),
            )[:, 0]
            by_feature[feat] = float(yhat[-1] - yhat[0])
        rows.append((depth_km, by_feature))

    if not rows or target_col is None:
        raise RuntimeError(f"No matching trained datasets found for suite={args.suite} variant={args.variant}")

    rows.sort(key=lambda item: item[0])
    depths = np.asarray([depth for depth, _ in rows], dtype=float)
    plot_features = present_features or ordered_features

    fig, ax = plt.subplots(figsize=(7.8, 6.2), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_features)))
    for color, feat in zip(colors, plot_features):
        vals = np.asarray([mapping.get(feat, np.nan) for _, mapping in rows], dtype=float)
        ax.plot(vals, depths, marker="o", lw=2.0, ms=4.5, label=labels.get(feat, feat), color=color)

    ax.axvline(0.0, color="#444444", lw=1.0, ls="--", alpha=0.8)
    ax.invert_yaxis()
    ax.set_xlabel(_target_label(target_col), fontsize=11)
    ax.set_ylabel("Depth (km)", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=10)

    out_path = outdir / f"{args.suite}_{args.variant}_{args.model_tag}_sensitivity_lines_all_depths.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()

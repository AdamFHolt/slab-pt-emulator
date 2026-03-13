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


def _default_feature_order() -> list[str]:
    return ["v_conv", "age_SP", "age_OP", "dip_int", "eta_UM", "t_conv"]


def _default_label_map() -> dict[str, str]:
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
    p = argparse.ArgumentParser(
        description=(
            "Summarize one-at-a-time single-depth emulator sensitivity over depth as a heatmap."
        )
    )
    p.add_argument("--suite", required=True, help="Suite name, e.g. const-vc or ramped-vc.")
    p.add_argument("--variant", default="dTdt", help="Single-depth dataset suffix to include.")
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
        "--grid-size",
        type=int,
        default=121,
        help="Number of evaluation points for each one-at-a-time curve.",
    )
    p.add_argument(
        "--range-quantiles",
        type=float,
        nargs=2,
        default=(0.05, 0.95),
        metavar=("QLOW", "QHIGH"),
        help="Quantile range used for one-at-a-time sweeps and effect-size ranking.",
    )
    p.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate heatmap cells with effect-size values.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    q_low, q_high = args.range_quantiles
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError("--range-quantiles must satisfy 0 <= low < high <= 1")

    data_root = Path(args.data_root).resolve() / args.suite / "runs"
    models_root = Path(args.models_root).resolve() / args.suite / "runs"
    outdir = Path(args.outdir).resolve() if args.outdir else (
        REPO_ROOT / "plots" / "science-emulator" / args.suite
    )
    outdir.mkdir(parents=True, exist_ok=True)

    label_map = _default_label_map()
    feature_order = _default_feature_order()
    present_features: list[str] | None = None

    depth_rows: list[tuple[int, str, np.ndarray, str]] = []
    for dataset_dir in sorted(data_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        m = re.match(r"(?P<depth>\d+)km_(?P<variant>.+)$", dataset_dir.name)
        if not m or m.group("variant") != args.variant:
            continue

        depth_km = int(m.group("depth"))
        model_dir = models_root / dataset_dir.name / args.model_tag
        if not model_dir.exists():
            continue
        model_path = model_dir / "model.joblib"
        report_path = model_dir / "report.json"
        if not model_path.exists() or not report_path.exists():
            continue

        bundle = _load_bundle(dataset_dir)
        model = joblib.load(model_path)
        report = json.loads(report_path.read_text())

        x_raw = np.asarray(bundle["x_raw"], dtype=float)
        x_train = x_raw[np.asarray(bundle["train_idx"], dtype=int)]
        baseline = np.median(x_train, axis=0)
        feature_cols = list(bundle["meta"]["feature_cols"])

        if present_features is None:
            present_features = [feat for feat in feature_order if feat in feature_cols]

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

        effect_vec = np.asarray([by_feature.get(feat, np.nan) for feat in (present_features or feature_order)], dtype=float)
        depth_rows.append((depth_km, dataset_dir.name, effect_vec, report["target_cols"][0]))

    if not depth_rows:
        raise RuntimeError(f"No matching single-depth datasets found for suite={args.suite} variant={args.variant}")

    depth_rows.sort(key=lambda item: item[0])
    depths = [row[0] for row in depth_rows]
    matrix = np.vstack([row[2] for row in depth_rows])
    target_col = depth_rows[0][3]
    plot_features = present_features or feature_order

    vmax = 60.0

    fig, ax = plt.subplots(figsize=(8.8, 6.2), constrained_layout=True)
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdBu_r",
        origin="upper",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xlabel("Parameter", fontsize=12)
    ax.set_ylabel("Depth (km)", fontsize=12)
    ax.set_xticks(np.arange(len(plot_features)))
    ax.set_xticklabels([label_map.get(feat, feat) for feat in plot_features], fontsize=11)
    ax.set_yticks(np.arange(len(depths)))
    ax.set_yticklabels([str(depth) for depth in depths], fontsize=11)
    ax.tick_params(axis="x", rotation=20)
    # Add thin white separators so the parameter columns read as distinct panels.
    for x in np.arange(0.5, len(plot_features) - 0.5 + 1e-9, 1.0):
        ax.axvline(x, color="white", lw=1.0, alpha=0.9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(_target_label(target_col), fontsize=11)

    if args.annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if np.isnan(val):
                    continue
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if abs(val) < vmax * 0.55 else "white",
                )

    out_path = outdir / f"{args.suite}_{args.variant}_{args.model_tag}_sensitivity_all_depths.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()

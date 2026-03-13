#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm, colors


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
    p = argparse.ArgumentParser(description="Plot stacked semi-3D response surfaces across depths.")
    p.add_argument("--suite", required=True)
    p.add_argument("--depths", default="10,40,70", help="Comma-separated depths in km.")
    p.add_argument("--variant", default="dTdt")
    p.add_argument("--model-tag", default="gp_m25")
    p.add_argument("--x-feature", default="v_conv")
    p.add_argument("--y-feature", default="age_OP")
    p.add_argument("--grid-size", type=int, default=61)
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

    depths = [int(d.strip()) for d in args.depths.split(",") if d.strip()]
    if not depths:
        raise ValueError("No depths provided.")

    data_root = Path(args.data_root).resolve() / args.suite / "runs"
    models_root = Path(args.models_root).resolve() / args.suite / "runs"
    outdir = Path(args.outdir).resolve() if args.outdir else REPO_ROOT / "plots" / "science-emulator" / args.suite
    outdir.mkdir(parents=True, exist_ok=True)

    surfaces: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, str]] = []
    labels = _feature_label_map()

    for depth in depths:
        data_name = f"{depth}km_{args.variant}"
        data_dir = data_root / data_name
        model_dir = models_root / data_name / args.model_tag
        if not data_dir.exists() or not model_dir.exists():
            raise FileNotFoundError(f"Missing dataset/model for {data_name}")

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

        x_vals = np.linspace(*np.quantile(x_train[:, y_idx], [q_low, q_high]), args.grid_size)
        y_vals = np.linspace(*np.quantile(x_train[:, x_idx], [q_low, q_high]), args.grid_size)
        xx, yy = np.meshgrid(x_vals, y_vals)

        x_eval = np.repeat(baseline.reshape(1, -1), xx.size, axis=0)
        x_eval[:, x_idx] = yy.ravel()
        x_eval[:, y_idx] = xx.ravel()
        z_val = _predict_raw(
            model,
            x_eval,
            np.asarray(bundle["x_mean"], dtype=float),
            np.asarray(bundle["x_std"], dtype=float),
            np.asarray(bundle["y_mean"], dtype=float),
            np.asarray(bundle["y_std"], dtype=float),
        )[:, 0].reshape(xx.shape)
        surfaces.append((depth, xx, yy, z_val, report["target_cols"][0]))

    target_col = surfaces[0][4]
    zmin_data = min(float(np.min(surface[3])) for surface in surfaces)
    zmax_data = max(float(np.max(surface[3])) for surface in surfaces)
    level_min = 10.0 * np.floor(zmin_data / 10.0)
    level_max = 10.0 * np.ceil(zmax_data / 10.0)
    contour_levels = np.arange(level_min, level_max + 0.1, 10.0)
    norm = colors.Normalize(vmin=level_min, vmax=level_max)
    cmap = plt.get_cmap("viridis")

    fig = plt.figure(figsize=(8.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")

    depth_min = min(depths)
    depth_max = max(depths)
    # Use a stretched plotting axis so the depth planes do not overlap too much
    # in perspective, while still labeling them with the true sampled depths.
    depth_positions = np.linspace(depth_min, depth_max + 1.0 * (depth_max - depth_min), len(depths))

    contour_offset = 0.01 * max(depth_positions[-1] - depth_positions[0], 1.0)

    for depth_plot, (depth, xx, yy, values, _) in zip(depth_positions, surfaces):
        plane = np.full_like(xx, float(depth_plot))

        # A light semi-transparent sheet preserves the stacked-surface feel
        # without overwhelming the contour information.
        facecolors = cmap(norm(values))
        facecolors[..., -1] = 0.42
        ax.plot_surface(
            xx,
            yy,
            plane,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False,
            shade=False,
            alpha=None,
        )

        # A faint wireframe helps the planes read as surfaces.
        ax.plot_wireframe(
            xx,
            yy,
            plane,
            rstride=max(1, xx.shape[0] // 14),
            cstride=max(1, xx.shape[1] // 14),
            color=(0.65, 0.65, 0.65, 0.18),
            linewidth=0.35,
        )

        # Draw a perimeter border around each depth layer so the stack reads as
        # distinct sheets rather than floating contours.
        border_color = (0.2, 0.2, 0.2, 0.9)
        ax.plot(xx[0, :], yy[0, :], plane[0, :], color=border_color, linewidth=1.2)
        ax.plot(xx[-1, :], yy[-1, :], plane[-1, :], color=border_color, linewidth=1.2)
        ax.plot(xx[:, 0], yy[:, 0], plane[:, 0], color=border_color, linewidth=1.2)
        ax.plot(xx[:, -1], yy[:, -1], plane[:, -1], color=border_color, linewidth=1.2)

        # Build 2D contours, then lift them into 3D as explicit lines slightly
        # above each plane. Color them by the contour value so the colorbar still
        # has meaning even without a filled surface.
        tmp_fig, tmp_ax = plt.subplots()
        contour_set = tmp_ax.contour(xx, yy, values, levels=contour_levels)
        plt.close(tmp_fig)
        for level, level_segs in zip(contour_set.levels, contour_set.allsegs):
            color = cmap(norm(level))
            for seg in level_segs:
                if len(seg) < 2:
                    continue
                ax.plot(
                    seg[:, 0],
                    seg[:, 1],
                    np.full(seg.shape[0], float(depth_plot - contour_offset)),
                    color=color,
                    linewidth=1.8,
                    alpha=0.98,
                )

    ax.set_xlabel(labels.get(args.y_feature, args.y_feature), labelpad=10)
    ax.set_ylabel(labels.get(args.x_feature, args.x_feature), labelpad=10)
    ax.set_zlabel("Depth (km)", labelpad=8)
    ax.invert_xaxis()
    ax.set_zticks(depth_positions)
    ax.set_zticklabels([str(depth) for depth in depths])
    z_pad = 0.06 * max(depth_positions[-1] - depth_positions[0], 1.0)
    ax.set_zlim(depth_positions[0] - z_pad, depth_positions[-1] + z_pad)
    ax.invert_zaxis()
    ax.set_box_aspect((1.0, 1.0, 1.45))
    ax.view_init(elev=28, azim=55)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]["linewidth"] = 0
    ax.yaxis._axinfo["grid"]["linewidth"] = 0
    ax.zaxis._axinfo["grid"]["linewidth"] = 0

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cax = fig.add_axes([0.62, 0.16, 0.03, 0.68])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(_target_label(target_col), fontsize=11)
    cbar.solids.set_alpha(0.42)
    for level in contour_levels:
        cbar.ax.hlines(
            level,
            0.0,
            1.0,
            colors=[cmap(norm(level))],
            linewidth=1.0,
            transform=cbar.ax.get_yaxis_transform(),
        )

    out_path = outdir / f"{args.suite}_{args.variant}_{args.model_tag}_stacked_surface_{args.x_feature}_vs_{args.y_feature}.png"
    fig.subplots_adjust(left=0.06, right=0.61, bottom=0.15, top=0.98)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_params(params_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(params_csv).copy()
    df["run_id"] = np.arange(len(df), dtype=int)
    return df


def pick_param_cols(df: pd.DataFrame, requested: list[str] | None) -> list[str]:
    if requested:
        missing = [c for c in requested if c not in df.columns]
        if missing:
            raise SystemExit(f"Requested params missing from CSV: {missing}")
        return requested

    candidates = ["v_conv", "age_SP", "age_OP", "dip_int", "eta_int", "eta_UM", "eps_trans", "t_conv"]
    out = [c for c in candidates if c in df.columns]
    if not out:
        raise SystemExit("No default parameter columns found in params CSV.")
    return out


def maybe_log_transform(x: pd.Series, name: str, use_log: bool) -> pd.Series:
    if not use_log:
        return x
    if name in {"eta_int", "eta_UM", "eps_trans"}:
        with np.errstate(invalid="ignore"):
            return np.log10(x.where(x > 0))
    return x


def corr_value(x: pd.Series, y: pd.Series, method: str) -> float:
    m = np.isfinite(x.to_numpy(float)) & np.isfinite(y.to_numpy(float))
    if int(m.sum()) < 3:
        return np.nan
    xv = x.to_numpy(float)[m]
    yv = y.to_numpy(float)[m]
    if np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
        return np.nan
    return float(pd.Series(xv).corr(pd.Series(yv), method=method))


def main() -> None:
    ap = argparse.ArgumentParser(description="Depth x parameter correlation heatmap for master DT data.")
    ap.add_argument("--params", required=True, help="Path to params-list.<suite>.csv")
    ap.add_argument("--master", required=True, help="Path to master_DT*.csv")
    ap.add_argument("--y", default="dTdt_C_per_Myr", help="Target column in master CSV")
    ap.add_argument("--method", choices=["pearson", "spearman"], default="pearson")
    ap.add_argument("--param-cols", nargs="+", default=None, help="Optional explicit parameter columns.")
    ap.add_argument("--no-log", action="store_true", help="Disable log10 transform of viscosity-like params.")
    ap.add_argument("--save-csv", action="store_true", help="Also save correlation matrix CSV (default: off).")
    ap.add_argument("--cmap", default="RdBu_r", help="Matplotlib colormap for the heatmap (default: RdBu_r).")
    ap.add_argument("--no-contours", action="store_true", help="Disable contour overlays.")
    ap.add_argument("--out", required=True, help="Output prefix (without .png/.csv)")
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    params_path = Path(args.params).resolve()
    master_path = Path(args.master).resolve()
    out_prefix = Path(args.out).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df_p = load_params(params_path)
    df_m = pd.read_csv(master_path).copy()

    need_master = {"run_id", "depth_km", args.y}
    if not need_master.issubset(df_m.columns):
        raise SystemExit(f"Master CSV missing required columns: {sorted(need_master - set(df_m.columns))}")

    df_m["run_id"] = pd.to_numeric(df_m["run_id"], errors="coerce")
    df_m["depth_km"] = pd.to_numeric(df_m["depth_km"], errors="coerce")
    df_m[args.y] = pd.to_numeric(df_m[args.y], errors="coerce")
    df_m = df_m.dropna(subset=["run_id", "depth_km"])
    df_m["run_id"] = df_m["run_id"].astype(int)

    param_cols = pick_param_cols(df_p, args.param_cols)
    use_log = not args.no_log

    depths = np.sort(df_m["depth_km"].unique())
    corr_mat = np.full((len(depths), len(param_cols)), np.nan, dtype=float)

    for i, depth in enumerate(depths):
        sub = df_m[np.isclose(df_m["depth_km"], depth)][["run_id", args.y]].copy()
        merged = sub.merge(df_p[["run_id"] + param_cols], on="run_id", how="inner")

        y = merged[args.y]
        for j, pname in enumerate(param_cols):
            x = maybe_log_transform(merged[pname], pname, use_log)
            corr_mat[i, j] = corr_value(x, y, method=args.method)

    corr_df = pd.DataFrame(corr_mat, index=depths, columns=param_cols)
    if args.save_csv:
        corr_csv = out_prefix.with_suffix(".csv")
        corr_df.to_csv(corr_csv, index_label="depth_km")

    fig, ax = plt.subplots(figsize=(1.6 + 1.0 * len(param_cols), 4.8))
    im = ax.imshow(
        corr_mat,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=args.cmap,
        vmin=-1,
        vmax=1,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(f"{args.method} corr")

    ax.set_xticks(np.arange(len(param_cols)))
    ax.set_xticklabels(param_cols, rotation=35, ha="right")

    wanted_depth_labels = np.arange(0, 81, 10, dtype=float)
    yt_idx = []
    yt_lab = []
    for d in wanted_depth_labels:
        if len(depths) == 0:
            continue
        k = int(np.argmin(np.abs(depths - d)))
        if k not in yt_idx:
            yt_idx.append(k)
            yt_lab.append(str(int(d)))
    ax.set_yticks(yt_idx)
    ax.set_yticklabels(yt_lab)
    ax.set_ylabel("depth_km")
    ax.set_title(f"{args.y} vs params: {args.method} correlation")
    ax.set_xticks(np.arange(-0.5, len(param_cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(depths), 1), minor=True)
    ax.grid(which="minor", color="k", alpha=0.10, linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    if not args.no_contours:
        # Draw per-parameter iso-correlation segments so lines never connect across columns.
        levels_major = np.arange(-0.8, 0.81, 0.2)
        levels_minor = np.arange(-0.9, 0.91, 0.1)
        levels_minor = np.array([lv for lv in levels_minor if np.all(np.abs(lv - levels_major) > 1e-9)])
        y_idx = np.arange(len(depths), dtype=float)
        for j in range(len(param_cols)):
            z = corr_mat[:, j]
            finite = np.isfinite(z)
            if finite.sum() < 2:
                continue
            for lvl in levels_minor:
                for k in range(len(z) - 1):
                    z0, z1 = z[k], z[k + 1]
                    if not (np.isfinite(z0) and np.isfinite(z1)):
                        continue
                    d0 = z0 - lvl
                    d1 = z1 - lvl
                    if d0 == 0:
                        y_cross = y_idx[k]
                    elif d0 * d1 > 0:
                        continue
                    else:
                        frac = abs(d0) / (abs(d0) + abs(d1))
                        y_cross = y_idx[k] + frac
                    ax.hlines(
                        y_cross,
                        j - 0.42,
                        j + 0.42,
                        colors="k",
                        linewidth=0.42,
                        alpha=0.35,
                        linestyles="--",
                    )
            for lvl in levels_major:
                for k in range(len(z) - 1):
                    z0, z1 = z[k], z[k + 1]
                    if not (np.isfinite(z0) and np.isfinite(z1)):
                        continue
                    d0 = z0 - lvl
                    d1 = z1 - lvl
                    if d0 == 0:
                        y_cross = y_idx[k]
                    elif d0 * d1 > 0:
                        continue
                    else:
                        frac = abs(d0) / (abs(d0) + abs(d1))
                        y_cross = y_idx[k] + frac
                    ax.hlines(
                        y_cross,
                        j - 0.42,
                        j + 0.42,
                        colors="k",
                        linewidth=0.55,
                        alpha=0.5,
                    )
                    ax.text(
                        j + 0.30,
                        y_cross,
                        f"{lvl:.1f}",
                        fontsize=5.5,
                        color="k",
                        alpha=0.70,
                        ha="left",
                        va="center",
                    )

    fig.tight_layout()
    out_png = out_prefix.with_suffix(".png")
    fig.savefig(out_png, dpi=args.dpi)
    plt.close(fig)

    if args.save_csv:
        print(f"[OK] wrote {corr_csv}")
    print(f"[OK] wrote {out_png}")


if __name__ == "__main__":
    main()

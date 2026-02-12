#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _pct(x: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * float(x) / float(total)


def build_run_summary(df: pd.DataFrame, ycol: str) -> pd.DataFrame:
    rows = []
    for run_id, sub in df.groupby("run_id", sort=True):
        n_total = len(sub)
        y = sub[ycol]
        n_missing = int(y.isna().sum())
        n_valid = int(y.notna().sum())
        rows.append(
            {
                "run_id": int(run_id),
                "n_total": n_total,
                "n_valid": n_valid,
                "n_missing": n_missing,
                "pct_missing": _pct(n_missing, n_total),
                "median": float(y.median()) if n_valid else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("run_id")


def plot_missing_heatmap(df: pd.DataFrame, ycol: str, out_png: Path) -> None:
    runs = np.sort(df["run_id"].unique())
    depths = np.sort(df["depth_km"].unique())
    pivot = (
        df.pivot_table(index="depth_km", columns="run_id", values=ycol, aggfunc="first")
        .reindex(index=depths, columns=runs)
    )
    miss = pivot.isna().astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(miss, aspect="auto", interpolation="nearest", cmap="Reds", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Missing (1=yes, 0=no)")
    ax.set_title("Master CSV Missingness (depth x run)")
    ax.set_xlabel("run_id index")
    ax.set_ylabel("depth_km")
    yt_idx = np.linspace(0, len(depths) - 1, min(10, len(depths)), dtype=int)
    ax.set_yticks(yt_idx)
    ax.set_yticklabels([str(int(depths[i])) for i in yt_idx])
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="QC health checks for master_DT*.csv before ML preprocessing.")
    ap.add_argument("--master", required=True, help="Path to master_DT*.csv")
    ap.add_argument("--out", required=True, help="Output prefix, e.g. plots/.../master_healthcheck")
    ap.add_argument("--y", default="dTdt_C_per_Myr", help="Target column name")
    args = ap.parse_args()

    master = Path(args.master)
    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(master)
    needed = {"depth_km", "run_id", args.y}
    missing_cols = needed - set(df.columns)
    if missing_cols:
        raise SystemExit(f"Missing required columns: {sorted(missing_cols)}")

    df["depth_km"] = pd.to_numeric(df["depth_km"], errors="coerce")
    df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce")
    df[args.y] = pd.to_numeric(df[args.y], errors="coerce")
    df = df.dropna(subset=["depth_km", "run_id"]).copy()
    df["run_id"] = df["run_id"].astype(int)

    run_summary = build_run_summary(df, args.y)

    run_csv = out_prefix.with_name(out_prefix.name + "_by-run.csv")
    run_summary.to_csv(run_csv, index=False)

    plot_missing_heatmap(df, args.y, out_prefix.with_name(out_prefix.name + "_missing-heatmap.png"))

    print(f"[OK] wrote {run_csv}")


if __name__ == "__main__":
    main()

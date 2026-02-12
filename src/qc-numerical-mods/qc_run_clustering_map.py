#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_profiles(master_csv: Path, ycol: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(master_csv).copy()
    need = {"run_id", "depth_km", ycol}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"Master CSV missing columns: {sorted(miss)}")

    df["run_id"] = pd.to_numeric(df["run_id"], errors="coerce")
    df["depth_km"] = pd.to_numeric(df["depth_km"], errors="coerce")
    df[ycol] = pd.to_numeric(df[ycol], errors="coerce")
    df = df.dropna(subset=["run_id", "depth_km"])
    df["run_id"] = df["run_id"].astype(int)

    depths = np.sort(df["depth_km"].unique())
    mat = (
        df.pivot_table(index="run_id", columns="depth_km", values=ycol, aggfunc="mean")
        .reindex(columns=depths)
        .sort_index()
    )

    # Fill missing values with per-depth medians for stable clustering.
    med = mat.median(axis=0)
    mat = mat.apply(lambda col: col.fillna(med[col.name]), axis=0)

    # Drop rows that remain fully missing after fill (unlikely).
    mat = mat.dropna(axis=0, how="all")
    return mat, depths


def load_params(params_csv: Path, n_rows: int) -> pd.DataFrame:
    df = pd.read_csv(params_csv).copy()
    df["run_id"] = np.arange(len(df), dtype=int)
    if len(df) < n_rows:
        raise SystemExit("Params CSV has fewer rows than expected run count.")
    return df


def choose_pairs(df: pd.DataFrame) -> tuple[tuple[str, str], tuple[str, str]]:
    cols = set(df.columns)
    pair1 = ("v_conv", "age_SP") if {"v_conv", "age_SP"}.issubset(cols) else None
    pair2 = ("dip_int", "eta_UM") if {"dip_int", "eta_UM"}.issubset(cols) else None

    numeric = [c for c in df.columns if c != "run_id" and pd.api.types.is_numeric_dtype(df[c])]
    if pair1 is None and len(numeric) >= 2:
        pair1 = (numeric[0], numeric[1])
    if pair2 is None:
        rest = [c for c in numeric if c not in set(pair1 or ())]
        if len(rest) >= 2:
            pair2 = (rest[0], rest[1])
        elif len(numeric) >= 2:
            pair2 = (numeric[0], numeric[1])
        else:
            raise SystemExit("Not enough numeric parameter columns for plotting.")
    return pair1, pair2


def maybe_log(v: np.ndarray, name: str) -> np.ndarray:
    if name in {"eta_int", "eta_UM", "eps_trans"}:
        out = np.full_like(v, np.nan, dtype=float)
        m = np.isfinite(v) & (v > 0)
        out[m] = np.log10(v[m])
        return out
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description="Cluster runs by dTdt(depth) profile and map clusters in parameter space.")
    ap.add_argument("--params", required=True, help="Path to params-list.<suite>.csv")
    ap.add_argument("--master", required=True, help="Path to master_DT*.csv")
    ap.add_argument("--y", default="dTdt_C_per_Myr", help="Profile target column to cluster on.")
    ap.add_argument("--n-clusters", type=int, default=4, help="KMeans cluster count.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--save-csv", action="store_true", help="Save run_id -> cluster CSV (default: off).")
    ap.add_argument("--out", required=True, help="Output prefix without extension.")
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    params_path = Path(args.params).resolve()
    master_path = Path(args.master).resolve()
    out_prefix = Path(args.out).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    prof, depths = load_profiles(master_path, args.y)
    X = prof.to_numpy(float)
    run_ids = prof.index.to_numpy(int)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=20)
    labels = km.fit_predict(Xs)

    params = load_params(params_path, n_rows=int(run_ids.max()) + 1)
    df = params.merge(pd.DataFrame({"run_id": run_ids, "cluster": labels}), on="run_id", how="inner")

    pair1, pair2 = choose_pairs(df)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    cmap = plt.get_cmap("tab10", args.n_clusters)

    for ax, pair in zip(axes[:2], (pair1, pair2)):
        xname, yname = pair
        x = maybe_log(df[xname].to_numpy(float), xname)
        y = maybe_log(df[yname].to_numpy(float), yname)
        for k in range(args.n_clusters):
            m = df["cluster"].to_numpy(int) == k
            ax.scatter(x[m], y[m], s=20, alpha=0.8, color=cmap(k), label=f"C{k}" if ax is axes[0] else None)
        ax.set_xlabel(f"log10({xname})" if xname in {"eta_int", "eta_UM", "eps_trans"} else xname)
        ax.set_ylabel(f"log10({yname})" if yname in {"eta_int", "eta_UM", "eps_trans"} else yname)
        ax.grid(alpha=0.3, linestyle=":")

    # Third panel: PCA view of clustered profile space.
    pca = PCA(n_components=2, random_state=args.seed)
    Z = pca.fit_transform(Xs)
    for k in range(args.n_clusters):
        m = labels == k
        axes[2].scatter(Z[m, 0], Z[m, 1], s=20, alpha=0.85, color=cmap(k), label=f"C{k}")
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")
    axes[2].set_title(f"PCA of {args.y}(depth)")
    axes[2].grid(alpha=0.3, linestyle=":")

    axes[0].legend(frameon=False, title="Cluster")
    fig.suptitle(f"Run Clusters (k={args.n_clusters}) from {args.y} depth profiles", y=1.02)

    out_png = out_prefix.with_suffix(".png")
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] wrote {out_png}")
    if args.save_csv:
        out_csv = out_prefix.with_name(out_prefix.name + "_assignments.csv")
        pd.DataFrame({"run_id": run_ids, "cluster": labels}).to_csv(out_csv, index=False)
        print(f"[OK] wrote {out_csv}")


if __name__ == "__main__":
    main()

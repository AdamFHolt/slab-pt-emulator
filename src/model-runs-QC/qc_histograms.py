#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PARAMS = ["v_conv","age_SP","age_OP","dip_int","eta_int","eta_UM","eps_trans"]

def load_master(master_path):
    df = pd.read_csv(master_path, dtype={"run_id": str})
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", default="../../data/params-list.csv")
    p.add_argument("--master", required=True, help="e.g. ../../subd-model-runs/run-outputs/analysis/master_50km_DT1-6.csv")
    p.add_argument("--out-prefix", default="../subd-model-runs/run-outputs/analysis/figs/histograms_")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    params_path = Path(args.params).resolve()
    master_path = Path(args.master).resolve()
    out_prefix = Path(args.out_prefix).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    dfp = pd.read_csv(params_path)
    dfp["run_id"] = [f"{i:03d}" for i in range(len(dfp))]
    dfm = load_master(master_path)
    df = pd.merge(dfp, dfm, on="run_id", how="inner")

    # 2A) Parameter histograms
    n = len(PARAMS)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2*nrows), constrained_layout=True)
    axes = axes.flatten()
    for i, col in enumerate(PARAMS):
        ax = axes[i]
        ax.hist(df[col].dropna(), bins=30, alpha=0.85)
        ax.set_title(col)
    for j in range(i+1, len(axes)): axes[j].axis("off")
    fig.suptitle("Parameter marginals", y=1.02)
    fig.savefig(f"{out_prefix}params.png", dpi=args.dpi, bbox_inches="tight")

    # 2B) Response histograms
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    axes2[0].hist(df["dT_C"].dropna(), bins=30, alpha=0.85)
    axes2[0].set_xlabel("ΔT (°C)"); axes2[0].set_title("ΔT distribution")
    axes2[1].hist(df["dTdt_C_per_Myr"].dropna(), bins=30, alpha=0.85)
    axes2[1].set_xlabel("ΔT/Δt (°C/Myr)"); axes2[1].set_title("ΔT/Δt distribution")
    fig2.savefig(f"{out_prefix}responses.png", dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out_prefix}params.png and {out_prefix}responses.png")

if __name__ == "__main__":
    main()


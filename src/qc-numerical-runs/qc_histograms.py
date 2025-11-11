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
    p.add_argument("--out", default="../../plots/numerical-mods/qc_histograms")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    params_path = Path(args.params).resolve()
    master_path = Path(args.master).resolve()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dfp = pd.read_csv(params_path)
    dfp["run_id"] = [f"{i:03d}" for i in range(len(dfp))]
    dfm = load_master(master_path)
    df = pd.merge(dfp, dfm, on="run_id", how="inner")

    # A) Parameter histograms
    n = len(PARAMS)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2*nrows), constrained_layout=True)
    axes = axes.flatten()
    for i, col in enumerate(PARAMS):
        ax = axes[i]
        ax.hist(df[col].dropna(), bins=30, alpha=0.85)
        ax.set_title(col)

    # B) Response histograms
    axes[7].hist(df["dT_C"].dropna(), bins=30, alpha=0.85, color='skyblue')
    axes[7].set_title("ΔT (°C)")
    axes[8].hist(df["dTdt_C_per_Myr"].dropna(), bins=30, alpha=0.85, color='skyblue')
    axes[8].set_title("ΔT/Δt (°C/Myr)")

    fig.savefig(f"{out_path}.png", dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out_path}.png")

if __name__ == "__main__":
    main()


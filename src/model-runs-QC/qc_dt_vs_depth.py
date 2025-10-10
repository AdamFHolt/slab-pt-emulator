#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def depth_from_fname(path):
    m = re.search(r"master_(\d+)km", Path(path).name)
    return int(m.group(1)) if m else None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--masters", nargs="+", required=True,
                   help="List of master CSVs for depths, e.g. master_25km_DT1-6.csv master_50km_DT1-6.csv ...")
    p.add_argument("--y", default="dTdt_C_per_Myr", choices=["dT_C","dTdt_C_per_Myr"])
    p.add_argument("--out", default="../../subd-model-runs/run-outputs/analysis/plots/DT_vs_depth.png")
    p.add_argument("--dpi", type=int, default=200)
    args = p.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    masters = sorted(args.masters, key=lambda s: depth_from_fname(s) or 0)
    n = len(masters)
    fig, axes = plt.subplots(1, n, figsize=(4.0*n, 3.6), constrained_layout=True)

    if n == 1: axes = [axes]
    for ax, mfile in zip(axes, masters):
        depth = depth_from_fname(mfile)
        df = pd.read_csv(mfile, dtype={"run_id": str})
        y = df[args.y].to_numpy(float)
        ax.scatter(np.arange(len(y)), y, s=14, alpha=0.8)
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.set_title(f"{args.y} @ {depth} km")
        ax.set_xlabel("run index")
        ax.set_ylabel(args.y)
        ax.grid(True, ls=":", alpha=0.4)

    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()

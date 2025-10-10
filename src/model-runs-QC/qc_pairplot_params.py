#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

RAW_PARAMS = ["v_conv","age_SP","age_OP","dip_int","eta_int","eta_UM","eps_trans"]
LOG_AUTO = ["eta_int","eta_UM","eps_trans"]  # heavy-tailed; plot in log10

def main():
    p = argparse.ArgumentParser(description="Pairplot of LHS parameters with optional log handling.")
    p.add_argument("--params", default="../../data/params-list.csv")
    p.add_argument("--out", default="../../subd-model-runs/run-outputs/analysis/plots/pairplot_params.png")
    p.add_argument("--dpi", type=int, default=200)

    # How to treat the 3 heavy-tailed params
    p.add_argument("--mode", choices=["compute-log10","already-log10","linear"], default="compute-log10",
                   help=("compute-log10: take log10 for eta_int/eta_UM/eps_trans before plotting (default); "
                         "already-log10: CSV already contains log10 values—just relabel; "
                         "linear: plot raw linear values."))
    args = p.parse_args()

    params_path = Path(args.params).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(params_path)[RAW_PARAMS].copy()

    df_plot = df.copy()
    if args.mode == "compute-log10":
        for col in LOG_AUTO:
            # guard against nonpositive values
            x = df_plot[col].to_numpy(float)
            x[x <= 0] = np.nan
            df_plot[f"log10({col})"] = np.log10(x)
            df_plot.drop(columns=[col], inplace=True)
    elif args.mode == "already-log10":
        # just relabel the three columns as log10(...) for clarity
        rename = {c: f"log10({c})" for c in LOG_AUTO}
        df_plot.rename(columns=rename, inplace=True)
    else:  # linear
        pass

    # Order columns: keep same order, with any renamed log cols in place
    ordered_cols = []
    for c in RAW_PARAMS:
        ordered_cols.append(f"log10({c})" if args.mode != "linear" and c in LOG_AUTO else c)
    df_plot = df_plot[ordered_cols]

    g = sns.pairplot(df_plot, corner=True, diag_kind="hist",
                     plot_kws=dict(s=18, alpha=0.7), diag_kws=dict(edgecolor="none"))

    g.fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()

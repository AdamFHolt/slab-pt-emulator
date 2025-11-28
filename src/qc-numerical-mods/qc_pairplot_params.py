#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Active parameters in the new suite:
RAW_PARAMS = ["v_conv", "age_SP", "age_OP", "dip_int", "eta_UM"]

# Heavy-tailed; plot in log10 if desired
LOG_AUTO = ["eta_UM"]


def main():
    p = argparse.ArgumentParser(
        description="Pairplot of LHS parameters with optional log10 handling."
    )
    p.add_argument(
        "--params",
        default="../../data/params/params-list.const-vc.csv",
        help="Path to params-list.<suite>.csv (LHS design).",
    )
    p.add_argument(
        "--out",
        default="../../plots/qc-numerical-mods/const-vc_pairplot_params",
        help="Output path prefix (without extension).",
    )
    p.add_argument("--dpi", type=int, default=200)

    # How to treat the heavy-tailed params (eta_UM here)
    p.add_argument(
        "--mode",
        choices=["compute-log10", "already-log10", "linear"],
        default="compute-log10",
        help=(
            "compute-log10: take log10 for eta_UM before plotting (default); "
            "already-log10: CSV already contains log10 values—just relabel; "
            'linear: plot raw linear values (no log transform).'
        ),
    )
    args = p.parse_args()

    params_path = Path(args.params).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load only the relevant columns (robust to extra columns in CSV)
    df = pd.read_csv(params_path)
    missing = [c for c in RAW_PARAMS if c not in df.columns]
    if missing:
        raise ValueError(
            f"The following required columns are missing from {params_path}:\n{missing}"
        )
    df = df[RAW_PARAMS].copy()

    df_plot = df.copy()

    if args.mode == "compute-log10":
        for col in LOG_AUTO:
            if col not in df_plot.columns:
                continue
            x = df_plot[col].to_numpy(float)
            # Avoid log of nonpositive values
            x[x <= 0] = np.nan
            df_plot[f"log10({col})"] = np.log10(x)
            df_plot.drop(columns=[col], inplace=True)

    elif args.mode == "already-log10":
        # Just relabel the three columns as log10(...) for clarity
        rename = {c: f"log10({c})" for c in LOG_AUTO if c in df_plot.columns}
        df_plot.rename(columns=rename, inplace=True)
    else:
        # linear: keep raw values
        pass

    # Order columns: keep same conceptual order, with log10(...) names where applicable
    ordered_cols = []
    for c in RAW_PARAMS:
        if args.mode != "linear" and c in LOG_AUTO:
            ordered_cols.append(f"log10({c})")
        else:
            ordered_cols.append(c)

    df_plot = df_plot[ordered_cols]

    g = sns.pairplot(
        df_plot,
        corner=True,
        diag_kind="hist",
        plot_kws=dict(s=18, alpha=0.7),
        diag_kws=dict(edgecolor="none"),
    )

    g.fig.savefig(f"{out_path}.png", dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {out_path}.png")


if __name__ == "__main__":
    main()


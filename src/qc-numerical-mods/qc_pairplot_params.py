#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Base parameters (present in both suites)
RAW_BASE_PARAMS = ["v_conv", "age_SP", "age_OP", "dip_int", "eta_UM"]

# Optional parameters (may exist in ramped-vc, but not const-vc)
OPTIONAL_PARAMS = ["t_conv"]

# Full list in conceptual order
RAW_PARAMS_ALL = RAW_BASE_PARAMS + OPTIONAL_PARAMS

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

    # Load CSV
    df = pd.read_csv(params_path)

    # Ensure all base params exist (these should be present in both const-vc and ramped-vc)
    missing_base = [c for c in RAW_BASE_PARAMS if c not in df.columns]
    if missing_base:
        raise ValueError(
            f"The following required columns are missing from {params_path}:\n{missing_base}"
        )

    # Keep only known params that actually exist in this suite (e.g., t_conv only for ramped-vc)
    present_cols = [c for c in RAW_PARAMS_ALL if c in df.columns]
    df = df[present_cols].copy()

    df_plot = df.copy()

    # Handle log10 transforms
    if args.mode == "compute-log10":
        for col in LOG_AUTO:
            if col not in df_plot.columns:
                continue
            x = df_plot[col].to_numpy(float)
            x[x <= 0] = np.nan
            df_plot[f"log10({col})"] = np.log10(x)
            df_plot.drop(columns=[col], inplace=True)

    elif args.mode == "already-log10":
        rename = {c: f"log10({c})" for c in LOG_AUTO if c in df_plot.columns}
        df_plot.rename(columns=rename, inplace=True)
    else:
        # linear: keep raw values
        pass

    # Order columns: follow RAW_PARAMS_ALL, with log10 names where applicable,
    # skipping anything not present in df_plot (e.g., t_conv for const-vc).
    ordered_cols = []
    for c in RAW_PARAMS_ALL:
        log_name = f"log10({c})"
        if args.mode != "linear" and c in LOG_AUTO and log_name in df_plot.columns:
            ordered_cols.append(log_name)
        elif c in df_plot.columns:
            ordered_cols.append(c)
        # else: parameter not present in this suite -> skip

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

#!/usr/bin/env python3
"""Slab tip depth at a given output step, from the raw field CSVs.

Answers "does the subducting crust approach the base of the model box by the
end of the extended window?" by finding, per run, the deepest point where the
oceanic-crust composition exceeds a threshold, plus the deepest point of the
cold thermal anomaly.

Usage:
    qc_slab_tip_depth.py --suite const-vc --step 20 --runs 000,001,...  \
        --out plots/qc-numerical-mods/const-vc/slab_tip_depth_t20.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="const-vc")
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--runs", default=None, help="Comma list of run ids; default = all with the field CSV.")
    ap.add_argument("--c-thresh", type=float, default=0.5)
    ap.add_argument("--cold-anomaly-c", type=float, default=200.0,
                    help="Depth of the deepest point colder than the adiabat-free ambient by this much.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    analysis = REPO_ROOT / "subd-model-runs" / args.suite / "analysis"
    if args.runs:
        run_dirs = [analysis / f"run_{r.strip().replace('run_', '').zfill(3)}" for r in args.runs.split(",")]
    else:
        run_dirs = sorted(analysis.glob("run_*"))

    rows = []
    for rd in run_dirs:
        f = rd / f"t{args.step}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        xc = [c for c in df.columns if c.lower().startswith("points:0")][0]
        yc = [c for c in df.columns if c.lower().startswith("points:1")][0]
        cc = [c for c in df.columns if c.lower().startswith("ocrust")][0]
        y = df[yc].to_numpy(float) / 1e3
        ymax = y.max()
        depth = ymax - y
        C = df[cc].to_numpy(float)
        T = df["T"].to_numpy(float) - 273.15

        m = C >= args.c_thresh
        tip = float(depth[m].max()) if m.any() else np.nan
        x_tip = float(df[xc].to_numpy(float)[m][np.argmax(depth[m])] / 1e3) if m.any() else np.nan

        # Deepest strongly cold material: compare each point to the horizontal
        # median T at its depth level (1 km bins), which tracks ambient mantle.
        dbin = np.round(depth).astype(int)
        med = pd.Series(T).groupby(dbin).transform("median").to_numpy(float)
        cold = (T < med - args.cold_anomaly_c) & (depth > 20.0)
        cold_tip = float(depth[cold].max()) if cold.any() else np.nan

        rows.append(dict(run_id=rd.name[len("run_"):], step=args.step, box_depth_km=float(ymax),
                         crust_tip_depth_km=tip, crust_tip_x_km=x_tip,
                         cold_anomaly_tip_depth_km=cold_tip,
                         tip_to_base_km=float(ymax) - tip if np.isfinite(tip) else np.nan))

    out_df = pd.DataFrame(rows)
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"[OK] wrote {out}  (runs={len(out_df)})")
    if len(out_df):
        print(out_df.describe().to_string(float_format=lambda v: f"{v:.1f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

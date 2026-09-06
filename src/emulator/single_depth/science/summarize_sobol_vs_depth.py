#!/usr/bin/env python3
"""Collate Sobol JSONs from one or more cooling windows into a depth table.

Writes a long-format CSV (window, depth, parameter, S1, ST, confidence
intervals) and prints the total-effect table per window, plus the depth at
which the ST of two named parameters cross.

Usage:
    summarize_sobol_vs_depth.py \
        --window "0.5-5 Myr=plots/science-emulator/single_depth/const-vc/sobol" \
        --window "0.5-10 Myr=plots/science-emulator/single_depth/const-vc/sobol_dt1-20" \
        --window "5-10 Myr=plots/science-emulator/single_depth/const-vc/sobol_dt10-20" \
        --crossover age_OP,v_conv \
        --out plots/science-emulator/single_depth/const-vc/sobol_windows_summary.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", action="append", required=True,
                    help="'LABEL=DIR' pair; repeatable. DIR holds <depth>km_dTdt_<tag>_sobol.json.")
    ap.add_argument("--model-tag", default="gp_m25")
    ap.add_argument("--crossover", default="age_OP,v_conv",
                    help="Two parameter names whose ST crossover depth is reported.")
    ap.add_argument("--report-depths", default="10,20,30,40,50,60,80")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for spec in args.window:
        label, _, dirname = spec.partition("=")
        d = Path(dirname)
        if not d.is_absolute():
            d = REPO_ROOT / d
        for jp in sorted(d.glob(f"*_dTdt_{args.model_tag}_sobol.json")):
            m = re.match(r"^(\d+(?:\.\d+)?)km", jp.name)
            if not m:
                continue
            js = json.loads(jp.read_text())
            for i, name in enumerate(js["feature_cols"]):
                rows.append(
                    dict(
                        window=label,
                        depth_km=float(m.group(1)),
                        param=name,
                        S1=js["S1"][i],
                        S1_conf=js["S1_conf"][i],
                        ST=js["ST"][i],
                        ST_conf=js["ST_conf"][i],
                        val_r2=js.get("val_r2"),
                        val_rmse=js.get("val_rmse"),
                        n_base=js.get("n_base"),
                    )
                )

    df = pd.DataFrame(rows).sort_values(["window", "depth_km", "param"]).reset_index(drop=True)
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[OK] wrote {out}")

    report_depths = [float(x) for x in args.report_depths.split(",")]
    pa, pb = [s.strip() for s in args.crossover.split(",")]

    for label in df["window"].unique():
        sub = df[df["window"] == label]
        piv = sub.pivot_table(index="depth_km", columns="param", values="ST")
        print(f"\n=== total-effect ST, window {label} ===")
        print(piv.loc[[d for d in report_depths if d in piv.index]]
              .to_string(float_format=lambda v: f"{v:.3f}"))
        r2 = sub.groupby("depth_km")["val_r2"].first()
        rm = sub.groupby("depth_km")["val_rmse"].first()
        print("  emulator val R2 range : "
              f"{r2.min():.4f} - {r2.max():.4f}")
        print("  emulator val RMSE (C/Myr) range : "
              f"{rm.min():.3f} - {rm.max():.3f}")

        if pa in piv.columns and pb in piv.columns:
            z = piv.index.to_numpy(float)
            diff = (piv[pa] - piv[pb]).to_numpy(float)
            cross = []
            for i in range(len(z) - 1):
                if np.isfinite(diff[i]) and np.isfinite(diff[i + 1]) and diff[i] * diff[i + 1] < 0:
                    zc = z[i] + (z[i + 1] - z[i]) * (-diff[i]) / (diff[i + 1] - diff[i])
                    cross.append((z[i], z[i + 1], zc))
            if cross:
                for lo, hi, zc in cross:
                    print(f"  ST({pa}) crosses ST({pb}) between {lo:g} and {hi:g} km "
                          f"(linear estimate {zc:.1f} km)")
            else:
                print(f"  no ST crossover between {pa} and {pb} on the sampled depths "
                      f"(sign of ST({pa})-ST({pb}) is constant)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

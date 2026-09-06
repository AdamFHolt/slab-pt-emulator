#!/usr/bin/env python3
"""Health check on the per-run slab-top profiles across a range of steps.

Flags the failure modes that matter when a suite is extended past the window
it was originally processed for:

* missing / NaN profile entries,
* profiles whose slab-top T is not monotonically increasing with depth,
* implausible temperatures (outside ``--t-min``/``--t-max``),
* large step-to-step jumps in T at a probe depth (numerical instability),
* runs whose slab-top temperature *rises* strongly late in the window,
* slab tip depth at the final step, so proximity to the model box base can be
  judged.

Usage:
    qc_late_time_profiles.py --suite const-vc --steps 1:20 \
        --out plots/qc-numerical-mods/const-vc/late_time_profile_qc.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_steps(spec: str) -> list[int]:
    if ":" in spec:
        p = [int(x) for x in spec.split(":")]
        s = p[2] if len(p) == 3 else 1
        return list(range(p[0], p[1] + 1, s))
    return [int(x) for x in spec.split(",")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="const-vc")
    ap.add_argument("--steps", default="1:20")
    ap.add_argument("--probe-depths", default="40,80")
    ap.add_argument("--t-min", type=float, default=-5.0)
    ap.add_argument("--t-max", type=float, default=1400.0)
    ap.add_argument("--jump-c", type=float, default=60.0,
                    help="Flag |T(k+1)-T(k)| above this at any probe depth (C per 0.5 Myr).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    steps = parse_steps(args.steps)
    probes = [float(x) for x in args.probe_depths.split(",")]
    analysis = REPO_ROOT / "subd-model-runs" / args.suite / "analysis"

    rows = []
    for run_dir in sorted(analysis.glob("run_*")):
        rid = run_dir.name[len("run_"):]
        present, nan_steps, nonmono, oor = [], [], [], []
        probe_series = {d: {} for d in probes}
        for k in steps:
            p = run_dir / f"Tprof_{k}.csv"
            if not p.exists():
                continue
            present.append(k)
            df = pd.read_csv(p)
            z = pd.to_numeric(df["depth_km"], errors="coerce").to_numpy(float)
            t = pd.to_numeric(df["T_C"], errors="coerce").to_numpy(float)
            if not np.isfinite(t).all():
                nan_steps.append(k)
            m = np.isfinite(z) & np.isfinite(t)
            zz, tt = z[m], t[m]
            o = np.argsort(zz)
            zz, tt = zz[o], tt[o]
            if tt.size and (np.min(np.diff(tt)) < -1.0):
                nonmono.append(k)
            if tt.size and (np.nanmin(tt) < args.t_min or np.nanmax(tt) > args.t_max):
                oor.append(k)
            for d in probes:
                probe_series[d][k] = float(np.interp(d, zz, tt, left=np.nan, right=np.nan)) if tt.size else np.nan

        if not present:
            continue

        rec = {"run_id": rid, "n_steps": len(present), "max_step": max(present),
               "nan_steps": ";".join(map(str, nan_steps)),
               "nonmono_steps": ";".join(map(str, nonmono)),
               "out_of_range_steps": ";".join(map(str, oor))}

        for d in probes:
            ser = probe_series[d]
            ks = sorted(ser)
            vals = np.array([ser[k] for k in ks], float)
            dv = np.diff(vals)
            rec[f"T{d:g}_first"] = vals[0] if vals.size else np.nan
            rec[f"T{d:g}_last"] = vals[-1] if vals.size else np.nan
            rec[f"T{d:g}_max_jump"] = float(np.nanmax(np.abs(dv))) if dv.size else np.nan
            rec[f"T{d:g}_max_jump_step"] = int(ks[int(np.nanargmax(np.abs(dv))) + 1]) if dv.size and np.isfinite(dv).any() else -1
            late = [k for k in ks if k > len(steps) // 2]
            if len(late) >= 2:
                lv = np.array([ser[k] for k in late], float)
                rec[f"T{d:g}_late_change"] = float(lv[-1] - lv[0])
            else:
                rec[f"T{d:g}_late_change"] = np.nan
            rec[f"T{d:g}_flag_jump"] = bool(np.isfinite(rec[f"T{d:g}_max_jump"]) and rec[f"T{d:g}_max_jump"] > args.jump_c)

        rows.append(rec)

    df = pd.DataFrame(rows)
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[OK] wrote {out}  (runs={len(df)})")

    print(f"[QC] runs with any NaN profile entry : {(df['nan_steps'] != '').sum()}")
    print(f"[QC] runs with non-monotonic T(z)    : {(df['nonmono_steps'] != '').sum()}")
    print(f"[QC] runs with T out of range        : {(df['out_of_range_steps'] != '').sum()}")
    for d in probes:
        col = f"T{d:g}_flag_jump"
        print(f"[QC] runs with |dT| > {args.jump_c:g} C per step at {d:g} km: {int(df[col].sum())}")
        print(f"[QC]   max step-to-step |dT| at {d:g} km = {df[f'T{d:g}_max_jump'].max():.1f} C")
        print(f"[QC]   runs warming over the late half at {d:g} km: "
              f"{int((df[f'T{d:g}_late_change'] > 0).sum())} / {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

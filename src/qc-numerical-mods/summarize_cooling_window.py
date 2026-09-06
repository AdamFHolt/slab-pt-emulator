#!/usr/bin/env python3
"""Suite-wide slab-top cooling statistics over a set of output steps.

Reads the per-run ``Tprof_<step>.csv`` profiles for a suite and reports, at
each requested depth, the across-run distribution of slab-top temperature at
each step plus the temperature drop and mean cooling rate over each requested
window.  Written for the const-vc 0.5-10 Myr summary but suite/step agnostic.

Only runs that have every requested step are used, so all statistics come from
one common set of runs.

Usage:
    summarize_cooling_window.py --suite const-vc --steps 1,10,20 \
        --depths 40,80 --windows "1,10;10,20;1,20" \
        --out plots/science-emulator/cooling-window-10myr/const-vc_cooling_stats.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_profile(path: Path) -> tuple[float, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    z = pd.to_numeric(df["depth_km"], errors="coerce").to_numpy(float)
    t = pd.to_numeric(df["T_C"], errors="coerce").to_numpy(float)
    tm = float(pd.to_numeric(df["time_Myr"], errors="coerce").to_numpy(float)[0])
    m = np.isfinite(z) & np.isfinite(t)
    z, t = z[m], t[m]
    o = np.argsort(z)
    return tm, z[o], t[o]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="const-vc")
    ap.add_argument("--steps", default="1,10,20")
    ap.add_argument("--depths", default="40,80")
    ap.add_argument("--windows", default="1,10;10,20;1,20")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    steps = [int(s) for s in args.steps.split(",")]
    depths = [float(d) for d in args.depths.split(",")]
    windows = [tuple(int(x) for x in w.split(",")) for w in args.windows.split(";") if w.strip()]

    analysis = REPO_ROOT / "subd-model-runs" / args.suite / "analysis"

    # Collect runs that carry every requested step.
    T: dict[int, dict[str, np.ndarray]] = {k: {} for k in steps}
    times: dict[int, dict[str, float]] = {k: {} for k in steps}
    run_ids: list[str] = []
    for run_dir in sorted(analysis.glob("run_*")):
        rid = run_dir.name[len("run_"):]
        ok = True
        cache = {}
        for k in steps:
            p = run_dir / f"Tprof_{k}.csv"
            if not p.exists():
                ok = False
                break
            tm, z, t = load_profile(p)
            vals = np.interp(depths, z, t, left=np.nan, right=np.nan)
            if not np.isfinite(vals).all():
                ok = False
                break
            cache[k] = (tm, vals)
        if not ok:
            continue
        run_ids.append(rid)
        for k, (tm, vals) in cache.items():
            T[k][rid] = vals
            times[k][rid] = tm

    n = len(run_ids)
    if n == 0:
        raise SystemExit("no runs carry all requested steps")

    Tmat = {k: np.vstack([T[k][r] for r in run_ids]) for k in steps}       # (n_runs, n_depths)
    tvec = {k: np.array([times[k][r] for r in run_ids]) for k in steps}    # (n_runs,)

    def pct(a):
        return {
            "p5": float(np.percentile(a, 5)),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "mean": float(np.mean(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
        }

    out: dict = {
        "suite": args.suite,
        "n_runs": n,
        "run_ids": run_ids,
        "depths_km": depths,
        "steps": steps,
        "step_times_myr": {str(k): pct(tvec[k]) for k in steps},
        "T_at_step": {},
        "windows": {},
        "fraction_of_total_cooling_in_first_window": {},
    }

    for k in steps:
        out["T_at_step"][str(k)] = {
            f"{d:g}km": pct(Tmat[k][:, j]) for j, d in enumerate(depths)
        }

    dT_by_window: dict[tuple[int, int], np.ndarray] = {}
    for a, b in windows:
        dT = Tmat[b] - Tmat[a]                      # (n_runs, n_depths)
        dt = (tvec[b] - tvec[a])[:, None]           # (n_runs, 1)
        rate = dT / dt
        dT_by_window[(a, b)] = dT
        out["windows"][f"{a}-{b}"] = {
            "dt_myr": pct(dt[:, 0]),
            "dT_C": {f"{d:g}km": pct(dT[:, j]) for j, d in enumerate(depths)},
            "dTdt_C_per_Myr": {f"{d:g}km": pct(rate[:, j]) for j, d in enumerate(depths)},
        }

    # Fraction of the full-window cooling delivered by each sub-window.
    if len(windows) >= 3:
        full = windows[-1]
        for a, b in windows[:-1]:
            key = f"{a}-{b}_of_{full[0]}-{full[1]}"
            frac_per_run = dT_by_window[(a, b)] / dT_by_window[full]
            out["fraction_of_total_cooling_in_first_window"][key] = {
                "per_run": {f"{d:g}km": pct(frac_per_run[:, j]) for j, d in enumerate(depths)},
                "of_median_dT": {
                    f"{d:g}km": float(
                        np.median(dT_by_window[(a, b)][:, j]) / np.median(dT_by_window[full][:, j])
                    )
                    for j, d in enumerate(depths)
                },
                "of_mean_dT": {
                    f"{d:g}km": float(
                        np.mean(dT_by_window[(a, b)][:, j]) / np.mean(dT_by_window[full][:, j])
                    )
                    for j, d in enumerate(depths)
                },
            }

    outp = Path(args.out)
    if not outp.is_absolute():
        outp = REPO_ROOT / outp
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"[OK] wrote {outp}  (n_runs={n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

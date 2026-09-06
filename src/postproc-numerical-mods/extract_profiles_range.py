#!/usr/bin/env python3
"""Batch slab-top profile extraction over a range of output timesteps.

This is a faster, in-process equivalent of the per-timestep
``compute_slab_cooling.py`` calls made by
``extract_cooling-rates_one-mod.py``.  It exists so the const-vc suite can be
extended from 5 Myr (step 10) to 10 Myr (step 20) at reasonable wall-clock
cost.

For every requested timestep the field CSV ``t{k}.csv`` (already written by
``extract_csv``/pvpython) is read once, gridded once, and turned into
``Tprof_{k}.csv``.  Optional ``--dt-pairs`` additionally writes
``DT_{a}_{b}.csv`` files with exactly the column layout of the legacy script.

Numerical conventions replicate the legacy pipeline exactly:

* Tprof files: ``--interp linear`` gridding, smoothed interface x(depth)
  (Savitzky-Golay, 14 km window, polyorder 2), temperature sampled at the
  smoothed x.
* DT pair files: ``--interp nearest`` gridding (the legacy DT call does not
  pass ``--interp``, so it uses the argparse default), same smoothing.
* The interpolation grid is restricted to 0 <= depth <= ``--grid-depth-max-km``.
  ``scipy.interpolate.griddata`` is pointwise, so the retained rows are
  bit-identical to the full-depth grid used by the legacy script as long as
  the requested depths lie inside the restricted range.

Times are read from ``solution.pvd`` (same values ParaView reports).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.compute_slab_cooling import (  # noqa: E402
    build_xraw_profile,
    pick_rightmost_x_at_depth,
    read_field_csv,
    sample_T_at_x_depth,
    smooth_x_profile,
)
from scipy.interpolate import griddata  # noqa: E402


def read_pvd_times(pvd_path: Path) -> dict[int, float]:
    """Return {timestep index: time in years} from a deal.II solution.pvd."""
    txt = pvd_path.read_text()
    times = [float(m) for m in re.findall(r'timestep="([^"]+)"', txt)]
    return {i: t for i, t in enumerate(times)}


def grid_field_depth_limited(
    x_m, y_m, T_C, C, grid_res_km, xmin_km, xmax_km, ymax_km, depth_max_km, interp
):
    """Same as utils.compute_slab_cooling.grid_field but only 0..depth_max_km."""
    x_km = x_m / 1e3
    z_km = ymax_km - (y_m / 1e3)
    dx = float(grid_res_km)
    X, Z = np.meshgrid(
        np.arange(xmin_km, xmax_km + dx, dx),
        np.arange(0, depth_max_km + dx, dx),
    )
    GT = griddata((x_km, z_km), T_C, (X, Z), method=interp)
    GC = griddata((x_km, z_km), C, (X, Z), method=interp)
    return X, Z, GT, GC


def load_grid(path: Path, interp: str, grid_res_km: float, depth_max_km: float):
    x, y, T, C = read_field_csv(path)
    xmin_km, xmax_km = x.min() / 1e3, x.max() / 1e3
    ymax_km = y.max() / 1e3
    X, Z, GT, GC = grid_field_depth_limited(
        x, y, T, C, grid_res_km, xmin_km, xmax_km, ymax_km, depth_max_km, interp
    )
    return X, Z, GT, GC


def interface_picks(X, Z, GT, GC, depths, c_thresh, x_min_km, smooth_window_km, smooth_polyorder):
    """Return (x_raw, x_smooth, T_raw, T_sampled) on the requested depths."""
    depth_used, x_raw = build_xraw_profile(X, Z, GC, GT, depths, c_thresh, x_min_km)
    x_s = smooth_x_profile(depth_used, x_raw, smooth_window_km, smooth_polyorder)
    T_raw = np.array(
        [pick_rightmost_x_at_depth(X, Z, GC, GT, d, c_thresh, x_min_km)[1] for d in depth_used],
        float,
    )
    T_samp = np.array(
        [sample_T_at_x_depth(X, Z, GT, d, xk) for d, xk in zip(depth_used, x_s)], float
    )
    return x_raw, x_s, T_raw, T_samp


def _r3(v):
    return float(np.round(v, 3)) if np.isfinite(v) else np.nan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="const-vc")
    ap.add_argument("--run", required=True, help="Run id, e.g. 000 or run_000.")
    ap.add_argument("--tprof-steps", default="", help="Comma list or a:b range of timesteps for Tprof files.")
    ap.add_argument("--dt-pairs", default="", help="Semicolon list of 'a,b' pairs, e.g. '1,20;10,20'.")
    ap.add_argument("--depths", default="0:80:1", help="Depths in km: 'a:b:step' or comma list.")
    ap.add_argument("--grid-res-km", type=float, default=1.0)
    ap.add_argument("--grid-depth-max-km", type=float, default=120.0)
    ap.add_argument("--c-thresh", type=float, default=0.5)
    ap.add_argument("--x-min-km", type=float, default=1600.0)
    ap.add_argument("--smooth-window-km", type=float, default=14.0)
    ap.add_argument("--smooth-polyorder", type=int, default=2)
    ap.add_argument("--outdir", default=None, help="Override analysis dir (default: suite analysis/run_XXX).")
    ap.add_argument("--overwrite", action="store_true", help="Rewrite outputs that already exist.")
    args = ap.parse_args()

    run_id = args.run.replace("run_", "")
    run_name = f"run_{run_id}"

    suite_root = REPO_ROOT / "subd-model-runs" / args.suite
    run_out = suite_root / "run-outputs" / run_name
    analysis = Path(args.outdir).resolve() if args.outdir else (suite_root / "analysis" / run_name)
    analysis.mkdir(parents=True, exist_ok=True)

    times = read_pvd_times(run_out / "solution.pvd")

    def parse_steps(s: str) -> list[int]:
        s = s.strip()
        if not s:
            return []
        if ":" in s:
            parts = [int(p) for p in s.split(":")]
            step = parts[2] if len(parts) == 3 else 1
            return list(range(parts[0], parts[1] + 1, step))
        return [int(p) for p in s.split(",")]

    tprof_steps = parse_steps(args.tprof_steps)
    dt_pairs: list[tuple[int, int]] = []
    if args.dt_pairs.strip():
        for chunk in args.dt_pairs.split(";"):
            a, b = chunk.split(",")
            dt_pairs.append((int(a), int(b)))

    d = args.depths.strip()
    if ":" in d:
        parts = [float(p) for p in d.split(":")]
        step = parts[2] if len(parts) == 3 else 1.0
        depths = np.arange(parts[0], parts[1] + 0.5 * step, step)
    else:
        depths = np.array([float(p) for p in d.split(",")])

    # ---- Tprof files (linear gridding) --------------------------------
    for k in tprof_steps:
        out = analysis / f"Tprof_{k}.csv"
        if out.exists() and not args.overwrite:
            print(f"[skip] {out}")
            continue
        f = analysis / f"t{k}.csv"
        if not f.exists():
            print(f"[MISS] {f}")
            continue
        X, Z, GT, GC = load_grid(f, "linear", args.grid_res_km, args.grid_depth_max_km)
        _, _, _, T_samp = interface_picks(
            X, Z, GT, GC, depths, args.c_thresh, args.x_min_km,
            args.smooth_window_km, args.smooth_polyorder,
        )
        time_myr = times[k] / 1e6
        pd.DataFrame(
            {
                "time_Myr": [time_myr] * len(depths),
                "depth_km": depths,
                "T_C": [_r3(v) for v in T_samp],
            }
        ).to_csv(out, index=False)
        print(f"[Tprof] wrote {out}")

    # ---- DT pair files (nearest gridding, legacy default) -------------
    cache: dict[int, tuple] = {}
    for a, b in dt_pairs:
        out = analysis / f"DT_{a}_{b}.csv"
        if out.exists() and not args.overwrite:
            print(f"[skip] {out}")
            continue
        ok = True
        for k in (a, b):
            if k in cache:
                continue
            f = analysis / f"t{k}.csv"
            if not f.exists():
                print(f"[MISS] {f}")
                ok = False
                break
            X, Z, GT, GC = load_grid(f, "nearest", args.grid_res_km, args.grid_depth_max_km)
            cache[k] = interface_picks(
                X, Z, GT, GC, depths, args.c_thresh, args.x_min_km,
                args.smooth_window_km, args.smooth_polyorder,
            )
        if not ok:
            continue

        x1_raw, x1_s, T1_raw, T1 = cache[a]
        x2_raw, x2_s, T2_raw, T2 = cache[b]
        dt_Myr = (times[b] - times[a]) / 1e6
        dT = T2 - T1
        dTdt = dT / dt_Myr if abs(dt_Myr) > 1e-12 else np.full_like(dT, np.nan)

        rows = []
        for i, dep in enumerate(depths):
            rows.append(
                dict(
                    depth_km=float(np.round(dep, 3)),
                    x1_km_raw=_r3(x1_raw[i]),
                    x2_km_raw=_r3(x2_raw[i]),
                    T1_C_raw=_r3(T1_raw[i]),
                    T2_C_raw=_r3(T2_raw[i]),
                    x1_km_smooth=_r3(x1_s[i]),
                    x2_km_smooth=_r3(x2_s[i]),
                    T1_C=_r3(T1[i]),
                    T2_C=_r3(T2[i]),
                    dT_C=_r3(dT[i]),
                    dt_Myr=float(np.round(dt_Myr, 9)),
                    dTdt_C_per_Myr=float(np.round(dTdt[i], 9)) if np.isfinite(dTdt[i]) else np.nan,
                )
            )
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"[DT] wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

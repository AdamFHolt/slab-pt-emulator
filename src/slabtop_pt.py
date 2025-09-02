#!/usr/bin/env python3

"""
Extract slab-top P–T path (only) from ParaView-exported CSVs.

Usage:
  python slabtop_pt_min.py \
      --template 'csv_inputs/GRL_0_0_0_HR3/part_{}.csv' \
      --tmin 0 --tmax 85 \
      --outdir outputs \
      --grid-res-m 1000 \
      --xy-filter 59 \
      --pt-filter 101

Requirements: numpy, pandas, matplotlib, scipy
"""

import argparse
import numpy as np, os, sys
import pandas as pd
from scipy import interpolate
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")  # no GUI
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

# -------------------- helpers --------------------
def odd_window(n, maxlen):
    """Return an odd window <= maxlen and >=3 (or 1 if too short)."""
    if maxlen < 3:
        return 1
    w = min(n, maxlen)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        w = 3
    return w

def make_grid(xmin, xmax, ymin, ymax, res_m):
    nx = max(2, int(np.ceil((xmax - xmin) / res_m)) + 1)
    ny = max(2, int(np.ceil((ymax - ymin) / res_m)) + 1)
    xg = np.linspace(xmin, xmax, nx)
    yg = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xg, yg)
    return X, Y

def interp_to_grid(x, y, values, X, Y, method="nearest"):
    return interpolate.griddata((x, y), values, (X, Y), method=method)

def extract_slabtop_from_mask_tracked(
    X_km, Z_km, mask_grid, *,
    surface_clip_km=0.2,        # keep it close to the surface
    min_span_km=30.0,
    min_thickness_km=2.0,       # a bit more permissive
    max_step_km=15.0,           # allow bigger changes across the trench
    med_win_cols=11,            # stronger de-noising
    thr=0.5,
    bridge_cols=8               # <= this many empty columns will be bridged
    ):
    """
    Track the *top* of ocrust with continuity:
      • per-column pick = shallowest rising-edge (0→1) crossing
      • bridge up to `bridge_cols` missing columns by carrying `prev_z`
      • allow larger re-attachment jump proportional to bridged width
    """
    ny, nx = mask_grid.shape
    xs, zs = [], []

    # ---- collect candidates per column (shallowest rising edge) ----
    candidates = []
    for j in range(nx):
        order = np.argsort(Z_km[:, j])             # shallow→deep
        zcol  = Z_km[order, j]
        vcol  = np.clip(mask_grid[order, j], 0.0, 1.0)

        s = (vcol >= thr).astype(np.int8)
        rises = np.flatnonzero(np.diff(s) == 1) + 1

        dz   = float(np.median(np.diff(zcol))) if zcol.size > 1 else 1.0
        need = max(1, int(np.ceil(min_thickness_km / max(dz, 1e-6))))

        picks = []
        for i0 in rises:
            i1 = min(i0 + need - 1, len(s) - 1)
            if np.all(s[i0:i1+1] == 1):
                z0 = float(zcol[i0])
                if z0 >= surface_clip_km:
                    x0 = float(X_km[order[i0], j])
                    picks.append((x0, z0))

        # whole column already >=thr from the top (no explicit rise)
        if not rises.size and s.any():
            i0 = int(np.argmax(s))
            i1 = min(i0 + need - 1, len(s) - 1)
            if np.all(s[i0:i1+1] == 1):
                z0 = float(zcol[i0])
                if z0 >= surface_clip_km:
                    x0 = float(X_km[order[i0], j])
                    picks.append((x0, z0))

        candidates.append(picks if picks else None)

    # ---- seed at first column with a candidate (shallowest) ----
    j0 = next((j for j, p in enumerate(candidates) if p), None)
    if j0 is None:
        raise RuntimeError("Slab-top not found (no robust crossings).")
    x0, z0 = sorted(candidates[j0], key=lambda t: t[1])[0]
    xs, zs = [x0], [z0]

    # ---- forward track with gap bridging ----
    prev_z = z0
    gap = 0
    for j in range(j0 + 1, nx):
        picks = candidates[j]
        if not picks:
            # bridge small gaps by carrying depth
            if gap < bridge_cols:
                gap += 1
                xs.append(float(X_km[0, j])); zs.append(prev_z)
                continue
            else:
                xs.append(float(X_km[0, j])); zs.append(np.nan)
                gap = 0
                continue
        # allow jump proportional to bridged width we just crossed
        jump_allow = max_step_km * max(1, gap if gap else 1)
        gap = 0
        picks_sorted = sorted(picks, key=lambda t: abs(t[1] - prev_z))
        chosen = next((p for p in picks_sorted if abs(p[1] - prev_z) <= jump_allow), None)
        if chosen is None:
            chosen = sorted(picks, key=lambda t: t[1])[0]
        xs.append(chosen[0]); zs.append(chosen[1]); prev_z = chosen[1]

    # ---- backward track with gap bridging ----
    prev_z = z0
    xs_back, zs_back = [], []
    gap = 0
    for j in range(j0 - 1, -1, -1):
        picks = candidates[j]
        if not picks:
            if gap < bridge_cols:
                gap += 1
                xs_back.append(float(X_km[0, j])); zs_back.append(prev_z)
                continue
            else:
                xs_back.append(float(X_km[0, j])); zs_back.append(np.nan)
                gap = 0
                continue
        jump_allow = max_step_km * max(1, gap if gap else 1)
        gap = 0
        picks_sorted = sorted(picks, key=lambda t: abs(t[1] - prev_z))
        chosen = next((p for p in picks_sorted if abs(p[1] - prev_z) <= jump_allow), None)
        if chosen is None:
            chosen = sorted(picks, key=lambda t: t[1])[0]
        xs_back.append(chosen[0]); zs_back.append(chosen[1]); prev_z = chosen[1]

    # ---- stitch, denoise, keep longest finite run ----
    x = np.array(xs_back[::-1] + xs)
    z = np.array(zs_back[::-1] + zs)

    if x.size >= med_win_cols:
        z = median_filter(z, size=med_win_cols, mode="nearest")

    x, z = _longest_finite_run(x, z)
    order = np.argsort(x); x, z = x[order], z[order]

    if x.size < 5 or (x.max() - x.min()) < min_span_km:
        raise RuntimeError("Slab-top too short after tracking.")
    return np.column_stack([x, z])

def _longest_finite_run(x, z):
    good = np.isfinite(x) & np.isfinite(z)
    if not good.any():
        raise RuntimeError("No finite slab-top samples after tracking.")
    idx = np.where(good)[0]
    # split into contiguous runs
    cuts = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, cuts)
    best = max(runs, key=len)
    return x[best], z[best]

def smooth_xy(arr, win):
    """Savitzky–Golay smooth in place for XY columns (N,2)."""
    if arr.shape[0] < 5:
        return arr
    w = odd_window(win, arr.shape[0])
    x_s, y_s = savgol_filter((arr[:,0], arr[:,1]), w, 3)
    out = arr.copy()
    out[:,0] = x_s
    out[:,1] = y_s
    return out

def trim_after_jumps(x, z, *, jump_km=20.0, slope_max=3.0, min_keep=25):
    """
    Return (x_cut, z_cut). Cut at the first big step in depth or slope.
    - jump_km: absolute |Δz| threshold between adjacent samples (km)
    - slope_max: |Δz/Δx| threshold (km/km)
    """
    if x.size < 3:
        return x, z
    dz = np.diff(z)
    dx = np.maximum(np.diff(x), 1e-9)
    bad = (np.abs(dz) > jump_km) | (np.abs(dz/dx) > slope_max)
    if not np.any(bad):
        return x, z
    i = int(np.argmax(bad))              # first offending step
    i_end = max(i, min_keep - 1)         # keep at least min_keep points
    return x[:i_end+1], z[:i_end+1]


def process_timestep(csv_path, grid_res_m, xy_filter, pt_filter,
                     *, cut_jump_km, slope_max, min_keep, pt_xmin_km):
    """
    Read one timestep CSV and return a DataFrame with slab-top P–T path.
    Output columns: x_km, depth_km, T_C, P_GPa
    """
    df = pd.read_csv(csv_path)

    # raw fields
    x   = df['Points:0'].to_numpy()          # meters
    y   = df['Points:1'].to_numpy()          # meters (0..ymax upward)
    tK  = df['T'].to_numpy()                 # Kelvin
    pPa = df['p'].to_numpy()                 # Pascals
    msk = df['ocrust'].to_numpy()            # 0..1

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # grid (meters)
    X, Y = make_grid(xmin, xmax, ymin, ymax, grid_res_m)

    # interpolate mask with 'linear' so a 0.5 contour exists between cells
    mask_grid = interp_to_grid(x, y, msk, X, Y, method="linear")
    mask_grid = np.clip(mask_grid, 0.0, 1.0)

    # convert to plotting coords (km): depth = (ymax - y)
    X_km = X / 1e3
    Z_km = (ymax - Y) / 1e3

    # pick the slab top
    slab_top = extract_slabtop_from_mask_tracked(
        X_km, Z_km, mask_grid,
        surface_clip_km=0.2,  
        min_span_km=30.0,
        min_thickness_km=2.0,
        max_step_km=15.0,
        med_win_cols=11,
        bridge_cols=8
    )
    
    slab_top = smooth_xy(slab_top, xy_filter)

    # existing path & (optional) depth-only smoothing
    x_s = slab_top[:,0]
    z_s = slab_top[:,1]
    wz  = odd_window(31, len(z_s))
    if wz >= 3:
        z_s = savgol_filter(z_s, wz, 3)

    # CUT the tail with jumps
    x_s, z_s = trim_after_jumps(
        x_s, z_s,
        jump_km=cut_jump_km,
        slope_max=slope_max,
        min_keep=min_keep
    )

    # rebuild slab_top and continue
    slab_top = np.column_stack([x_s, z_s])
    if slab_top.shape[0] < 5:
        raise RuntimeError("Too few points after cutting jumpy tail.")

    # final safety: drop any residual NaN rows
    finite_rows = np.isfinite(slab_top).all(axis=1)
    slab_top = slab_top[finite_rows]
    if slab_top.shape[0] < 5:
        raise RuntimeError("Too few finite slab-top samples for interpolation.")

    # keep only points to the right of the threshold
    keep = slab_top[:, 0] >= pt_xmin_km
    slab_top = slab_top[keep]
    if slab_top.shape[0] < 5:
        raise RuntimeError("Too few points after x-threshold filtering.")

    # sample T and P along that curve (nearest is robust)
    pts_xkm = x / 1e3
    pts_zkm = (ymax - y) / 1e3
    T_C  = interpolate.griddata((pts_xkm, pts_zkm), tK - 273.15,
                                (slab_top[:,0], slab_top[:,1]), method="nearest")
    P_Pa = interpolate.griddata((pts_xkm, pts_zkm), pPa,
                                (slab_top[:,0], slab_top[:,1]), method="nearest")

    # smooth P–T if enough samples
    if slab_top.shape[0] >= 5:
        w_pt = odd_window(pt_filter, slab_top.shape[0])
        T_C  = savgol_filter(T_C,  w_pt, 3)
        P_Pa = savgol_filter(P_Pa, w_pt, 3)

    out = pd.DataFrame({
        "x_km": slab_top[:,0],
        "depth_km": slab_top[:,1],
        "T_C": T_C,
        "P_GPa": P_Pa / 1e9
    }).dropna()

    if out.empty:
        raise RuntimeError("No finite P–T samples on slab top.")
    return out


# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True,
                    help="CSV template with {} for timestep (e.g. 'csv_inputs/run_005/part_{}.csv')")
    ap.add_argument("--tmin", type=int, required=True)
    ap.add_argument("--tmax", type=int, required=True)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--grid-res-m", type=float, default=1000.0,
                    help="Interpolation grid spacing in meters (default 1000)")
    ap.add_argument("--xy-filter", type=int, default=11,
                    help="Savitzky–Golay window for XY smoothing (odd, auto-clamped)")
    ap.add_argument("--pt-filter", type=int, default=21,
                    help="Savitzky–Golay window for P–T smoothing (odd, auto-clamped)")
    ap.add_argument("--cut-jump-km", type=float, default=20.0,
                    help="Cut path at first |Δz| jump larger than this (km).")
    ap.add_argument("--slope-max", type=float, default=3.0,
                    help="Also cut if |Δz/Δx| exceeds this (km/km).")
    ap.add_argument("--min-keep", type=int, default=25,
                    help="Keep at least this many points before cutting.")
    ap.add_argument("--pt-xmin-km", type=float, default=1800.0,
                    help="Only keep slab-top points with x >= this (km).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    for t in range(args.tmin, args.tmax + 1):
        csv_path = args.template.format(t)
        if not os.path.exists(csv_path):
            print(f"[pt] [skip] {csv_path} not found")
            continue
        out_path = os.path.join(args.outdir, f"slabtop_PT_{t}.csv")
        if os.path.exists(out_path):
            print(f"[pt] [keep] {out_path} exists")
            continue
        try:
            df_out = process_timestep(
                csv_path, args.grid_res_m, args.xy_filter, args.pt_filter,
                cut_jump_km=args.cut_jump_km, slope_max=args.slope_max,
                min_keep=args.min_keep, pt_xmin_km=args.pt_xmin_km
            )
            df_out.to_csv(out_path, index=False)
            print(f"[pt] wrote {out_path}  (n={len(df_out)})")
        except Exception as e:
            print(f"[pt] [fail] t={t}: {e}")

if __name__ == "__main__":
    main()

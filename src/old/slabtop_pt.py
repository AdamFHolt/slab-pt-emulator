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
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

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


def extract_slabtop_from_contour(X_km, Z_km, mask_grid, *,
                                 min_span_km=30.0,
                                 max_depth_km=60.0):
    """
    Pick the slab top from the 0.5 contour of a binary-ish 'ocrust' mask.

    Steps:
      1) contour at 0.5
      2) keep segments with decent x-span and shallow depths
      3) sort by x and, where multi-valued, keep the *shallowest* z per x
      4) light Savitzky–Golay smoothing of z only
    Returns: (N,2) array of [x_km, depth_km] sorted by x.
    """
    # 0.5 contour
    fig, ax = plt.subplots(figsize=(3, 2))
    cs = ax.contour(X_km, Z_km, mask_grid, levels=[0.5])
    plt.close(fig)

    segs = cs.allsegs[0] if cs.allsegs and cs.allsegs[0] else []
    if not segs:
        raise RuntimeError("No 0.5 contour found for ocrust.")

    # candidate = (shallow score, negative x-span for tie-break, segment)
    cands = []
    for seg in segs:
        x = seg[:, 0]; z = seg[:, 1]
        span = x.max() - x.min()
        if span < min_span_km:
            continue
        shallow_score = np.percentile(z, 25)  # prefer shallower segments
        if shallow_score <= max_depth_km:
            cands.append((shallow_score, -span, seg))

    if not cands:
        raise RuntimeError("No suitable near-surface contour segment found.")

    seg = sorted(cands, key=lambda t: (t[0], t[1]))[0][2]
    # monotonic in x + collapse folds by taking top-most z at each x bin
    order = np.argsort(seg[:, 0])
    x = seg[order, 0]; z = seg[order, 1]

    # bin by the grid spacing in x
    dx = np.median(np.diff(X_km[0, :])) if X_km.shape[1] > 1 else 1.0
    bins = np.floor((x - x.min()) / max(dx, 1e-6)).astype(int)

    x_u, z_u = [], []
    for b in np.unique(bins):
        m = (bins == b)
        if not np.any(m):
            continue
        x_u.append(np.mean(x[m]))
        z_u.append(np.min(z[m]))  # keep the *shallowest* (smallest depth)
    x_u = np.asarray(x_u); z_u = np.asarray(z_u)

    # gentle depth-only smoothing
    w = odd_window(31, len(z_u))
    if w >= 3:
        z_u = savgol_filter(z_u, w, 3)

    if x_u.size < 5:
        raise RuntimeError("Contour-derived slab top too short.")
    return np.column_stack([x_u, z_u])

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

def process_timestep(csv_path, grid_res_m, xy_filter, pt_filter, *, pt_xmin_km):
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

    # grid (meters) for contouring the mask
    X, Y = make_grid(xmin, xmax, ymin, ymax, grid_res_m)
    mask_grid = interp_to_grid(x, y, msk, X, Y, method="linear")
    mask_grid = np.clip(mask_grid, 0.0, 1.0)

    # plotting coords (km): depth = (ymax - y)
    X_km = X / 1e3
    Z_km = (ymax - Y) / 1e3

    # slab top from the 0.5 contour
    slab_top = extract_slabtop_from_contour(
        X_km, Z_km, mask_grid, min_span_km=30.0, max_depth_km=60.0
    )

    # optional smoothing along the curve
    slab_top = smooth_xy(slab_top, xy_filter)
    x_s = slab_top[:, 0]
    z_s = slab_top[:, 1]
    wz  = odd_window(31, len(z_s))
    if wz >= 3:
        z_s = savgol_filter(z_s, wz, 3)
    slab_top = np.column_stack([x_s, z_s])

    # keep only points to the right of the threshold
    slab_top = slab_top[slab_top[:, 0] >= pt_xmin_km]
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
        wp = odd_window(pt_filter, slab_top.shape[0])
        if wp >= 3:
            T_C  = savgol_filter(T_C,  wp, 3)
            P_Pa = savgol_filter(P_Pa, wp, 3)

    out = pd.DataFrame({
        "x_km": slab_top[:,0],
        "depth_km": slab_top[:,1],
        "T_C": T_C,
        "P_GPa": P_Pa / 1e9
    }).dropna()

    if out.empty:
        raise RuntimeError("No finite P–T samples on slab top.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True,
                    help="CSV template with {} for timestep (e.g. 'csv_inputs/run_005/part_{}.csv')")
    ap.add_argument("--tmin", type=int, required=True)
    ap.add_argument("--tmax", type=int, required=True)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--grid-res-m", type=float, default=1000.0)
    ap.add_argument("--xy-filter", type=int, default=11)
    ap.add_argument("--pt-filter", type=int, default=21)
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
                pt_xmin_km=args.pt_xmin_km
            )
            df_out.to_csv(out_path, index=False)
            print(f"[pt] wrote {out_path}  (n={len(df_out)})")
        except Exception as e:
            print(f"[pt] [fail] t={t}: {e}")
            had_fail = True

 
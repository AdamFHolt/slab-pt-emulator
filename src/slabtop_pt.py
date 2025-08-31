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

def extract_slabtop_from_mask(X_km, Z_km, mask_grid, depth_thresh_km=5.0):
    """
    From a binary-ish mask (ocrust==1 in slab), get the slab-top polyline.
    Works by contouring at 0.5 and picking the longest path, then selecting
    the upper branch (shallower y/depth).
    Returns Nx2 array of [x_km, depth_km] ordered from trenchward to downdip.
    """
    # Contour at 0.5
    fig, ax = plt.subplots(figsize=(4, 3))
    cs = ax.contour(X_km, Z_km, mask_grid, levels=[0.5])
    plt.close(fig)

    # cs.allsegs is a list of list of (N,2) arrays
    if not cs.allsegs or not cs.allsegs[0]:
        raise RuntimeError("No contour at 0.5 found for slab mask.")

    paths = cs.allsegs[0]
    lens = [seg.shape[0] for seg in paths]
    pts = paths[np.argmax(lens)]

    # Drop the very near-surface wiggles by thresholding on depth
    if np.any(pts[:,1] > depth_thresh_km):
        x_thresh = (pts[pts[:,1] > depth_thresh_km]).min(0)[0]
        slab = pts[pts[:,0] > x_thresh]
    else:
        slab = pts

    if slab.shape[0] < 8:
        raise RuntimeError("Slab contour too short after thresholding.")

    # Split into two branches around the most extreme depth (tip vs updip end)
    itip = np.argmax(slab[:,1])   # deepest
    itop = np.argmin(slab[:,1])   # shallowest
    mid = len(slab) // 2
    iset = itop if abs(itop - mid) < abs(itip - mid) else itip

    # Decide which half is the slab-top (shallower branch)
    left  = slab[:iset, :]
    right = slab[iset:, :][::-1, :]
    # Compare mean depths; shallower is slab top
    slab_top = left if left[:,1].mean() < right[:,1].mean() else right

    # Enforce monotonic-ish along-arc ordering (left-to-right by x)
    slab_top = slab_top[np.argsort(slab_top[:,0])]
    return slab_top

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

# -------------------- main per-timestep --------------------
def process_timestep(csv_path, grid_res_m, xy_filter, pt_filter):
    """
    Read one timestep CSV and return a DataFrame with slab-top P–T path.
    Output columns: x_km, depth_km, T_C, P_GPa
    """
    df = pd.read_csv(csv_path)
    # Raw coords
    x = df['Points:0'].to_numpy()            # meters
    y = df['Points:1'].to_numpy()            # meters (0..ymax upward)
    tK = df['T'].to_numpy()                  # Kelvin
    pPa = df['p'].to_numpy()                 # Pascals
    mask = df['ocrust'].to_numpy()           # 0..1

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # Grid in model coords (meters), then convert to km for contouring
    X, Y = make_grid(xmin, xmax, ymin, ymax, grid_res_m)
    mask_grid = interp_to_grid(x, y, mask, X, Y, method="nearest")

    # Convert to plotting coords: x_km, depth_km = (ymax - y)/1e3
    X_km = X / 1e3
    Z_km = (ymax - Y) / 1e3

    # Slab top polyline (x_km, depth_km)
    slab_top = extract_slabtop_from_mask(X_km, Z_km, mask_grid, depth_thresh_km=5.0)
    slab_top = smooth_xy(slab_top, xy_filter)

    # Interpolate T and p at slab-top points
    # Build data in (x_km, depth_km) to match slab_top coords
    pts_xkm = x / 1e3
    pts_zkm = (ymax - y) / 1e3

    T_C = interpolate.griddata(
        (pts_xkm, pts_zkm), tK - 273.15, (slab_top[:,0], slab_top[:,1]),
        method="nearest"
    )
    P_Pa = interpolate.griddata(
        (pts_xkm, pts_zkm), pPa, (slab_top[:,0], slab_top[:,1]),
        method="nearest"
    )

    # Smooth P–T (independently), auto window
    if slab_top.shape[0] >= 5:
        w_pt = odd_window(pt_filter, slab_top.shape[0])
        T_C  = savgol_filter(T_C, w_pt, 3)
        P_Pa = savgol_filter(P_Pa, w_pt, 3)

    out = pd.DataFrame({
        "x_km": slab_top[:,0],
        "depth_km": slab_top[:,1],
        "T_C": T_C,
        "P_GPa": P_Pa / 1e9
    })
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
    ap.add_argument("--xy-filter", type=int, default=59,
                    help="Savitzky–Golay window for XY smoothing (odd, auto-clamped)")
    ap.add_argument("--pt-filter", type=int, default=101,
                    help="Savitzky–Golay window for P–T smoothing (odd, auto-clamped)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    for t in range(args.tmin, args.tmax + 1):
        csv_path = args.template.format(t)
        if not os.path.exists(csv_path):
            print(f"[skip] {csv_path} not found")
            continue
        out_path = os.path.join(args.outdir, f"slabtop_PT_{t}.csv")
        if os.path.exists(out_path):
            print(f"[keep] {out_path} exists")
            continue
        try:
            df_out = process_timestep(csv_path, args.grid_res_m, args.xy_filter, args.pt_filter)
            df_out.to_csv(out_path, index=False)
            print(f"[ok] wrote {out_path}  (n={len(df_out)})")
        except Exception as e:
            print(f"[fail] t={t}: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata
from scipy.signal import savgol_filter


def read_field_csv(path: Path):
    df = pd.read_csv(path)
    x = df[[c for c in df.columns if c.lower().startswith("points:0")][0]].to_numpy()
    y = df[[c for c in df.columns if c.lower().startswith("points:1")][0]].to_numpy()
    tK = df["T"].to_numpy()
    c = df[[c for c in df.columns if c.lower().startswith("ocrust")][0]].to_numpy()
    return x, y, tK - 273.15, c


def grid_field(x_m, y_m, T_C, C, grid_res_km, xmin_km, xmax_km, ymax_km, interp):
    x_km = x_m / 1e3
    z_km = ymax_km - (y_m / 1e3)
    dx = float(grid_res_km)
    X, Z = np.meshgrid(
        np.arange(xmin_km, xmax_km + dx, dx),
        np.arange(0, ymax_km + dx, dx),
    )
    GT = griddata((x_km, z_km), T_C, (X, Z), method=interp)
    GC = griddata((x_km, z_km), C, (X, Z), method=interp)
    return X, Z, GT, GC


def row_index_for_depth(Z, depth_km: float) -> int:
    return int(np.argmin(np.abs(Z[:, 0] - depth_km)))


def pick_rightmost_x_at_depth(X, Z, GC, GT, depth_km, c_thresh, x_min_km):
    """
    Return (x_km, T_C) at the rightmost x where:
      - GC >= c_thresh
      - GT finite
      - X >= x_min_km
    Uses nearest row to depth_km.
    """
    i = row_index_for_depth(Z, depth_km)
    row_ok = np.isfinite(GC[i, :])
    cols = np.where(
        row_ok
        & np.isfinite(GT[i, :])
        & (GC[i, :] >= c_thresh)
        & (X[i, :] >= x_min_km)
    )[0]
    if cols.size == 0:
        return np.nan, np.nan
    j = int(cols.max())
    return float(X[i, j]), float(GT[i, j])


def build_xraw_profile(X, Z, GC, GT, depths_km, c_thresh, x_min_km):
    """
    For each requested depth, pick x_raw(depth) as the rightmost interface location.
    Returns arrays: depth_used, x_raw
    """
    depths_km = np.asarray(depths_km, dtype=float)
    x_raw = np.full_like(depths_km, np.nan, dtype=float)

    for k, d in enumerate(depths_km):
        xk, _ = pick_rightmost_x_at_depth(X, Z, GC, GT, d, c_thresh, x_min_km)
        x_raw[k] = xk

    return depths_km, x_raw


def _interp_nans_1d(x, y):
    """Fill NaNs in y by linear interpolation in x (only within finite span)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return y.copy()

    y2 = y.copy()
    # interp only where y is nan and within min/max of finite x
    x_f = x[m]
    y_f = y[m]
    m_nan = ~np.isfinite(y2)
    if np.any(m_nan):
        within = (x >= x_f.min()) & (x <= x_f.max())
        fill = m_nan & within
        if np.any(fill):
            y2[fill] = np.interp(x[fill], x_f, y_f)
    return y2


def smooth_x_profile(depths_km, x_raw, window_km: float, polyorder: int):
    """
    Smooth x(depth) using Savitzky–Golay after filling NaNs by interpolation.
    - window_km is converted to a window in points based on median depth spacing.
    - window is forced to be odd and >= polyorder+2.
    """
    depths_km = np.asarray(depths_km, float)
    x_raw = np.asarray(x_raw, float)

    # Fill NaNs first (so the filter has something to work with)
    x_fill = _interp_nans_1d(depths_km, x_raw)

    # Need at least a few points
    finite = np.isfinite(x_fill)
    if finite.sum() < (polyorder + 2):
        return x_raw.copy()  # not enough info

    # Estimate spacing
    dd = np.diff(depths_km)
    dd = dd[np.isfinite(dd)]
    if dd.size == 0:
        return x_raw.copy()

    dz = float(np.median(np.abs(dd)))
    if dz <= 0:
        return x_raw.copy()

    # Convert km to points
    win_pts = int(np.round(window_km / dz))
    win_pts = max(win_pts, polyorder + 2)
    if win_pts % 2 == 0:
        win_pts += 1

    # If window bigger than array, shrink
    if win_pts > len(depths_km):
        win_pts = len(depths_km) if (len(depths_km) % 2 == 1) else (len(depths_km) - 1)
        win_pts = max(win_pts, polyorder + 2)
        if win_pts < (polyorder + 2):
            return x_raw.copy()

    x_smooth = savgol_filter(x_fill, window_length=win_pts, polyorder=polyorder, mode="interp")

    # Preserve original NaN gaps if you want (I prefer keeping them filled for continuity),
    # but for safety we’ll keep NaNs where x_raw was NaN AND not bracketed by finite points.
    # Here: keep x_smooth everywhere (already limited by filled region).
    return x_smooth


def sample_T_at_x_depth(X, Z, GT, depth_km, x_km):
    """
    Sample GT at (x_km, depth_km) using 1D interpolation along the depth row.
    Returns NaN if row has insufficient finite values or x outside finite range.
    """
    if not np.isfinite(x_km):
        return np.nan

    i = row_index_for_depth(Z, depth_km)
    xrow = X[i, :]
    trow = GT[i, :]

    m = np.isfinite(xrow) & np.isfinite(trow)
    if m.sum() < 2:
        return np.nan

    x_f = xrow[m]
    t_f = trow[m]

    # Ensure monotonic x for interp
    order = np.argsort(x_f)
    x_f = x_f[order]
    t_f = t_f[order]

    if x_km < x_f.min() or x_km > x_f.max():
        return np.nan

    return float(np.interp(x_km, x_f, t_f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)  # e.g. ".../analysis/run_010/t{}.csv"
    ap.add_argument("--t1", type=int, required=True)
    ap.add_argument("--t2", type=int, required=True)
    ap.add_argument("--t1-yr", type=float, required=True)
    ap.add_argument("--t2-yr", type=float, required=True)
    ap.add_argument("--depths-km", type=float, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid-res-km", type=float, default=1.0)
    ap.add_argument("--c-thresh", type=float, default=0.5)
    ap.add_argument("--x-min-km", type=float, default=0.0)
    ap.add_argument("--interp", choices=["nearest", "linear"], default="nearest")

    # NEW: smooth the picked interface x(depth)
    ap.add_argument("--smooth-x", action="store_true",
                    help="If set, smooth x(depth) picks (per timestep) before sampling T.")
    ap.add_argument("--smooth-window-km", type=float, default=11.0,
                    help="Savgol smoothing window in km (converted to points from depth spacing).")
    ap.add_argument("--smooth-polyorder", type=int, default=2,
                    help="Savgol polynomial order (usually 2).")

    args = ap.parse_args()

    f1 = Path(args.template.format(args.t1))
    f2 = Path(args.template.format(args.t2))
    if not f1.exists() or not f2.exists():
        raise SystemExit("input CSV(s) missing")

    depths = np.asarray(args.depths_km, float)

    # Infer bounds from file 1 (safe; both timesteps share box)
    x1, y1, T1, C1 = read_field_csv(f1)
    xmin_km, xmax_km = x1.min() / 1e3, x1.max() / 1e3
    ymax_km = y1.max() / 1e3

    # Grid both timesteps
    X, Z, GT1, GC1 = grid_field(
        x1, y1, T1, C1, args.grid_res_km, xmin_km, xmax_km, ymax_km, args.interp
    )
    x2, y2, T2, C2 = read_field_csv(f2)
    _, _, GT2, GC2 = grid_field(
        x2, y2, T2, C2, args.grid_res_km, xmin_km, xmax_km, ymax_km, args.interp
    )

    # Time difference (handle t1 == t2 safely)
    dt_yr = args.t2_yr - args.t1_yr
    if abs(dt_yr) < 1e-12:
        dt_Myr = 0.0
        inv_dt_Myr = np.nan
    else:
        dt_Myr = dt_yr / 1e6
        inv_dt_Myr = 1.0 / dt_Myr

    # Build raw x(depth) picks for each timestep
    depth_used, x1_raw = build_xraw_profile(X, Z, GC1, GT1, depths, args.c_thresh, args.x_min_km)
    _,          x2_raw = build_xraw_profile(X, Z, GC2, GT2, depths, args.c_thresh, args.x_min_km)

    # Smooth x(depth) if requested
    if args.smooth_x:
        x1_s = smooth_x_profile(depth_used, x1_raw, args.smooth_window_km, args.smooth_polyorder)
        x2_s = smooth_x_profile(depth_used, x2_raw, args.smooth_window_km, args.smooth_polyorder)
    else:
        x1_s = x1_raw.copy()
        x2_s = x2_raw.copy()

    # Sample T at raw picks (for diagnostics) and at smoothed picks (for final)
    T1_raw = np.array([pick_rightmost_x_at_depth(X, Z, GC1, GT1, d, args.c_thresh, args.x_min_km)[1] for d in depth_used], float)
    T2_raw = np.array([pick_rightmost_x_at_depth(X, Z, GC2, GT2, d, args.c_thresh, args.x_min_km)[1] for d in depth_used], float)

    T1_samp = np.array([sample_T_at_x_depth(X, Z, GT1, d, xk) for d, xk in zip(depth_used, x1_s)], float)
    T2_samp = np.array([sample_T_at_x_depth(X, Z, GT2, d, xk) for d, xk in zip(depth_used, x2_s)], float)

    # If not smoothing, keep the original “picked” temperature (so outputs match legacy)
    if not args.smooth_x:
        T1_samp = T1_raw.copy()
        T2_samp = T2_raw.copy()

    dT = T2_samp - T1_samp
    if np.isnan(inv_dt_Myr):
        dTdt = np.full_like(dT, np.nan, dtype=float)
    else:
        dTdt = dT * inv_dt_Myr

    rows = []
    for k, d in enumerate(depth_used):
        rows.append(dict(
            depth_km=float(np.round(d, 3)),

            # raw picks (rightmost C>=thr, snapped to grid columns)
            x1_km_raw=float(np.round(x1_raw[k], 3)) if np.isfinite(x1_raw[k]) else np.nan,
            x2_km_raw=float(np.round(x2_raw[k], 3)) if np.isfinite(x2_raw[k]) else np.nan,
            T1_C_raw=float(np.round(T1_raw[k], 3)) if np.isfinite(T1_raw[k]) else np.nan,
            T2_C_raw=float(np.round(T2_raw[k], 3)) if np.isfinite(T2_raw[k]) else np.nan,

            # smoothed x + sampled T (or same as raw if --smooth-x not set)
            x1_km_smooth=float(np.round(x1_s[k], 3)) if np.isfinite(x1_s[k]) else np.nan,
            x2_km_smooth=float(np.round(x2_s[k], 3)) if np.isfinite(x2_s[k]) else np.nan,
            T1_C=float(np.round(T1_samp[k], 3)) if np.isfinite(T1_samp[k]) else np.nan,
            T2_C=float(np.round(T2_samp[k], 3)) if np.isfinite(T2_samp[k]) else np.nan,

            dT_C=float(np.round(dT[k], 3)) if np.isfinite(dT[k]) else np.nan,
            dt_Myr=float(np.round(dt_Myr, 9)),
            dTdt_C_per_Myr=float(np.round(dTdt[k], 9)) if np.isfinite(dTdt[k]) else np.nan,
        ))

    pd.DataFrame(rows).to_csv(args.out, index=False)

if __name__ == "__main__":
    main()

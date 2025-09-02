#!/usr/bin/env python3
import argparse
import numpy as np, pandas as pd
from pathlib import Path
from scipy.interpolate import griddata

def read_field_csv(path: Path):
    df = pd.read_csv(path)
    x = df[[c for c in df.columns if c.lower().startswith("points:0")][0]].to_numpy()
    y = df[[c for c in df.columns if c.lower().startswith("points:1")][0]].to_numpy()
    tK= df["T"].to_numpy()
    c = df[[c for c in df.columns if c.lower().startswith("ocrust")][0]].to_numpy()
    return x, y, tK - 273.15, c

def grid_field(x_m, y_m, T_C, C, grid_res_km, xmin_km, xmax_km, ymax_km, interp):
    x_km = x_m / 1e3
    z_km = ymax_km - (y_m / 1e3)
    dx = float(grid_res_km)
    X, Z = np.meshgrid(np.arange(xmin_km, xmax_km + dx, dx),
                       np.arange(0,        ymax_km + dx, dx))
    GT = griddata((x_km, z_km), T_C, (X, Z), method=interp)
    GC = griddata((x_km, z_km), C,   (X, Z), method=interp)
    return X, Z, GT, GC

def pick_rightmost_x_at_depth(X, Z, GC, GT, depth_km, c_thresh, x_min_km):
    # Row nearest to target depth
    i = int(np.argmin(np.abs(Z[:,0] - depth_km)))
    row_ok = np.isfinite(GC[i,:])
    # Rightmost column with C>=thr AND finite T; obey x_min_km
    cols = np.where(row_ok & np.isfinite(GT[i,:]) & (GC[i,:] >= c_thresh) & (X[i,:] >= x_min_km))[0]
    if cols.size == 0:
        return np.nan, np.nan
    j = int(cols.max())
    return float(X[i,j]), float(GT[i,j])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)         # e.g. ".../analysis/run_010/{}.csv"
    ap.add_argument("--t1", type=int, required=True)
    ap.add_argument("--t2", type=int, required=True)
    ap.add_argument("--depths-km", type=float, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid-res-km", type=float, default=1.0)
    ap.add_argument("--c-thresh", type=float, default=0.5)
    ap.add_argument("--x-min-km", type=float, default=0.0)
    ap.add_argument("--interp", choices=["nearest","linear"], default="nearest")
    args = ap.parse_args()

    f1 = Path(args.template.format(args.t1))
    f2 = Path(args.template.format(args.t2))
    if not f1.exists() or not f2.exists():
        raise SystemExit("input CSV(s) missing")

    # Infer bounds from file 1 (safe; both timesteps share the box)
    x1, y1, T1, C1 = read_field_csv(f1)
    xmin_km, xmax_km = x1.min()/1e3, x1.max()/1e3
    ymax_km          = y1.max()/1e3

    # Grid both timesteps
    X, Z, GT1, GC1 = grid_field(x1, y1, T1, C1, args.grid_res_km, xmin_km, xmax_km, ymax_km, args.interp)
    x2, y2, T2, C2 = read_field_csv(f2)
    _, _, GT2, GC2 = grid_field(x2, y2, T2, C2, args.grid_res_km, xmin_km, xmax_km, ymax_km, args.interp)

    rows = []
    for d in args.depths_km:
        x1_km, t1_C = pick_rightmost_x_at_depth(X, Z, GC1, GT1, d, args.c_thresh, args.x_min_km)
        x2_km, t2_C = pick_rightmost_x_at_depth(X, Z, GC2, GT2, d, args.c_thresh, args.x_min_km)
        rows.append(dict(depth_km=d, x1_km=x1_km, T1_C=t1_C, x2_km=x2_km, T2_C=t2_C, dT_C=(t2_C - t1_C)))

    # Write DT output
    pd.DataFrame(rows).to_csv(args.out, index=False)

    print(f"[DT] wrote: {args.out}")


if __name__ == "__main__":
    main()

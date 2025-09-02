#!/usr/bin/env python3
"""
Make either:
  - 2 panels (field + PT), or
  - 3 panels when --field2-csv is given (field @ t1, field @ t2, PT)

Usage example:
  python3 plot_slabtop_and_field.py \
    --field-csv /path/to/2.csv \
    --field2-csv /path/to/10.csv \
    --pt1 /path/to/slabtop_PT_2.csv \
    --pt2 /path/to/slabtop_PT_10.csv \
    --out /path/to/compare_2_10.png \
    --grid-res-km 10 --xmin-km 0 --xmax-km 9000 --ymax-km 1750 --depth-max-km 700 \
    --cmap coolwarm
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

def read_field_csv(path: Path):
    
    df = pd.read_csv(path)
    xcol = next((c for c in df.columns if c.lower().startswith("points:0")), None)
    ycol = next((c for c in df.columns if c.lower().startswith("points:1")), None)
    tcol = "T" if "T" in df.columns else next((c for c in df.columns if c.lower().startswith("t")), None)
    ccol = next((c for c in df.columns if c.lower().startswith("ocrust")), None) 
    if not all([xcol, ycol, tcol, ccol]):
        raise RuntimeError(f"CSV missing required columns. Found: {df.columns.tolist()}")
    x_m = df[xcol].to_numpy()
    y_m = df[ycol].to_numpy()
    comp = df[ccol].to_numpy()
    T_C = df[tcol].to_numpy() - 273.15
    return x_m, y_m, T_C, comp

def read_pt_csv(path: Path):

    if not path or not Path(path).exists():
        return None
    df = pd.read_csv(path)

    def find_ci(*names):
        cols = {c.lower(): c for c in df.columns}
        for n in names:
            c = cols.get(n.lower())
            if c:
                return c
        return None

    # x coordinate (km)
    xc = find_ci("x_km", "x", "xkm")
    # vertical coordinate: prefer y_km, else depth_km (both are in km)
    yc = find_ci("y_km", "ykm", "depth_km", "depthkm", "z_km", "zkm")
    # temperature
    tc = find_ci("T_C", "t_c", "temp_c", "temperature_c", "temperature")

    # pressure (prefer GPa; fall back to p in Pa)
    pc_gpa = find_ci("P_GPa", "p_gpa", "pressure_gpa")
    pc_pa  = find_ci("p", "pressure")

    if tc is None or (pc_gpa is None and pc_pa is None):
        raise RuntimeError(f"P–T CSV missing T/P columns: {path} (cols: {df.columns.tolist()})")

    T_C = df[tc].to_numpy()
    if pc_gpa is not None:
        P_GPa = df[pc_gpa].to_numpy()
    else:
        P_pa = df[pc_pa].to_numpy()
        P_GPa = P_pa / 1e9

    x_km = df[xc].to_numpy() if xc else None
    y_km = df[yc].to_numpy() if yc else None

    return dict(x_km=x_km, y_km=y_km, T_C=T_C, P_GPa=P_GPa)


def grid_field(x_m, y_m, T_C, C, grid_res_km, xmin_km, xmax_km, ymax_km, interp, y_origin):

    x_km = x_m / 1e3
    depth_km = ymax_km - (y_m / 1e3)

    dx = float(grid_res_km)
    Xg, Zg = np.meshgrid(np.arange(xmin_km, xmax_km + dx, dx),
                         np.arange(0, ymax_km + dx, dx))

    GT = griddata((x_km, depth_km), T_C, (Xg, Zg), method=interp)
    GC = griddata((x_km, depth_km), C,   (Xg, Zg), method=interp)

    return Xg, Zg, GT, GC


def overlay_path(ax, pt, style, label):
    if pt and pt.get("x_km") is not None and pt.get("y_km") is not None:
        ax.plot(pt["x_km"], pt["y_km"], style, lw=1.0, color="k", alpha=1, zorder=3, label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field-csv", required=True)
    ap.add_argument("--field2-csv", required=False)
    ap.add_argument("--pt1", required=False)
    ap.add_argument("--pt2", required=False)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid-res-km", type=float, default=10.0)
    ap.add_argument("--xmin-km", type=float, default=0)
    ap.add_argument("--xmax-km", type=float, default=9000)
    ap.add_argument("--ymax-km", type=float, default=1750)
    ap.add_argument("--depth-max-km", type=float, default=None)
    ap.add_argument("--cmap", default="coolwarm")  
    ap.add_argument("--interp", choices=["nearest", "linear"], default="nearest")
    ap.add_argument("--y-origin", choices=["bottom","top"], default="bottom")
    # ap.add_argument("--show-sample", action="store_true")
    args = ap.parse_args()

    xmin_km = float(args.xmin_km)
    xmax_km = float(args.xmax_km)
    ymax_km = float(args.ymax_km)
    depth_max = float(args.depth_max_km) if args.depth_max_km else ymax_km

    # Load & grid first field
    x1, y1, T1, C1   = read_field_csv(Path(args.field_csv))
    X1, Z1, GT1, GC1 = grid_field(x1, y1, T1, C1, args.grid_res_km, xmin_km, xmax_km, ymax_km,
                            args.interp, args.y_origin)

    # second field
    x2, y2, T2, C2   = read_field_csv(Path(args.field2_csv))
    X2, Z2, GT2, GC2 = grid_field(x2, y2, T2, C2, args.grid_res_km, xmin_km, xmax_km, ymax_km,
                            args.interp, args.y_origin)

    # P–T paths
    pt1 = read_pt_csv(Path(args.pt1)) if args.pt1 else None
    pt2 = read_pt_csv(Path(args.pt2)) if args.pt2 else None

    # Shared color scale across fields
    vmin = min(np.nanmin(GT1), np.nanmin(GT2))
    vmax = max(np.nanmax(GT1), np.nanmax(GT2))

    # --- layout: stack fields (left column), PT spans right column ---
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, width_ratios=[2.1, 1.0])  # 4 rows × 2 cols

    axes_field = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0], sharex=None, sharey=None),
        fig.add_subplot(gs[2, 0], sharex=None, sharey=None),
        fig.add_subplot(gs[3, 0], sharex=None, sharey=None),
    ]

    ax_pt = fig.add_subplot(gs[:, 1])  # spans all 4 rows in  right column

    # --- field 1, T ---
    GT1m = np.ma.masked_invalid(GT1)
    im0 = axes_field[0].pcolormesh(X1, Z1, GT1m, shading="auto", cmap=args.cmap, vmin=vmin, vmax=vmax)
    axes_field[0].set_title("Temperature field (t1)")
    axes_field[0].set_ylabel("Depth (km)")
    axes_field[0].set_xlim(xmin_km, xmax_km); axes_field[0].set_ylim(0, depth_max); axes_field[0].invert_yaxis()
    overlay_path(axes_field[0], pt1, "-",  "slab-top t1")

    # --- field 1, C ---
    im1 = axes_field[1].pcolormesh(X1, Z1, GC1, shading="auto", cmap=args.cmap, vmin=0, vmax=1)
    axes_field[1].set_title("Comp. field (t1)")
    axes_field[1].set_ylabel("Depth (km)")
    axes_field[1].set_xlim(xmin_km, xmax_km); axes_field[1].set_ylim(0, depth_max); axes_field[1].invert_yaxis()
    overlay_path(axes_field[1], pt1, "-",  "slab-top t1")

    # --- field 2, T ---
    GT2m = np.ma.masked_invalid(GT2)
    im2 = axes_field[2].pcolormesh(X2, Z2, GT2m, shading="auto", cmap=args.cmap, vmin=vmin, vmax=vmax)
    axes_field[2].set_title("Temperature field (t2)")
    axes_field[2].set_ylabel("Depth (km)")
    axes_field[2].set_xlim(xmin_km, xmax_km); axes_field[2].set_ylim(0, depth_max); axes_field[2].invert_yaxis()
    overlay_path(axes_field[2], pt2, "-", "slab-top t2")

    # --- field 2, C ---
    im3 = axes_field[3].pcolormesh(X2, Z2, GC2, shading="auto", cmap=args.cmap, vmin=0, vmax=1)
    axes_field[3].set_title("Comp. field (t2)")
    axes_field[3].set_ylabel("Depth (km)")
    axes_field[3].set_xlabel("Distance (km)")
    axes_field[3].set_xlim(xmin_km, xmax_km); axes_field[3].set_ylim(0, depth_max); axes_field[3].invert_yaxis()
    overlay_path(axes_field[3], pt2, "-", "slab-top t2")

    # enforce 1:1 km aspect on the fields
    for ax in axes_field:
        ax.set_aspect('equal', adjustable='box')

    # # one shared colorbar for the left column 
    # cbar_src = im1 if args.field2_csv else im0
    # cbar = fig.colorbar(cbar_src, ax=axes_field, location="right", fraction=0.046, pad=0.02)
    # cbar.set_label("Temperature (°C)")

    # --- PT panel (right column) ---
    if pt1: ax_pt.plot(pt1["T_C"], pt1["P_GPa"], "-",  lw=2, label="t1")
    if pt2: ax_pt.plot(pt2["T_C"], pt2["P_GPa"], "--", lw=2, label="t2")
    ax_pt.set_xlabel("Temperature (°C)")
    ax_pt.set_ylabel("Pressure (GPa)")
    ax_pt.grid(True, alpha=0.3)
    ax_pt.set_ylim(0,3.5)
    if pt1 or pt2: ax_pt.legend()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"[plot][ok] wrote {out}")

if __name__ == "__main__":
    main()
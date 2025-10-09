#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['pdf.compression'] = 9     # max stream compression
mpl.rcParams['pdf.fonttype']   = 42     # TrueType fonts (Illustrator-editable)
mpl.rcParams['ps.fonttype']    = 42


def read_field_csv(path: Path):
    df = pd.read_csv(path)
    xcol = next(c for c in df.columns if c.lower().startswith("points:0"))
    ycol = next(c for c in df.columns if c.lower().startswith("points:1"))
    tcol = "T" if "T" in df.columns else next(c for c in df.columns if c.lower().startswith("t"))
    ccol = next(c for c in df.columns if c.lower().startswith("ocrust"))
    x_m = df[xcol].to_numpy()
    y_m = df[ycol].to_numpy()
    T_C = df[tcol].to_numpy() - 273.15
    C   = df[ccol].to_numpy()
    return x_m, y_m, T_C, C

def grid_field(x_m, y_m, T_C, C, grid_res_km, xmin_km, xmax_km, ymax_km, interp):
    x_km = x_m / 1e3
    z_km = ymax_km - (y_m / 1e3)
    dx = float(grid_res_km)
    Xg, Zg = np.meshgrid(np.arange(xmin_km, xmax_km + dx, dx),
                         np.arange(0,        ymax_km + dx, dx))
    GT = griddata((x_km, z_km), T_C, (Xg, Zg), method=interp)
    GC = griddata((x_km, z_km), C,   (Xg, Zg), method=interp)
    return Xg, Zg, GT, GC

def load_markers(path: str):
    if not path: return None
    p = Path(path)
    if not p.exists(): return None
    df = pd.read_csv(p)
    if not {"x1_km","x2_km","depth_km"}.issubset(df.columns): return None
    return df[["x1_km","x2_km","depth_km"]].to_numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field-csv", required=True)
    ap.add_argument("--field2-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out2", required=False)  # optional PDF output
    ap.add_argument("--markers", required=True)   
    ap.add_argument("--grid-res-km", type=float, default=10.0)
    ap.add_argument("--xmin-km", type=float, default=0)
    ap.add_argument("--xmax-km", type=float, default=9000)
    ap.add_argument("--ymax-km", type=float, default=1750)
    ap.add_argument("--depth-max-km", type=float, default=None)
    ap.add_argument("--cmap", default="coolwarm")
    ap.add_argument("--interp", choices=["nearest","linear"], default="nearest")
    ap.add_argument("--y-origin", choices=["bottom","top"], default="bottom")
    ap.add_argument("--annot", default="",help="Annotation string for the figure")

    args = ap.parse_args()

    xmin_km = float(args.xmin_km); xmax_km = float(args.xmax_km)
    ymax_km = float(args.ymax_km)
    depth_max = float(args.depth_max_km) if args.depth_max_km else ymax_km
    iso_levels = np.arange(200, 1401, 200) 

    x1,y1,T1,C1 = read_field_csv(Path(args.field_csv))
    X1,Z1,GT1,GC1 = grid_field(x1,y1,T1,C1,args.grid_res_km,xmin_km,xmax_km,ymax_km,args.interp)

    x2,y2,T2,C2 = read_field_csv(Path(args.field2_csv))
    X2,Z2,GT2,GC2 = grid_field(x2,y2,T2,C2,args.grid_res_km,xmin_km,xmax_km,ymax_km,args.interp)

    DT_markers = load_markers(args.markers)

    vmin = np.nanmin([np.nanmin(GT1), np.nanmin(GT2)])
    vmax = np.nanmax([np.nanmax(GT1), np.nanmax(GT2)])

    fig = plt.figure(figsize=(10, 12), constrained_layout=True)
    gs  = fig.add_gridspec(4, 1)
    axes = [fig.add_subplot(gs[i,0]) for i in range(4)]

    # T1
    im0 = axes[0].pcolormesh(X1, Z1, np.ma.masked_invalid(GT1), shading="auto", cmap=args.cmap, vmin=vmin, vmax=vmax)
    axes[0].contour(X1, Z1, GC1, levels=[0.5], colors="k", linewidths=1.2)
    cs_iso0 = axes[0].contour(X1, Z1, GT1, levels=iso_levels,
                                    colors="k", linestyles="--", linewidths=1, alpha=0.6)
    axes[0].clabel(cs_iso0, fmt="%d", fontsize=8, inline=True)
    axes[0].set_title("Temperature (t1)")
    cbar = fig.colorbar(im0, ax=axes[0], location="right", fraction=0.05, pad=0.02)
    cbar.set_label("Temperature (°C)")

    if args.annot:
        axes[0].text(0.4, 0.05, args.annot, transform=axes[0].transAxes,
                     fontsize=11, ha="center", va="bottom")

    # C1
    im1 = axes[1].pcolormesh(X1, Z1, GC1, shading="auto", cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("ocrust (t1)")

    # T2
    im2  = axes[2].pcolormesh(X2, Z2, np.ma.masked_invalid(GT2), shading="auto", cmap=args.cmap, vmin=vmin, vmax=vmax)
    axes[2].contour(X2, Z2, GC2, levels=[0.5], colors="k", linewidths=1.2)
    cs_iso2 = axes[2].contour(X2, Z2, GT2, levels=iso_levels,
                                    colors="k", linestyles="--", linewidths=1, alpha=0.6)
    axes[2].clabel(cs_iso2, fmt="%d", fontsize=8, inline=True)
    axes[2].set_title("Temperature (t2)")
    # C2
    im3 = axes[3].pcolormesh(X2, Z2, GC2, shading="auto", cmap="viridis", vmin=0, vmax=1)
    axes[3].set_title("ocrust (t2)")

    for im in (im0, im1, im2, im3):
        im.set_rasterized(True)

    for ax in axes:
        ax.set_xlim(xmin_km, xmax_km)
        ax.set_ylim(0, depth_max)
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        ax.set_ylabel("Depth (km)")
    axes[-1].set_xlabel("Distance (km)")

    # overlay markers as stars, t1
    axes[0].plot(DT_markers[:,0], DT_markers[:,2], marker="*", linestyle="None", markersize=15, color="k", zorder=5)
    axes[1].plot(DT_markers[:,0], DT_markers[:,2], marker="*", linestyle="None", markersize=15, color="k", zorder=5)
    # t2
    axes[2].plot(DT_markers[:,1], DT_markers[:,2], marker="*", linestyle="None", markersize=15, color="k", zorder=5)
    axes[3].plot(DT_markers[:,1], DT_markers[:,2], marker="*", linestyle="None", markersize=15, color="k", zorder=5)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"[plot]: wrote {out}")
    if args.out2:
        out2 = Path(args.out2)
        out2.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out2, dpi=250, bbox_inches="tight")
        print(f"[plot]: wrote {out2}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Pairplot-style scatter-matrix of parameters, with points colored by a response
(e.g. cooling rate dTdt_C_per_Myr) and a smoothed background field.

Background method: 2D binned mean + Gaussian smoothing (NOT griddata),
which avoids Delaunay triangle artifacts and is well-behaved for LHS projections.

Typical usage:
  python qc_pairplot_params_colored.py \
    --params ../../data/params/params-list.const-vc.csv \
    --master ../../subd-model-runs/const-vc/analysis/master_DT1-10.csv \
    --depth-km 50 \
    --y dTdt_C_per_Myr \
    --out ../../plots/qc-numerical-mods/const-vc/params-pairplot_colored_50km
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# font setup
import matplotlib as mpl
import matplotlib.font_manager as fm
font_path = "/home/holt/.local/share/fonts/MYRIADPRO-REGULAR.OTF"
myriad_pro = fm.FontProperties(fname=font_path)
mpl.rcParams['font.family'] = 'Myriad Pro'  
mpl.rcParams['font.size'] = 11.5
mpl.rcParams['axes.labelsize'] = 11.5
mpl.rcParams['axes.labelpad'] = 1.5
mpl.rcParams['xtick.labelsize'] = 9.75
mpl.rcParams['ytick.labelsize'] = 9.75
mpl.rcParams['xtick.major.pad'] = 2
mpl.rcParams['ytick.major.pad'] = 2
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['xtick.minor.size'] = 1.5
mpl.rcParams['ytick.minor.size'] = 1.5


LOG_PARAMS = {"eta_int", "eta_UM", "eps_trans", "thermal_param"}


def zero_pad_runids(n: int):
    width = max(3, len(str(n - 1)))
    return [f"{i:0{width}d}" for i in range(n)]

def nice_label(param: str) -> str:
    labels = {
        "v_conv": r"Convergence rate (cm/yr)",
        "v_conv_over_tconv": r"$v_{\rm conv}/t_{\rm conv}$ (cm/yr/Myr)",
        "t_conv": r"$t_{\rm conv}$ (Myr)",
        "age_SP": r"Age$_{\rm SP}$ (Ma)",
        "age_OP": r"Age$_{\rm OP}$ (Ma)",
        "dip_int": r"Initial dip (°)",
        "eta_int": r"$\eta_{\rm int}$ (Pa·s)",
        "eta_UM": r"$\eta_{\rm UM}$ (Pa·s)",
        "eps_trans": r"$\dot\epsilon_{\rm trans}$ (s$^{-1}$)",
        "thermal_param": r"$v\; \mathrm{age}_{\rm SP}\; \sin(\mathrm{dip})$ (km)",
        "dT_C": r"$\Delta T$ (°C)",
        "dTdt_C_per_Myr": r"$\Delta T/\Delta t$ (°C/Myr)",
    }

    # log10(...) columns
    if param.startswith("log10(") and param.endswith(")"):
        base = param[len("log10("):-1]
        base_lbl = labels.get(base, base)

        # If base label already has a math chunk like "$...$ (units)", reuse it
        if isinstance(base_lbl, str) and base_lbl.startswith("$") and "$" in base_lbl[1:]:
            j = base_lbl.find("$", 1)         # end of first math chunk
            math_inner = base_lbl[1:j]        # inside $...$
            units = base_lbl[j+1:]            # rest (e.g. " (Pa·s)")
            return rf"$\log_{{10}}\!\left({math_inner}\right)$" + units

        # Fallback: just put base inside log as text in math mode
        return rf"$\log_{{10}}\!\left({base}\right)$"

    return labels.get(param, param)

def display_name(v: str, mode: str) -> str:
    # if we log10-transformed this variable, label it as log10(...)
    if mode == "compute-log10" and v in LOG_PARAMS:
        return nice_label(f"log10({v})")
    return nice_label(v)


def compute_thermal_param(df: pd.DataFrame) -> np.ndarray:
    # v_conv * age_SP * sin(dip), units ~ km
    v = df["v_conv"].to_numpy(float) / 1e3     # cm/yr → km/yr
    age = df["age_SP"].to_numpy(float) * 1e6   # Myr → yr
    dip = np.deg2rad(df["dip_int"].to_numpy(float))
    return v * np.maximum(age, 0) * np.sin(np.clip(dip, 0, np.pi / 2))


def _bin2d_sum_count(x, y, w, nx, ny, xmin, xmax, ymin, ymax):
    # histogram2d returns shape (nx, ny) for sums/counts
    zsum, xedges, yedges = np.histogram2d(
        x, y, bins=[nx, ny],
        range=[[xmin, xmax], [ymin, ymax]],
        weights=w
    )
    zcnt, _, _ = np.histogram2d(
        x, y, bins=[nx, ny],
        range=[[xmin, xmax], [ymin, ymax]]
    )
    # centers
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    return zsum, zcnt, xc, yc


def smooth_field_binned(x, y, z, nx=180, ny=180, sigma=1.2):
    """
    Stable background: binned mean + gaussian smoothing.
    Smooth numerator and denominator separately, then divide.
    """
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[m]; y = y[m]; z = z[m]
    if len(z) < 30:
        return None

    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    if not np.isfinite([xmin, xmax, ymin, ymax]).all():
        return None
    if xmin == xmax or ymin == ymax:
        return None

    zsum, zcnt, xc, yc = _bin2d_sum_count(x, y, z, nx, ny, xmin, xmax, ymin, ymax)

    # Smooth sum and count
    zsum_s = gaussian_filter(zsum, sigma=sigma)
    zcnt_s = gaussian_filter(zcnt, sigma=sigma)

    with np.errstate(divide="ignore", invalid="ignore"):
        Z = zsum_s / zcnt_s
        Z[zcnt_s <= 1e-12] = np.nan

    # Light inpaint: fill remaining NaNs by nearest valid value (via iterative blur)
    # (keeps background spanning the full axis region without sharp edges)
    if np.any(~np.isfinite(Z)):
        Zfill = Z.copy()
        # initialize NaNs to global median
        med = np.nanmedian(Zfill)
        Zfill[~np.isfinite(Zfill)] = med if np.isfinite(med) else 0.0
        # create a validity mask and smooth it
        valid = np.isfinite(Z).astype(float)
        valid_s = gaussian_filter(valid, sigma=max(1.5, sigma * 1.2))
        Zfill_s = gaussian_filter(Zfill, sigma=max(1.5, sigma * 1.2))
        with np.errstate(divide="ignore", invalid="ignore"):
            Z2 = Zfill_s / valid_s
        Z2[valid_s <= 1e-6] = np.nan
        # wherever original Z was nan, use Z2
        Z = np.where(np.isfinite(Z), Z, Z2)

    # imshow expects array indexed [y, x]
    # histogram2d gave [xbin, ybin] so transpose
    return xc, yc, Z.T


def jitter_for_plot(a, frac=0.0, rng=None):
    """
    Optional tiny jitter for plotting only (helps with discrete params/striping).
    frac is a fraction of the data range. Set frac=0 to disable.
    """
    if frac <= 0:
        return a
    a = np.asarray(a)
    r = np.nanmax(a) - np.nanmin(a)
    if not np.isfinite(r) or r <= 0:
        return a
    rng = np.random.default_rng() if rng is None else rng
    return a + rng.normal(scale=frac * r, size=a.shape)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True)
    p.add_argument("--master", required=True)
    p.add_argument("--depth-km", type=float, default=50.0)
    p.add_argument("--y", default="dTdt_C_per_Myr")
    p.add_argument("--out", required=True)
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--mode", choices=["compute-log10", "raw"], default="compute-log10")
    p.add_argument("--vars", nargs="*", default=None)
    p.add_argument("--max-vars", type=int, default=8)
    # background controls
    p.add_argument("--nx", type=int, default=180, help="background grid bins in x")
    p.add_argument("--ny", type=int, default=180, help="background grid bins in y")
    p.add_argument("--sigma", type=float, default=1.2, help="gaussian smoothing for background (in bins)")
    p.add_argument("--bg-alpha", type=float, default=0.92, help="background opacity")
    # point controls
    p.add_argument("--pt-size", type=float, default=14.0)
    p.add_argument("--pt-alpha", type=float, default=0.95)
    p.add_argument("--jitter-frac", type=float, default=0.0, help="tiny jitter fraction for plotting (0 disables)")
    args = p.parse_args()

    params_path = Path(args.params).resolve()
    master_path = Path(args.master).resolve()
    out_prefix = Path(args.out).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df_p = pd.read_csv(params_path)
    df_p["run_id"] = zero_pad_runids(len(df_p))

    # ramped support
    if "t_conv" in df_p.columns:
        t = df_p["t_conv"].replace(0, np.nan).to_numpy(float)
        v = df_p["v_conv"].to_numpy(float)
        df_p["v_conv_over_tconv"] = v / t

    # add thermal param if possible
    if {"v_conv", "age_SP", "dip_int"}.issubset(df_p.columns):
        df_p["thermal_param"] = compute_thermal_param(df_p)

    df_m = pd.read_csv(master_path, dtype={"run_id": str})
    if "depth_km" not in df_m.columns:
        raise ValueError("Master CSV must contain a 'depth_km' column.")
    if args.y not in df_m.columns:
        raise ValueError(f"Master CSV missing requested y variable '{args.y}'.")

    df_m = df_m[np.isclose(df_m["depth_km"].to_numpy(float), float(args.depth_km))]
    if df_m.empty:
        raise ValueError(f"No rows in master match depth_km={args.depth_km}.")

    df = pd.merge(df_p, df_m[["run_id", args.y]], on="run_id", how="inner")
    if df.empty:
        raise ValueError("Merge produced empty dataframe. Check run_id formatting.")

    # Choose x-vars
    default_vars = []
    for name in [
        "v_conv", "v_conv_over_tconv", "t_conv",
        "age_SP", "age_OP", "dip_int",
        "eta_int", "eta_UM", "eps_trans",
        "thermal_param"
    ]:
        if name in df.columns:
            default_vars.append(name)

    if args.vars:
        vars_ = [v for v in args.vars if v in df.columns]
        if not vars_:
            raise ValueError("None of the requested --vars exist in the merged dataframe.")
    else:
        vars_ = default_vars

    vars_ = vars_[: args.max_vars]
    n = len(vars_)
    if n < 2:
        raise ValueError("Need at least 2 variables for a pairplot matrix.")

    # Prepare arrays (log-transform some)
    X = {}
    for v in vars_:
        x = df[v].to_numpy(float)
        if args.mode == "compute-log10" and v in LOG_PARAMS:
            x = np.where(x > 0, np.log10(x), np.nan)
        X[v] = x

    Y = df[args.y].to_numpy(float)

    # Robust color scaling
    vmin = np.nanpercentile(Y, 2)
    vmax = np.nanpercentile(Y, 98)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = np.nanmin(Y), np.nanmax(Y)

    fig, axes = plt.subplots(
        n, n,
        figsize=(2.35 * n + 2.0, 2.35 * n + 0.8),
        constrained_layout=True
    )
    cmap = plt.cm.viridis
    rng = np.random.default_rng(0)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]

            # Lower triangle only
            if j >= i:
                ax.axis("off")
                continue

            xj = X[vars_[j]]
            xi = X[vars_[i]]

            # Background: binned + smoothed (no Delaunay triangles)
            field = smooth_field_binned(
                xj, xi, Y,
                nx=args.nx, ny=args.ny, sigma=args.sigma
            )
            if field is not None:
                gx, gy, Z = field
                ax.imshow(
                    Z,
                    origin="lower",
                    extent=(gx.min(), gx.max(), gy.min(), gy.max()),
                    aspect="auto",
                    cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    alpha=args.bg_alpha,
                    interpolation="bilinear",
                    zorder=1,
                )

            # Points on top (optionally jittered for plotting)
            m = np.isfinite(xj) & np.isfinite(xi) & np.isfinite(Y)
            x_plot = jitter_for_plot(xj[m], frac=args.jitter_frac, rng=rng)
            y_plot = jitter_for_plot(xi[m], frac=args.jitter_frac, rng=rng)

            ax.scatter(
                x_plot, y_plot,
                c=Y[m],
                s=args.pt_size,
                alpha=args.pt_alpha,
                cmap=cmap,
                vmin=vmin, vmax=vmax,
                edgecolors="none",
                zorder=3,
            )

            # Labels only outer axes
            if i == n - 1:
                ax.set_xlabel(display_name(vars_[j], args.mode))
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(display_name(vars_[i], args.mode))
            else:
                ax.set_yticklabels([])

            ax.grid(True, ls=":", alpha=0.22)

    # Colorbar (manual axis, like your current script)
    cax = fig.add_axes([0.92, 0.12, 0.02, 0.76])
    cb = plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)),
        cax=cax
    )
    cb.set_label(nice_label(args.y))

    fig.savefig(f"{out_prefix}.png", dpi=args.dpi, bbox_inches="tight")
    print("Saved:", f"{out_prefix}.png")


if __name__ == "__main__":
    main()

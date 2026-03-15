#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compute_burial_path import (
    DATA_ROOT_DEFAULT,
    MODELS_ROOT_DEFAULT,
    OUT_ROOT_DEFAULT,
    _time_tag,
    compute_path,
    make_path_figure,
    PLOT_DEPTH_MAX_KM,
    PLOT_TIME_MAX_MYR,
    PLOT_TIME_MIN_MYR,
    PLOT_TEMP_MAX_C,
    PLOT_TEMP_MIN_C,
)


def _parse_grid(spec: str) -> list[float]:
    spec = spec.strip()
    if not spec:
        raise ValueError("Empty grid specification")
    if ":" in spec:
        parts = [p.strip() for p in spec.split(":")]
        if len(parts) != 3:
            raise ValueError(f"Range spec must be start:stop:num, got {spec!r}")
        start, stop, num = float(parts[0]), float(parts[1]), int(parts[2])
        if num < 1:
            raise ValueError("num must be >= 1")
        if num == 1:
            return [start]
        return list(np.linspace(start, stop, num))
    return [float(x) for x in spec.replace(",", " ").split() if x.strip()]


def _range_text(values: list[float], unit: str) -> str:
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return f"{vals[0]:.2f} {unit}"
    if len(vals) >= 2:
        diffs = np.diff(vals)
        step = diffs[0] if np.allclose(diffs, diffs[0]) else None
        if step is not None:
            return f"{vals[0]:.2f} to {vals[-1]:.2f} {unit} ({step:.2f} step)"
    return ", ".join(f"{v:.2f}" for v in vals) + f" {unit}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute and plot many burial/exhumation paths from the profile-PCA emulator.")
    ap.add_argument("--suite", required=True, choices=["const-vc", "ramped-vc"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--model-tag", default="gp_m25")
    ap.add_argument("--start-time-myr", type=float, default=0.5)
    ap.add_argument("--burial-rates-cm-per-yr", required=True, help="Comma list or start:stop:num")
    ap.add_argument("--max-depths-km", required=True, help="Comma list or start:stop:num")
    ap.add_argument("--exhumation-times-myr", required=True, help="Comma list or start:stop:num")
    ap.add_argument("--exhumation-rate-cm-per-yr", type=float, default=None)
    ap.add_argument("--transition-time-myr", type=float, default=0.0)
    ap.add_argument("--dt-myr", type=float, default=0.05)
    ap.add_argument("--density-kg-m3", type=float, default=3300.0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--data-root", default=str(DATA_ROOT_DEFAULT))
    ap.add_argument("--models-root", default=str(MODELS_ROOT_DEFAULT))
    ap.add_argument("--v-conv", type=float, default=None)
    ap.add_argument("--t-conv", type=float, default=None)
    ap.add_argument("--age-sp", type=float, default=None)
    ap.add_argument("--age-op", type=float, default=None)
    ap.add_argument("--dip-int", type=float, default=None)
    ap.add_argument("--eta-um", type=float, default=None)
    args = ap.parse_args()

    burial_rates = _parse_grid(args.burial_rates_cm_per_yr)
    max_depths = _parse_grid(args.max_depths_km)
    exhumation_times = _parse_grid(args.exhumation_times_myr)

    feature_values = {
        "v_conv": args.v_conv,
        "t_conv": args.t_conv,
        "age_SP": args.age_sp,
        "age_OP": args.age_op,
        "dip_int": args.dip_int,
        "eta_UM": args.eta_um,
    }
    data_root = Path(args.data_root).resolve()
    models_root = Path(args.models_root).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else (OUT_ROOT_DEFAULT / args.suite / "paths")
    outdir.mkdir(parents=True, exist_ok=True)

    combos = list(product(burial_rates, max_depths, exhumation_times))
    if not combos:
        raise SystemExit("[ERR] No path combinations requested.")

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.12, 0.92, len(combos)))

    rows = []
    fig, ax0, ax1 = make_path_figure()

    y_limit = PLOT_DEPTH_MAX_KM
    for i, ((vburial, zmax, thold), color) in enumerate(zip(combos, colors), start=1):
        result = compute_path(
            suite=args.suite,
            k=args.k,
            model_tag=args.model_tag,
            start_time_myr=args.start_time_myr,
            burial_rate_cm_per_yr=vburial,
            exhumation_rate_cm_per_yr=args.exhumation_rate_cm_per_yr,
            exhumation_time_myr=thold,
            transition_time_myr=args.transition_time_myr,
            max_depth_km=zmax,
            dt_myr=args.dt_myr,
            density_kg_m3=args.density_kg_m3,
            feature_values=feature_values,
            data_root=data_root,
            models_root=models_root,
        )
        path_id = f"path_{i:02d}"
        i_star = int(np.nanargmax(result["path_temperature"]))
        ax0.plot(result["path_times"], result["path_depth"], color=color, lw=1.4, alpha=0.55)
        ax0.plot(
            result["path_times"][i_star],
            result["path_depth"][i_star],
            marker="*",
            ms=5.5,
            color=color,
            mec="white",
            mew=0.4,
            zorder=5,
        )
        ax1.plot(result["path_temperature"], result["path_depth"], color=color, lw=1.4, alpha=0.55)
        ax1.plot(
            result["path_temperature"][i_star],
            result["path_depth"][i_star],
            marker="*",
            ms=5.5,
            color=color,
            mec="white",
            mew=0.4,
            zorder=5,
        )

        for t, dt, z, p, temp in zip(
            result["path_times"],
            result["path_offsets"],
            result["path_depth"],
            result["path_pressure_gpa"],
            result["path_temperature"],
        ):
            rows.append(
                {
                    "path_id": path_id,
                    "burial_rate_cm_per_yr": vburial,
                    "exhumation_rate_cm_per_yr": result["exhumation_rate_cm_per_yr"],
                    "exhumation_time_myr": thold,
                    "max_depth_km": zmax,
                    "time_myr": t,
                    "time_since_burial_start_myr": dt,
                    "depth_km": z,
                    "pressure_gpa": p,
                    "temperature_c": temp,
                    "truncated_to_model_time_range": result["truncated_to_model_time_range"],
                }
            )

    ax0.set_xlabel("Time (Myr)")
    ax0.set_ylabel("Depth (km)")
    ax0.set_xlim(PLOT_TIME_MIN_MYR, PLOT_TIME_MAX_MYR)
    ax0.set_ylim(y_limit, 0.0)
    ax0.grid(True, ls=":", alpha=0.35)

    ax1.set_xlabel("Temperature ($^\\circ$C)")
    ax1.set_ylabel("Depth (km)")
    ax1.set_xlim(PLOT_TEMP_MIN_C, PLOT_TEMP_MAX_C)
    ax1.set_ylim(y_limit, 0.0)
    ax1.grid(True, ls=":", alpha=0.35)

    summary = (
        f"$v_b$: {_range_text(burial_rates, 'cm/yr')}\n"
        f"$z_{{max}}$: {_range_text(max_depths, 'km')}\n"
        f"$t_h$: {_range_text(exhumation_times, 'Myr')}"
    )
    ax1.text(
        0.02,
        0.02,
        summary,
        transform=ax1.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
    )

    basename = args.name or (
        f"{args.suite}_manypaths_tstart{_time_tag(args.start_time_myr)}"
        f"_nb{len(burial_rates)}_nz{len(max_depths)}_nh{len(exhumation_times)}"
    )
    csv_path = outdir / f"{basename}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    meta = {
        "suite": args.suite,
        "k": args.k,
        "model_tag": args.model_tag,
        "start_time_myr": float(args.start_time_myr),
        "burial_rates_cm_per_yr": burial_rates,
        "max_depths_km": max_depths,
        "exhumation_times_myr": exhumation_times,
        "n_paths": len(combos),
        "output_csv": str(csv_path),
    }
    meta_path = outdir / f"{basename}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    png_path = outdir / f"{basename}.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")

    print(f"[OK] wrote many-path CSV: {csv_path}")
    print(f"[OK] wrote metadata: {meta_path}")
    print(f"[OK] wrote preview: {png_path}")
    print(f"[OK] n_paths={len(combos)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compute_burial_path import (
    DATA_ROOT_DEFAULT,
    MODELS_ROOT_DEFAULT,
    OUT_ROOT_DEFAULT,
    _build_default_basename,
    _load_run_prediction_context,
    compute_path,
    make_path_figure,
    PLOT_DEPTH_MAX_KM,
    PLOT_TIME_MAX_MYR,
    PLOT_TIME_MIN_MYR,
    PLOT_TEMP_MAX_C,
    PLOT_TEMP_MIN_C,
)


def _feature_key_to_model_name(name: str) -> str:
    mapping = {
        "v-conv": "v_conv",
        "t-conv": "t_conv",
        "age-sp": "age_SP",
        "age-op": "age_OP",
        "dip-int": "dip_int",
        "eta-um": "eta_UM",
        "v_conv": "v_conv",
        "t_conv": "t_conv",
        "age_SP": "age_SP",
        "age_OP": "age_OP",
        "dip_int": "dip_int",
        "eta_UM": "eta_UM",
    }
    if name not in mapping:
        raise KeyError(f"Unsupported feature {name!r}")
    return mapping[name]


def _feature_label(name: str) -> str:
    labels = {
        "v_conv": r"$v_{\rm conv}$",
        "t_conv": r"$t_{\rm conv}$",
        "age_SP": r"$\mathrm{age}_{\rm SP}$",
        "age_OP": r"$\mathrm{age}_{\rm OP}$",
        "dip_int": r"$\theta_{\rm init}$",
        "eta_UM": r"$\eta_{\rm UM}$",
    }
    return labels.get(name, name)


def _load_feature_samples(
    *,
    suite: str,
    k: int,
    model_tag: str,
    feature_name: str,
    data_root: Path,
    models_root: Path,
    ref_time_myr: float,
) -> np.ndarray:
    ctx = _load_run_prediction_context(
        suite=suite,
        time_myr=ref_time_myr,
        k=k,
        model_tag=model_tag,
        data_root=data_root,
        models_root=models_root,
    )
    feature_cols = list(ctx["meta"]["feature_cols"])
    if feature_name not in feature_cols:
        raise KeyError(f"Feature {feature_name!r} not in dataset feature columns: {feature_cols}")
    fidx = feature_cols.index(feature_name)
    return np.asarray(ctx["X_raw"][:, fidx], dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute a single burial path with uncertainty envelope from one varying geodynamic parameter.")
    ap.add_argument("--suite", required=True, choices=["const-vc", "ramped-vc"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--model-tag", default="gp_m25")
    ap.add_argument("--start-time-myr", type=float, default=0.5)
    ap.add_argument("--burial-rate-cm-per-yr", type=float, required=True)
    ap.add_argument("--exhumation-rate-cm-per-yr", type=float, default=None)
    ap.add_argument("--exhumation-time-myr", type=float, default=0.0)
    ap.add_argument("--transition-time-myr", type=float, default=0.0)
    ap.add_argument("--max-depth-km", type=float, required=True)
    ap.add_argument("--dt-myr", type=float, default=0.05)
    ap.add_argument("--density-kg-m3", type=float, default=3300.0)
    ap.add_argument("--uncertain-feature", default="age-sp", help="One of: v-conv, t-conv, age-sp, age-op, dip-int, eta-um")
    ap.add_argument("--n-samples", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
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

    data_root = Path(args.data_root).resolve()
    models_root = Path(args.models_root).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else (OUT_ROOT_DEFAULT / args.suite / "paths")
    outdir.mkdir(parents=True, exist_ok=True)

    feature_name = _feature_key_to_model_name(args.uncertain_feature)
    feature_values = {
        "v_conv": args.v_conv,
        "t_conv": args.t_conv,
        "age_SP": args.age_sp,
        "age_OP": args.age_op,
        "dip_int": args.dip_int,
        "eta_UM": args.eta_um,
    }

    samples_all = _load_feature_samples(
        suite=args.suite,
        k=args.k,
        model_tag=args.model_tag,
        feature_name=feature_name,
        data_root=data_root,
        models_root=models_root,
        ref_time_myr=args.start_time_myr,
    )
    rng = np.random.default_rng(args.seed)
    if samples_all.size > args.n_samples:
        sample_vals = np.sort(rng.choice(samples_all, size=args.n_samples, replace=False))
    else:
        sample_vals = np.sort(samples_all.copy())

    baseline_value = float(np.median(samples_all))
    feature_values[feature_name] = baseline_value
    baseline = compute_path(
        suite=args.suite,
        k=args.k,
        model_tag=args.model_tag,
        start_time_myr=args.start_time_myr,
        burial_rate_cm_per_yr=args.burial_rate_cm_per_yr,
        exhumation_rate_cm_per_yr=args.exhumation_rate_cm_per_yr,
        exhumation_time_myr=args.exhumation_time_myr,
        transition_time_myr=args.transition_time_myr,
        max_depth_km=args.max_depth_km,
        dt_myr=args.dt_myr,
        density_kg_m3=args.density_kg_m3,
        feature_values=feature_values,
        data_root=data_root,
        models_root=models_root,
    )

    temp_stack = []
    depth_stack = []
    for value in sample_vals:
        varied = dict(feature_values)
        varied[feature_name] = float(value)
        result = compute_path(
            suite=args.suite,
            k=args.k,
            model_tag=args.model_tag,
            start_time_myr=args.start_time_myr,
            burial_rate_cm_per_yr=args.burial_rate_cm_per_yr,
            exhumation_rate_cm_per_yr=args.exhumation_rate_cm_per_yr,
            exhumation_time_myr=args.exhumation_time_myr,
            transition_time_myr=args.transition_time_myr,
            max_depth_km=args.max_depth_km,
            dt_myr=args.dt_myr,
            density_kg_m3=args.density_kg_m3,
            feature_values=varied,
            data_root=data_root,
            models_root=models_root,
        )
        temp_stack.append(np.asarray(result["path_temperature"], dtype=float))
        depth_stack.append(np.asarray(result["path_depth"], dtype=float))

    temp_arr = np.vstack(temp_stack)
    depth_arr = np.vstack(depth_stack)
    temp_lo = np.percentile(temp_arr, 5, axis=0)
    temp_hi = np.percentile(temp_arr, 95, axis=0)
    depth_lo = np.percentile(depth_arr, 5, axis=0)
    depth_hi = np.percentile(depth_arr, 95, axis=0)

    basename = args.name or (
        _build_default_basename(
            suite=args.suite,
            start_time_myr=args.start_time_myr,
            burial_rate_cm_per_yr=args.burial_rate_cm_per_yr,
            exhumation_rate_cm_per_yr=baseline["exhumation_rate_cm_per_yr"],
            exhumation_time_myr=args.exhumation_time_myr,
            transition_time_myr_used=baseline["transition_time_myr_used"],
            max_depth_km=args.max_depth_km,
        )
        + f"_uncertain_{feature_name}"
    )

    df = pd.DataFrame(
        {
            "time_myr": baseline["path_times"],
            "time_since_burial_start_myr": baseline["path_offsets"],
            "depth_km_median": baseline["path_depth"],
            "depth_km_p05": depth_lo,
            "depth_km_p95": depth_hi,
            "temperature_c_median": baseline["path_temperature"],
            "temperature_c_p05": temp_lo,
            "temperature_c_p95": temp_hi,
        }
    )
    csv_path = outdir / f"{basename}.csv"
    df.to_csv(csv_path, index=False)

    meta = {
        "suite": args.suite,
        "k": args.k,
        "model_tag": args.model_tag,
        "uncertain_feature": feature_name,
        "uncertain_feature_label": _feature_label(feature_name),
        "n_samples": int(sample_vals.size),
        "sample_min": float(np.min(sample_vals)),
        "sample_median": baseline_value,
        "sample_max": float(np.max(sample_vals)),
        "output_csv": str(csv_path),
    }
    meta_path = outdir / f"{basename}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    fig, ax0, ax1 = make_path_figure()

    ax0.fill_between(
        baseline["path_times"],
        depth_lo,
        depth_hi,
        color="0.7",
        alpha=0.45,
        linewidth=0,
    )
    ax0.plot(baseline["path_times"], baseline["path_depth"], color="tab:blue", lw=2.0)
    i_star = int(np.nanargmax(baseline["path_temperature"]))
    ax0.plot(
        baseline["path_times"][i_star],
        baseline["path_depth"][i_star],
        marker="*",
        ms=8,
        color="k",
        mec="white",
        mew=0.5,
        zorder=5,
    )
    ax0.set_xlabel("Time (Myr)")
    ax0.set_ylabel("Depth (km)")
    ax0.set_xlim(PLOT_TIME_MIN_MYR, PLOT_TIME_MAX_MYR)
    ax0.set_ylim(PLOT_DEPTH_MAX_KM, 0.0)
    ax0.grid(True, ls=":", alpha=0.35)

    ax1.fill_betweenx(
        baseline["path_depth"],
        temp_lo,
        temp_hi,
        color="0.7",
        alpha=0.45,
        linewidth=0,
    )
    ax1.plot(baseline["path_temperature"], baseline["path_depth"], color="tab:red", lw=2.2)
    ax1.plot(
        baseline["path_temperature"][i_star],
        baseline["path_depth"][i_star],
        marker="*",
        ms=8,
        color="k",
        mec="white",
        mew=0.5,
        zorder=5,
    )
    ax1.set_xlabel("Temperature ($^\\circ$C)")
    ax1.set_ylabel("Depth (km)")
    ax1.set_xlim(PLOT_TEMP_MIN_C, PLOT_TEMP_MAX_C)
    ax1.set_ylim(PLOT_DEPTH_MAX_KM, 0.0)
    ax1.grid(True, ls=":", alpha=0.35)
    ax1.text(
        0.02,
        0.02,
        f"{_feature_label(feature_name)} range: {np.min(sample_vals):.2f} to {np.max(sample_vals):.2f}\nmedian: {baseline_value:.2f}",
        transform=ax1.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
    )

    png_path = outdir / f"{basename}.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")

    print(f"[OK] wrote uncertain-path CSV: {csv_path}")
    print(f"[OK] wrote metadata: {meta_path}")
    print(f"[OK] wrote preview: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

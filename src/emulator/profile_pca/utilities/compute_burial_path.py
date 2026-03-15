#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT_DEFAULT = REPO_ROOT / "src" / "emulator" / "data" / "profile_pca"
MODELS_ROOT_DEFAULT = REPO_ROOT / "src" / "emulator" / "models" / "profile_pca"
OUT_ROOT_DEFAULT = REPO_ROOT / "plots" / "science-emulator" / "profile-pca"
PLOT_DEPTH_MAX_KM = 60.0
PLOT_TEMP_MIN_C = -50.0
PLOT_TEMP_MAX_C = 600.0
PLOT_TIME_MIN_MYR = 0.0
PLOT_TIME_MAX_MYR = 5.0


def _time_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _time_from_dataset_name(name: str) -> float:
    m = re.search(r"_t([0-9p]+)Myr_", name)
    if not m:
        raise ValueError(f"Could not parse time from dataset name: {name}")
    return float(m.group(1).replace("p", "."))


def _predict_raw_scores(
    model: Any,
    X_std: np.ndarray,
    y_mu: np.ndarray,
    y_sd: np.ndarray,
    score_space: str,
    score_scale: np.ndarray,
) -> np.ndarray:
    y_pred_std = np.asarray(model.predict(X_std))
    if y_pred_std.ndim == 1:
        y_pred_std = y_pred_std.reshape(-1, 1)
    y_pred = y_pred_std * y_sd[None, :] + y_mu[None, :]
    if score_space == "whitened":
        y_pred = y_pred * score_scale[None, :]
    return y_pred


def _load_run_prediction_context(
    *,
    suite: str,
    time_myr: float,
    k: int,
    model_tag: str,
    data_root: Path,
    models_root: Path,
) -> dict[str, Any]:
    tlabel = _time_tag(time_myr)
    dname = f"profileT_pca_t{tlabel}Myr_k{k}"
    ds = data_root / suite / "runs" / dname
    md = models_root / suite / "runs" / dname / model_tag
    if not ds.exists():
        raise FileNotFoundError(f"Missing dataset dir: {ds}")
    if not md.exists():
        raise FileNotFoundError(f"Missing model dir: {md}")

    with open(ds / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "dataset_name": dname,
        "dataset_dir": ds,
        "model_dir": md,
        "meta": meta,
        "X_raw": np.load(ds / "X_raw.npy"),
        "depth_grid": np.asarray(meta["profile"]["depth_grid_km"], dtype=float),
        "mean_profile": np.load(ds / "pca_mean_profile.npy"),
        "components": np.load(ds / "pca_components.npy"),
        "score_scale": np.load(ds / "pca_score_scale.npy") if (ds / "pca_score_scale.npy").exists() else None,
        "model": joblib.load(md / "model.joblib"),
    }


def _predict_profile(
    *,
    feature_values: dict[str, float],
    ctx: dict[str, Any],
) -> np.ndarray:
    meta = ctx["meta"]
    feature_cols = list(meta["feature_cols"])
    X_raw_train = np.asarray(ctx["X_raw"], dtype=float)

    x_mu = np.asarray(meta["scalers"]["X"]["mean"], dtype=float)
    x_sd = np.asarray(meta["scalers"]["X"]["std"], dtype=float)
    y_mu = np.asarray(meta["scalers"]["Y"]["mean"], dtype=float)
    y_sd = np.asarray(meta["scalers"]["Y"]["std"], dtype=float)

    score_space = str(meta.get("target", {}).get("score_space", "raw")).strip().lower()
    score_scale = ctx["score_scale"]
    if score_scale is None:
        score_scale = np.ones(ctx["components"].shape[0], dtype=float)

    x_row = np.empty(len(feature_cols), dtype=float)
    for i, name in enumerate(feature_cols):
        if name in feature_values and feature_values[name] is not None:
            x_row[i] = float(feature_values[name])
        else:
            x_row[i] = float(np.median(X_raw_train[:, i]))

    X_std = ((x_row[None, :] - x_mu[None, :]) / x_sd[None, :]).astype(float)
    pred_scores_raw = _predict_raw_scores(ctx["model"], X_std, y_mu, y_sd, score_space, score_scale)
    recon = ctx["mean_profile"][None, :] + pred_scores_raw @ ctx["components"]
    return recon[0]


def _interpolate_profile_in_time(
    target_time: float,
    anchor_times: np.ndarray,
    profiles: np.ndarray,
) -> np.ndarray:
    if target_time <= anchor_times[0]:
        return profiles[0]
    if target_time >= anchor_times[-1]:
        return profiles[-1]
    hi = int(np.searchsorted(anchor_times, target_time, side="right"))
    lo = hi - 1
    t0 = float(anchor_times[lo])
    t1 = float(anchor_times[hi])
    w = (target_time - t0) / (t1 - t0)
    return (1.0 - w) * profiles[lo] + w * profiles[hi]


def _depth_path(
    time_offset_myr: np.ndarray,
    burial_rate_km_per_myr: float,
    exhumation_rate_km_per_myr: float,
    dwell_time_myr: float,
    max_depth_km: float,
) -> np.ndarray:
    t_down = max_depth_km / burial_rate_km_per_myr
    t_hold_end = t_down + dwell_time_myr
    t_up = max_depth_km / exhumation_rate_km_per_myr
    total = t_hold_end + t_up
    depth = np.empty_like(time_offset_myr)
    down_mask = time_offset_myr <= t_down
    hold_mask = (time_offset_myr > t_down) & (time_offset_myr <= t_hold_end)
    up_mask = time_offset_myr > t_hold_end
    depth[down_mask] = burial_rate_km_per_myr * time_offset_myr[down_mask]
    depth[hold_mask] = max_depth_km
    depth[up_mask] = max_depth_km - exhumation_rate_km_per_myr * (time_offset_myr[up_mask] - t_hold_end)
    depth = np.clip(depth, 0.0, max_depth_km)
    depth[time_offset_myr > total] = 0.0
    return depth


def _hermite_blend(y0: np.ndarray, y1: np.ndarray, m0: float, m1: float, x0: float, x1: float, x: np.ndarray) -> np.ndarray:
    # Cubic Hermite interpolation with endpoint slopes m0/m1.
    h = x1 - x0
    u = (x - x0) / h
    h00 = 2 * u**3 - 3 * u**2 + 1
    h10 = u**3 - 2 * u**2 + u
    h01 = -2 * u**3 + 3 * u**2
    h11 = u**3 - u**2
    return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1


def _depth_path_smoothed(
    time_offset_myr: np.ndarray,
    burial_rate_km_per_myr: float,
    exhumation_rate_km_per_myr: float,
    dwell_time_myr: float,
    max_depth_km: float,
    transition_time_myr: float,
) -> tuple[np.ndarray, float]:
    base = _depth_path(
        time_offset_myr,
        burial_rate_km_per_myr,
        exhumation_rate_km_per_myr,
        dwell_time_myr,
        max_depth_km,
    )
    if transition_time_myr <= 0.0:
        return base, 0.0

    t_down = max_depth_km / burial_rate_km_per_myr
    t_hold_end = t_down + dwell_time_myr
    t_total = t_hold_end + max_depth_km / exhumation_rate_km_per_myr

    if dwell_time_myr <= 0.0:
        w = min(transition_time_myr, 0.5 * min(t_down, t_total - t_down))
        if w <= 0.0:
            return base, 0.0
        a = t_down - w
        b = t_down + w
        mask = (time_offset_myr >= a) & (time_offset_myr <= b)
        if np.any(mask):
            y_a = np.array([_depth_path(np.array([a]), burial_rate_km_per_myr, exhumation_rate_km_per_myr, dwell_time_myr, max_depth_km)[0]])
            y_b = np.array([_depth_path(np.array([b]), burial_rate_km_per_myr, exhumation_rate_km_per_myr, dwell_time_myr, max_depth_km)[0]])
            base[mask] = _hermite_blend(y_a, y_b, burial_rate_km_per_myr, -exhumation_rate_km_per_myr, a, b, time_offset_myr[mask])
        return np.clip(base, 0.0, max_depth_km), w

    w1 = min(transition_time_myr, 0.5 * min(t_down, dwell_time_myr))
    w2 = min(transition_time_myr, 0.5 * min(dwell_time_myr, t_total - t_hold_end))
    used = 0.0

    if w1 > 0.0:
        a1 = t_down - w1
        b1 = t_down + w1
        mask1 = (time_offset_myr >= a1) & (time_offset_myr <= b1)
        if np.any(mask1):
            y_a1 = np.array([_depth_path(np.array([a1]), burial_rate_km_per_myr, exhumation_rate_km_per_myr, dwell_time_myr, max_depth_km)[0]])
            y_b1 = np.array([_depth_path(np.array([b1]), burial_rate_km_per_myr, exhumation_rate_km_per_myr, dwell_time_myr, max_depth_km)[0]])
            base[mask1] = _hermite_blend(y_a1, y_b1, burial_rate_km_per_myr, 0.0, a1, b1, time_offset_myr[mask1])
        used = max(used, w1)

    if w2 > 0.0:
        a2 = t_hold_end - w2
        b2 = t_hold_end + w2
        mask2 = (time_offset_myr >= a2) & (time_offset_myr <= b2)
        if np.any(mask2):
            y_a2 = np.array([_depth_path(np.array([a2]), burial_rate_km_per_myr, exhumation_rate_km_per_myr, dwell_time_myr, max_depth_km)[0]])
            y_b2 = np.array([_depth_path(np.array([b2]), burial_rate_km_per_myr, exhumation_rate_km_per_myr, dwell_time_myr, max_depth_km)[0]])
            base[mask2] = _hermite_blend(y_a2, y_b2, 0.0, -exhumation_rate_km_per_myr, a2, b2, time_offset_myr[mask2])
        used = max(used, w2)

    return np.clip(base, 0.0, max_depth_km), used


def _pressure_from_depth(depth_km: np.ndarray, density_kg_m3: float) -> np.ndarray:
    return density_kg_m3 * 9.81 * (depth_km * 1000.0) / 1.0e9


def compute_path(
    *,
    suite: str,
    k: int,
    model_tag: str,
    start_time_myr: float,
    burial_rate_cm_per_yr: float,
    exhumation_rate_cm_per_yr: float | None,
    exhumation_time_myr: float,
    transition_time_myr: float,
    max_depth_km: float,
    dt_myr: float,
    density_kg_m3: float,
    feature_values: dict[str, float | None],
    data_root: Path,
    models_root: Path,
) -> dict[str, Any]:
    burial_rate_km_per_myr = float(burial_rate_cm_per_yr) * 10.0
    exhumation_rate_cm_per_yr = float(exhumation_rate_cm_per_yr) if exhumation_rate_cm_per_yr is not None else float(burial_rate_cm_per_yr)
    exhumation_rate_km_per_myr = exhumation_rate_cm_per_yr * 10.0
    if burial_rate_km_per_myr <= 0.0:
        raise SystemExit("[ERR] --burial-rate-cm-per-yr must be positive")
    if exhumation_rate_km_per_myr <= 0.0:
        raise SystemExit("[ERR] --exhumation-rate-cm-per-yr must be positive")
    if max_depth_km <= 0.0:
        raise SystemExit("[ERR] --max-depth-km must be positive")
    if dt_myr <= 0.0:
        raise SystemExit("[ERR] --dt-myr must be positive")
    if exhumation_time_myr < 0.0:
        raise SystemExit("[ERR] --exhumation-time-myr must be non-negative")
    if transition_time_myr < 0.0:
        raise SystemExit("[ERR] --transition-time-myr must be non-negative")

    available = []
    suite_runs = models_root / suite / "runs"
    for model_dir in sorted(suite_runs.glob(f"profileT_pca_t*Myr_k{k}/{model_tag}")):
        try:
            time_myr = _time_from_dataset_name(model_dir.parent.name)
        except ValueError:
            continue
        available.append(time_myr)
    if not available:
        raise SystemExit(f"[ERR] No matching profile-PCA runs found under {suite_runs}")

    anchor_times = np.asarray(sorted(set(available)), dtype=float)
    time_min = float(anchor_times[0])
    time_max = float(anchor_times[-1])

    t_down = max_depth_km / burial_rate_km_per_myr
    t_hold = float(exhumation_time_myr)
    t_up = max_depth_km / exhumation_rate_km_per_myr
    total_cycle = t_down + t_hold + t_up
    requested_end = start_time_myr + total_cycle
    usable_end = min(requested_end, time_max)
    if start_time_myr < time_min or start_time_myr > time_max:
        raise SystemExit(f"[ERR] start time {start_time_myr:g} Myr is outside available emulator time range [{time_min:g}, {time_max:g}] Myr")
    if usable_end <= start_time_myr:
        raise SystemExit("[ERR] No usable time window after applying model time bounds.")

    run_contexts = []
    for t in anchor_times:
        ctx = _load_run_prediction_context(
            suite=suite,
            time_myr=float(t),
            k=k,
            model_tag=model_tag,
            data_root=data_root,
            models_root=models_root,
        )
        profile = _predict_profile(feature_values=feature_values, ctx=ctx)
        run_contexts.append((float(t), ctx["depth_grid"], profile, ctx["meta"]))

    depth_grid = run_contexts[0][1]
    for _, dg, _, _ in run_contexts[1:]:
        if not np.allclose(depth_grid, dg):
            raise SystemExit("[ERR] Profile depth grids are inconsistent across times.")
    profiles = np.vstack([item[2] for item in run_contexts])

    path_times = np.arange(start_time_myr, usable_end + 0.5 * dt_myr, dt_myr)
    path_times = np.clip(path_times, start_time_myr, usable_end)
    path_times = np.unique(np.concatenate([path_times, np.array([start_time_myr, usable_end])]))
    path_offsets = path_times - start_time_myr
    path_depth, used_transition = _depth_path_smoothed(
        path_offsets,
        burial_rate_km_per_myr,
        exhumation_rate_km_per_myr,
        t_hold,
        max_depth_km,
        float(transition_time_myr),
    )

    interp_profiles = np.vstack([
        _interpolate_profile_in_time(float(t), anchor_times, profiles)
        for t in path_times
    ])
    path_temperature = np.array([
        float(np.interp(d, depth_grid, profile))
        for d, profile in zip(path_depth, interp_profiles)
    ])
    path_pressure_gpa = _pressure_from_depth(path_depth, density_kg_m3)

    resolved_features = {}
    feature_cols = list(run_contexts[0][3]["feature_cols"])
    X_raw_ref = np.asarray(_load_run_prediction_context(
        suite=suite,
        time_myr=float(anchor_times[0]),
        k=k,
        model_tag=model_tag,
        data_root=data_root,
        models_root=models_root,
    )["X_raw"], dtype=float)
    for i, name in enumerate(feature_cols):
        if feature_values.get(name) is not None:
            resolved_features[name] = float(feature_values[name])
        else:
            resolved_features[name] = float(np.median(X_raw_ref[:, i]))

    return {
        "anchor_times": anchor_times,
        "depth_grid": depth_grid,
        "profiles": profiles,
        "path_times": path_times,
        "path_offsets": path_offsets,
        "path_depth": path_depth,
        "path_temperature": path_temperature,
        "path_pressure_gpa": path_pressure_gpa,
        "requested_end_time_myr": float(requested_end),
        "used_end_time_myr": float(usable_end),
        "truncated_to_model_time_range": bool(requested_end > time_max),
        "burial_rate_cm_per_yr": float(burial_rate_cm_per_yr),
        "burial_rate_km_per_myr": burial_rate_km_per_myr,
        "exhumation_rate_cm_per_yr": exhumation_rate_cm_per_yr,
        "exhumation_rate_km_per_myr": exhumation_rate_km_per_myr,
        "exhumation_time_myr": t_hold,
        "transition_time_myr_requested": float(transition_time_myr),
        "transition_time_myr_used": float(used_transition),
        "max_depth_km": float(max_depth_km),
        "time_step_myr": float(dt_myr),
        "density_kg_m3": float(density_kg_m3),
        "resolved_feature_values": resolved_features,
    }


def _build_default_basename(
    *,
    suite: str,
    start_time_myr: float,
    burial_rate_cm_per_yr: float,
    exhumation_rate_cm_per_yr: float,
    exhumation_time_myr: float,
    transition_time_myr_used: float,
    max_depth_km: float,
) -> str:
    return (
        f"{suite}_tstart{_time_tag(start_time_myr)}"
        f"_vburial{_time_tag(burial_rate_cm_per_yr)}cm"
        f"_vexhum{_time_tag(exhumation_rate_cm_per_yr)}cm"
        f"_thold{_time_tag(exhumation_time_myr)}"
        f"{f'_tsmooth{_time_tag(transition_time_myr_used)}' if transition_time_myr_used > 0 else ''}"
        f"_zmax{_time_tag(max_depth_km)}km"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute a burial/exhumation depth-temperature-time path from profile-PCA emulators.")
    ap.add_argument("--suite", required=True, choices=["const-vc", "ramped-vc"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--model-tag", default="gp_m25")
    ap.add_argument("--start-time-myr", type=float, default=0.5, help="Burial start time in Myr.")
    ap.add_argument("--burial-rate-cm-per-yr", type=float, required=True)
    ap.add_argument("--exhumation-rate-cm-per-yr", type=float, default=None)
    ap.add_argument("--exhumation-time-myr", type=float, default=0.0, help="Residence time at maximum depth before exhumation.")
    ap.add_argument("--transition-time-myr", type=float, default=0.0, help="Optional smoothing half-width for velocity transitions.")
    ap.add_argument("--max-depth-km", type=float, required=True)
    ap.add_argument("--dt-myr", type=float, default=0.05, help="Time sampling interval along the path.")
    ap.add_argument("--density-kg-m3", type=float, default=3300.0, help="Lithostatic density for pressure conversion.")
    ap.add_argument("--name", default=None, help="Optional basename for outputs.")
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
        feature_values=feature_values,
        data_root=data_root,
        models_root=models_root,
    )

    basename = args.name or _build_default_basename(
        suite=args.suite,
        start_time_myr=args.start_time_myr,
        burial_rate_cm_per_yr=args.burial_rate_cm_per_yr,
        exhumation_rate_cm_per_yr=result["exhumation_rate_cm_per_yr"],
        exhumation_time_myr=args.exhumation_time_myr,
        transition_time_myr_used=result["transition_time_myr_used"],
        max_depth_km=args.max_depth_km,
    )
    outdir = Path(args.outdir).resolve() if args.outdir else (OUT_ROOT_DEFAULT / args.suite / "paths")
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "time_myr": result["path_times"],
            "time_since_burial_start_myr": result["path_offsets"],
            "depth_km": result["path_depth"],
            "pressure_gpa": result["path_pressure_gpa"],
            "temperature_c": result["path_temperature"],
        }
    )
    csv_path = outdir / f"{basename}_dtt.csv"
    df.to_csv(csv_path, index=False)

    meta = {
        "suite": args.suite,
        "k": args.k,
        "model_tag": args.model_tag,
        "available_model_times_myr": result["anchor_times"].tolist(),
        "start_time_myr": float(args.start_time_myr),
        "requested_end_time_myr": result["requested_end_time_myr"],
        "used_end_time_myr": result["used_end_time_myr"],
        "truncated_to_model_time_range": result["truncated_to_model_time_range"],
        "burial_rate_cm_per_yr": float(args.burial_rate_cm_per_yr),
        "burial_rate_km_per_myr": result["burial_rate_km_per_myr"],
        "exhumation_rate_cm_per_yr": result["exhumation_rate_cm_per_yr"],
        "exhumation_rate_km_per_myr": result["exhumation_rate_km_per_myr"],
        "exhumation_time_myr": result["exhumation_time_myr"],
        "transition_time_myr_requested": result["transition_time_myr_requested"],
        "transition_time_myr_used": result["transition_time_myr_used"],
        "max_depth_km": float(args.max_depth_km),
        "time_step_myr": float(args.dt_myr),
        "density_kg_m3": float(args.density_kg_m3),
        "resolved_feature_values": result["resolved_feature_values"],
        "output_csv": str(csv_path),
    }
    meta_path = outdir / f"{basename}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.2, 4.8), constrained_layout=True)
    ax0.plot(result["path_temperature"], result["path_depth"], color="tab:red", lw=2.2)
    i_star = int(np.nanargmax(result["path_temperature"]))
    ax0.plot(result["path_temperature"][i_star], result["path_depth"][i_star], marker="*", ms=8, color="k", mec="white", mew=0.5, zorder=5)
    ax0.set_xlabel("Temperature ($^\\circ$C)")
    ax0.set_ylabel("Depth (km)")
    ax0.set_xlim(PLOT_TEMP_MIN_C, PLOT_TEMP_MAX_C)
    ax0.set_ylim(PLOT_DEPTH_MAX_KM, 0.0)
    ax0.grid(True, ls=":", alpha=0.35)

    ax1.plot(result["path_times"], result["path_depth"], color="tab:blue", lw=2.0)
    ax1.plot(result["path_times"][i_star], result["path_depth"][i_star], marker="*", ms=8, color="k", mec="white", mew=0.5, zorder=5)
    ax1.set_xlabel("Time (Myr)")
    ax1.set_ylabel("Depth (km)")
    ax1.set_xlim(PLOT_TIME_MIN_MYR, PLOT_TIME_MAX_MYR)
    ax1.set_ylim(PLOT_DEPTH_MAX_KM, 0.0)
    ax1.grid(True, ls=":", alpha=0.35)

    png_path = outdir / f"{basename}_preview.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")

    print(f"[OK] wrote path CSV: {csv_path}")
    print(f"[OK] wrote metadata: {meta_path}")
    print(f"[OK] wrote preview: {png_path}")
    if result["truncated_to_model_time_range"]:
        print(
            f"[WARN] requested path extends to {result['requested_end_time_myr']:.3f} Myr, "
            f"but model support ends at {result['anchor_times'][-1]:.3f} Myr; output truncated."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""EXPLORATORY: transient cooling scaling for single-depth dTdt.

Tests whether the Sobol-derived age_OP -> v_conv crossover in transient
forearc cooling is organized by physical length scales:

  - diffusive / initial-condition scale  delta_OP = sqrt(kappa * age_OP)
    (overriding-plate conductive-lid thickness)
  - advective penetration scale          L_adv = v_conv * sin(dip) * dt

Hypothesis: the depth where control of cooling flips from age_OP (shallow,
diffusive) to v_conv (deep, advective) tracks delta_OP rather than L_adv.

This is a first-pass diagnostic on the raw ensemble (no emulator), with
explicit, stated assumptions (z = depth below surface; slab-normal vertical
advection ~ v*sin(dip); kappa = 1e-6 m^2/s). Geometry assumptions should be
confirmed against the model setup before any of this is treated as final.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[4]

KAPPA_KM2_PER_MYR = 1.0e-6 * 3.1536e13 / 1.0e6  # 1e-6 m^2/s -> km^2/Myr ~= 31.5
CM_PER_YR_TO_KM_PER_MYR = 10.0  # 1 cm/yr = 10 km/Myr
PARAM_COLS = ["v_conv", "age_SP", "age_OP", "dip_int", "eta_UM"]


def _masked_corr(y: np.ndarray, x: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 5 or np.std(y[m]) < 1e-9 or np.std(x[m]) < 1e-9:
        return float("nan")
    return float(np.corrcoef(y[m], x[m])[0, 1])


def _load_ensemble(suite: str) -> tuple[pd.DataFrame, list[str]]:
    params_path = REPO_ROOT / "data" / "params" / f"params-list.{suite}.csv"
    master_path = REPO_ROOT / "subd-model-runs" / suite / "analysis" / "master_DT1-10.csv"
    params = pd.read_csv(params_path)
    param_cols = list(params.columns)
    params["run_id"] = np.arange(len(params))
    master = pd.read_csv(master_path)
    df = master.merge(params, on="run_id", how="inner")
    return df, param_cols


# Cooling window: snapshots 1..10 span ~0.5 -> 5.0 Myr (dt ~= 4.5 Myr).
WINDOW_START_MYR = 0.5


def _effective_velocity_cm_yr(suite: str, df: pd.DataFrame, t_end: np.ndarray) -> np.ndarray:
    """Mean convergence velocity over [0, t_end] (cm/yr).

    const-vc: equals v_conv. ramped-vc: v(t) = v_conv*min(t/t_conv, 1), so the
    displacement D(t_end) = v_conv*t_end^2/(2 t_conv) while ramping, else
    v_conv*(t_end - t_conv/2); v_eff = D/t_end.
    """
    vc = df["v_conv"].to_numpy(float)
    if suite != "ramped-vc" or "t_conv" not in df.columns:
        return vc
    tconv = np.maximum(df["t_conv"].to_numpy(float), 1e-6)
    ramping = t_end <= tconv
    disp = np.where(ramping, vc * t_end ** 2 / (2 * tconv), vc * (t_end - tconv / 2))
    return disp / t_end


def _crossover_depth(depths: np.ndarray, c_age: np.ndarray, c_vadv: np.ndarray) -> float | None:
    """First depth where |corr(dTdt, v*sin dip)| overtakes |corr(dTdt, age_OP)|."""
    diff = np.abs(c_vadv) - np.abs(c_age)
    fin = np.isfinite(diff)
    d = depths[fin]
    g = diff[fin]
    for i in range(1, len(g)):
        if g[i - 1] < 0 <= g[i]:
            d0, d1 = d[i - 1], d[i]
            y0, y1 = g[i - 1], g[i]
            return float(d1 if y1 == y0 else d0 + (d1 - d0) * (-y0) / (y1 - y0))
    return None


def _corr_by_depth(df: pd.DataFrame, depths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c_age = np.full(len(depths), np.nan)
    c_vadv = np.full(len(depths), np.nan)
    for i, z in enumerate(depths):
        sub = df[df["depth_km"] == z]
        if len(sub) < 5:
            continue
        y = sub["dTdt_C_per_Myr"].to_numpy(float)
        v_adv = sub["v_conv"].to_numpy(float) * np.sin(np.deg2rad(sub["dip_int"].to_numpy(float)))
        c_age[i] = _masked_corr(y, sub["age_OP"].to_numpy(float))
        c_vadv[i] = _masked_corr(y, v_adv)
    return c_age, c_vadv


def _fit_eval(d, cols, y, is_test):
    from sklearn.ensemble import RandomForestRegressor

    X = d[cols].to_numpy(float)
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1)
    rf.fit(X[~is_test], y[~is_test])
    pred = rf.predict(X[is_test])
    yt = y[is_test]
    rmse = float(np.sqrt(np.mean((pred - yt) ** 2)))
    r2 = float(1 - np.sum((yt - pred) ** 2) / np.sum((yt - np.mean(yt)) ** 2))
    return r2, rmse, pred, yt


def _predict_dT_from_scaling(df: pd.DataFrame, suite: str, param_cols: list[str], outdir: Path) -> None:
    """Does ΔT collapse onto the dimensionless groups well enough to predict it?

    Fits the same flexible learner (RandomForest), split by *run* (no depth
    leakage), and reports held-out R²/RMSE in °C for:
      S2  = {eta, zeta}              -- the two dimensionless groups
      S3  = {eta, zeta, Pe}          -- + advective Peclet (refines deep regime)
      R3  = {age_OP, v_conv, dip, z} -- the raw ingredients of those groups
      R6  = all raw params + z       -- full information the emulator has
    ramped-vc additionally compares zeta built from instantaneous v_conv vs from
    cumulative convergence (effective velocity), testing whether velocity history
    matters.  S2≈R3 => the dimensionless reduction preserves information;
    R6−R3 => what age_SP / eta_UM add beyond the scaling's ingredients.
    """
    d = df.dropna(subset=["dT_C"]).copy()
    z = d["depth_km"].to_numpy(float)
    dt = d["dt_Myr"].to_numpy(float)
    t_end = WINDOW_START_MYR + dt
    sin_dip = np.sin(np.deg2rad(d["dip_int"].to_numpy(float)))

    w_inst = d["v_conv"].to_numpy(float) * CM_PER_YR_TO_KM_PER_MYR * sin_dip
    v_eff = _effective_velocity_cm_yr(suite, d, t_end)
    w_eff = v_eff * CM_PER_YR_TO_KM_PER_MYR * sin_dip

    d["eta"] = z / np.sqrt(KAPPA_KM2_PER_MYR * d["age_OP"].to_numpy(float))
    d["zeta"] = z / np.maximum(w_inst * dt, 1e-6)          # instantaneous v
    d["zeta_cum"] = z / np.maximum(w_eff * dt, 1e-6)       # cumulative/effective v
    d["Pe"] = w_eff * z / KAPPA_KM2_PER_MYR
    d["z"] = z
    y = d["dT_C"].to_numpy(float)

    is_ramped = suite == "ramped-vc"
    adv = "zeta_cum" if is_ramped else "zeta"
    feature_sets = {
        "S2 {eta,zeta}": ["eta", "zeta"],
        "S3 {eta,zeta,Pe}": ["eta", adv, "Pe"],
        "R3 {age_OP,v_conv,dip,z}": ["age_OP", "v_conv", "dip_int", "z"],
        "R6 {all raw + z}": param_cols + ["z"],
    }
    if is_ramped:
        feature_sets = {"S2cum {eta,zeta_cum}": ["eta", "zeta_cum"], **feature_sets}

    rng = np.random.default_rng(42)
    runs = d["run_id"].unique()
    rng.shuffle(runs)
    n_test = int(0.2 * len(runs))
    is_test = d["run_id"].isin(set(runs[:n_test].tolist())).to_numpy()

    print("\npredicting dT_C (°C) from scaling vs raw (held-out runs):")
    print(f"  {len(runs) - n_test} train runs / {n_test} test runs")
    results = {}
    for name, cols in feature_sets.items():
        r2, rmse, pred, yt = _fit_eval(d, cols, y, is_test)
        results[name] = (r2, rmse, pred, yt)
        print(f"  {name:28s} R2={r2:.3f}  RMSE={rmse:6.2f} °C")
    if is_ramped:
        print("  (compare S2 instantaneous vs S2cum: does velocity history help?)")

    # Figure: predicted-vs-true (colored by depth) + residual-vs-depth, for S2.
    r2_s2, rmse_s2, pred_s2, yt_s2 = results["S2 {eta,zeta}"]
    z_test = d["z"].to_numpy(float)[is_test]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    lim = [min(yt_s2.min(), pred_s2.min()), max(yt_s2.max(), pred_s2.max())]
    sc = a0.scatter(yt_s2, pred_s2, c=z_test, s=7, alpha=0.5, cmap="viridis")
    a0.plot(lim, lim, "k--", lw=1)
    a0.set_xlabel("true ΔT (°C)"); a0.set_ylabel("predicted ΔT (°C)")
    a0.set_title(f"S2 scaling {{η,ζ}}:  R²={r2_s2:.3f}, RMSE={rmse_s2:.1f} °C")
    a0.grid(alpha=0.25)
    cb = fig.colorbar(sc, ax=a0, fraction=0.046, pad=0.04)
    cb.set_label("depth (km)")
    a1.scatter(z_test, pred_s2 - yt_s2, s=6, alpha=0.25, color="#EE6677")
    a1.axhline(0, color="k", lw=1)
    a1.axvline(43.5, color="0.4", ls=":", lw=1.2, label=r"$\sqrt{\kappa\,\mathrm{age}_{OP}}$")
    a1.set_xlabel("depth (km)"); a1.set_ylabel("residual (pred − true, °C)")
    a1.set_title("S2 residual vs depth"); a1.grid(alpha=0.25); a1.legend(fontsize=9)
    f = outdir / f"{suite}_dT_scaling_prediction.png"
    fig.savefig(f, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {f}")


def _closed_form_scaling(df: pd.DataFrame, suite: str, outdir: Path) -> None:
    """Fit physically-grounded CLOSED-FORM scalings (real coefficients).

    Anchored on the overriding-plate half-space geotherm:
        E(z) = erf( z / (2 sqrt(kappa*age_OP)) ) = erf(eta/2)   [known shape]
    Model 1 (1 constant):  dT = -A * E
    Model 2 (2 constants): dT = -A * E * (1 - exp(-(w_eff*dt)/(b*z)))   advective
                                 factor -> 1 where cold has reached depth z.
    Held-out by run; reports A (and implied efficiency A/T_m), b, R2, RMSE(°C).
    """
    from scipy.optimize import curve_fit
    from scipy.special import erf

    d = df.dropna(subset=["dT_C"]).copy()
    z = d["depth_km"].to_numpy(float)
    dt = d["dt_Myr"].to_numpy(float)
    t_end = WINDOW_START_MYR + dt
    sin_dip = np.sin(np.deg2rad(d["dip_int"].to_numpy(float)))
    v_eff = _effective_velocity_cm_yr(suite, d, t_end)
    w_eff = np.maximum(v_eff * CM_PER_YR_TO_KM_PER_MYR * sin_dip, 1e-6)
    E = erf(z / (2.0 * np.sqrt(KAPPA_KM2_PER_MYR * d["age_OP"].to_numpy(float))))
    y = d["dT_C"].to_numpy(float)

    rng = np.random.default_rng(42)
    runs = d["run_id"].unique()
    rng.shuffle(runs)
    n_test = int(0.2 * len(runs))
    is_test = d["run_id"].isin(set(runs[:n_test].tolist())).to_numpy()
    tr, te = ~is_test, is_test

    def _scores(pred, yt):
        rmse = float(np.sqrt(np.mean((pred - yt) ** 2)))
        r2 = float(1 - np.sum((yt - pred) ** 2) / np.sum((yt - np.mean(yt)) ** 2))
        return r2, rmse

    T_m = float(np.nanmax(df["T1_C"]))  # deep mantle temperature from the data

    # Model 1: dT = -A * E  (linear through origin -> closed-form A).
    A1 = float(-np.sum(E[tr] * y[tr]) / np.sum(E[tr] ** 2))
    p1 = -A1 * E[te]
    r2_1, rmse_1 = _scores(p1, y[te])

    # Model 2: non-separable slab-top form (McKenzie / Molnar-England style).
    # Geometry = slab-top (interface) T at vertical depth z, reached by descending
    # the interface in t_desc ~ z/w. Cooling = ambient OP geotherm minus the steady
    # slab-top temperature, whose effective thermal equilibration is reduced by
    # advection: the descent Peclet Pe = w_eff*z/kappa shrinks the erf argument
    # (fast/deep -> colder slab top). A finite-window transient factor
    # tau = 1 - exp(-t_end/t_desc) accounts for the cooling being measured over a
    # finite window rather than at steady state.
    #   dT = -A * [erf(eta/2) - erf(eta/(2*(1 + a*Pe^b)))] * tau
    eta = z / np.sqrt(KAPPA_KM2_PER_MYR * d["age_OP"].to_numpy(float))
    Pe = w_eff * z / KAPPA_KM2_PER_MYR
    t_desc = z / w_eff
    tau = 1.0 - np.exp(-t_end / np.maximum(t_desc, 1e-6))

    def model2(X, A, a, b):
        et, pe, tf = X
        return -A * (erf(et / 2) - erf(et / (2.0 * (1.0 + a * pe ** b)))) * tf

    try:
        (A2, a2, b2), _ = curve_fit(
            model2, (eta[tr], Pe[tr], tau[tr]), y[tr], p0=[1500.0, 0.4, 0.3],
            bounds=([0.0, 1e-5, 0.1], [4000.0, 50.0, 3.0]), maxfev=60000,
        )
        p2pred = model2((eta[te], Pe[te], tau[te]), A2, a2, b2)
        r2_2, rmse_2 = _scores(p2pred, y[te])
    except Exception as exc:  # pragma: no cover
        A2 = a2 = b2 = float("nan"); p2pred = None; r2_2 = rmse_2 = float("nan")
        print(f"  [model2 fit failed: {exc}]")

    # Context ceilings: flexible RandomForest given the same physical variable(s).
    from sklearn.ensemble import RandomForestRegressor

    zeta_cf = z / np.maximum(w_eff * dt, 1e-6)
    ceil = {}
    for nm, Xc in (("eta", eta[:, None]), ("eta,Pe", np.c_[eta, Pe]),
                   ("eta,Pe,zeta", np.c_[eta, Pe, zeta_cf])):
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1)
        rf.fit(Xc[tr], y[tr])
        ceil[nm] = _scores(rf.predict(Xc[te]), y[te])[0]

    print("\nCLOSED-FORM scaling (held-out runs):")
    print(f"  T_m (from data) = {T_m:.0f} °C")
    print(f"  ceilings (flexible RF):  η R2={ceil['eta']:.3f}   η,Pe R2={ceil['eta,Pe']:.3f}"
          f"   η,Pe,ζ R2={ceil['eta,Pe,zeta']:.3f}")
    print(f"  Model 1  dT = -A·erf(η/2)                       A={A1:7.1f} °C "
          f"(A/T_m={A1/T_m:.2f})  R2={r2_1:.3f}  RMSE={rmse_1:5.1f} °C")
    print(f"  Model 2  -A·[erf(η/2)-erf(η/(2(1+a·Pe^b)))]·τ   A={A2:6.0f} a={a2:.3f} b={b2:.3f}"
          f"  R2={r2_2:.3f}  RMSE={rmse_2:5.1f} °C  [non-separable slab-top]")

    # Figure: closed-form slab-top prediction vs true (colored by depth) + residual.
    z_te = z[te]
    pred = p2pred if p2pred is not None else p1
    d_OP = float(np.sqrt(KAPPA_KM2_PER_MYR * np.median(d["age_OP"].to_numpy(float))))
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    lim = [min(y[te].min(), pred.min()), max(y[te].max(), pred.max())]
    sc = a0.scatter(y[te], pred, c=z_te, s=7, alpha=0.5, cmap="viridis")
    a0.plot(lim, lim, "k--", lw=1)
    a0.set_xlabel("true ΔT (°C)")
    a0.set_ylabel(r"closed-form ΔT (°C)")
    a0.set_title(
        f"{suite} slab-top closed form\n"
        rf"$-A[\mathrm{{erf}}(\eta/2)-\mathrm{{erf}}(\eta/(2(1+a\,Pe^b)))]\tau$:  "
        f"R²={r2_2:.3f}, RMSE={rmse_2:.0f} °C"
    )
    a0.grid(alpha=0.25)
    cb = fig.colorbar(sc, ax=a0, fraction=0.046, pad=0.04); cb.set_label("depth (km)")
    a1.scatter(z_te, pred - y[te], s=6, alpha=0.25, color="#EE6677")
    a1.axhline(0, color="k", lw=1)
    a1.axvline(d_OP, color="0.4", ls=":", lw=1.2, label=rf"$\sqrt{{\kappa\,\mathrm{{age}}_{{OP}}}}$ = {d_OP:.0f} km")
    a1.set_xlabel("depth (km)"); a1.set_ylabel("residual (pred − true, °C)")
    a1.set_title("closed-form residual vs depth"); a1.grid(alpha=0.25); a1.legend(fontsize=9)
    f = outdir / f"{suite}_dT_closed_form.png"
    fig.savefig(f, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", default="const-vc")
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    df, param_cols = _load_ensemble(args.suite)
    depths = np.array(sorted(df["depth_km"].unique()), dtype=float)
    dt = float(df["dt_Myr"].median())

    # Median length scales.
    age_med = float(df["age_OP"].median())
    v_med = float(df["v_conv"].median())
    dip_med = float(df["dip_int"].median())
    delta_OP = float(np.sqrt(KAPPA_KM2_PER_MYR * age_med))
    L_adv = float(v_med * CM_PER_YR_TO_KM_PER_MYR * np.sin(np.deg2rad(dip_med)) * dt)

    # Correlation crossover (whole ensemble).
    c_age, c_vadv = _corr_by_depth(df, depths)
    z_cross = _crossover_depth(depths, c_age, c_vadv)

    # Conditional test: does the crossover move like sqrt(age_OP) when we split
    # the ensemble by age_OP?  Predicted ratio = sqrt(age_hi/age_lo).
    lo = df[df["age_OP"] <= df["age_OP"].median()]
    hi = df[df["age_OP"] > df["age_OP"].median()]
    z_lo = _crossover_depth(depths, *_corr_by_depth(lo, depths))
    z_hi = _crossover_depth(depths, *_corr_by_depth(hi, depths))
    age_lo, age_hi = float(lo["age_OP"].median()), float(hi["age_OP"].median())

    print(f"== {args.suite} transient-scaling exploration ==")
    print(f"kappa = {KAPPA_KM2_PER_MYR:.2f} km^2/Myr   dt = {dt:.3f} Myr")
    print(f"medians: v_conv={v_med:.2f} cm/yr  age_OP={age_med:.1f} Myr  dip={dip_med:.1f} deg")
    print(f"delta_OP = sqrt(kappa*age_OP) = {delta_OP:.1f} km   (OP conductive-lid thickness)")
    print(f"L_adv    = v*sin(dip)*dt      = {L_adv:.1f} km   (advective penetration)")
    print(f"observed correlation crossover z_cross = {z_cross}")
    print(f"  -> delta_OP predicts {delta_OP:.1f} km ; L_adv predicts {L_adv:.1f} km")
    print("conditional (split by age_OP) -- CONFOUNDED by range restriction,")
    print("  treat as inconclusive; proper test needs emulator-based crossover at")
    print("  controlled age_OP (see notes):")
    print(f"  low  age_OP={age_lo:.1f}  z_cross={z_lo}")
    print(f"  high age_OP={age_hi:.1f}  z_cross={z_hi}")
    if z_lo and z_hi:
        print(f"  observed z ratio  = {z_hi / z_lo:.3f}")
        print(f"  sqrt(age) ratio   = {np.sqrt(age_hi / age_lo):.3f}  (prediction)")

    outdir = Path(args.outdir).resolve() if args.outdir else (
        REPO_ROOT / "plots" / "science-emulator" / "single_depth" / args.suite / "scaling"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: correlation vs depth, with delta_OP / L_adv marked ---
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    ax.plot(depths, np.abs(c_age), "-o", ms=3, color="#228833", label=r"$|\mathrm{corr}(\dot T,\ \mathrm{age}_{OP})|$")
    ax.plot(depths, np.abs(c_vadv), "-o", ms=3, color="#4477AA", label=r"$|\mathrm{corr}(\dot T,\ v\sin\theta)|$")
    if z_cross:
        ax.axvline(z_cross, color="0.4", ls="--", lw=1.2, label=f"crossover ≈ {z_cross:.0f} km")
    ax.axvline(delta_OP, color="#EE6677", ls=":", lw=1.6, label=rf"$\sqrt{{\kappa\,\mathrm{{age}}_{{OP}}}}$ = {delta_OP:.0f} km")
    ax.set_xlabel("Depth (km)")
    ax.set_ylabel("|correlation| with cooling rate")
    ax.set_title(f"{args.suite}: control of transient cooling vs depth")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    f1 = outdir / f"{args.suite}_cooling_control_vs_depth.png"
    fig.savefig(f1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {f1}")

    # --- Figure 2: collapse of normalized cooling vs eta = z / delta_OP ---
    # Per-run normalized cooling profile Theta = dT_C(z) / max|dT_C|(run).
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    rng = np.random.default_rng(0)
    run_ids = df["run_id"].unique()
    sample = rng.choice(run_ids, size=min(120, len(run_ids)), replace=False)
    for rid in sample:
        sub = df[df["run_id"] == rid].sort_values("depth_km")
        dT = sub["dT_C"].to_numpy(float)
        denom = np.max(np.abs(dT))
        if denom < 1e-6:
            continue
        theta = dT / denom
        eta = sub["depth_km"].to_numpy(float) / float(np.sqrt(KAPPA_KM2_PER_MYR * sub["age_OP"].iloc[0]))
        ax.plot(eta, theta, color="#4477AA", alpha=0.12, lw=0.8)
    ax.axvline(1.0, color="#EE6677", ls=":", lw=1.6, label=r"$\eta = z/\sqrt{\kappa\,\mathrm{age}_{OP}} = 1$")
    ax.set_xlabel(r"$\eta = z\,/\,\sqrt{\kappa\,\mathrm{age}_{OP}}$  (depth / OP lid thickness)")
    ax.set_ylabel(r"normalized cooling  $\Delta T(z)\,/\,\max|\Delta T|$")
    ax.set_xlim(0, 4)
    ax.set_title(f"{args.suite}: cooling-profile collapse on diffusive similarity variable")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    f2 = outdir / f"{args.suite}_cooling_collapse_eta.png"
    fig.savefig(f2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {f2}")

    # --- Predictive test: can the dimensionless groups predict dT? ---
    _predict_dT_from_scaling(df, args.suite, param_cols, outdir)

    # --- Closed-form physically-grounded scaling (real coefficients) ---
    _closed_form_scaling(df, args.suite, outdir)

    # --- dip-tracks-velocity check from existing Sobol JSONs ---
    sobol_dir = REPO_ROOT / "plots" / "science-emulator" / "single_depth" / args.suite / "sobol"
    js = sorted(sobol_dir.glob("*km_dTdt_gp_m25_sobol.json"))
    if js:
        rows = []
        for jp in js:
            d = json.loads(jp.read_text())
            feats = d["feature_cols"]
            st = dict(zip(feats, d["ST"]))
            if "v_conv" in st and "dip_int" in st and d.get("depth_km") is not None:
                rows.append((d["depth_km"], st["v_conv"], st["dip_int"]))
        rows.sort()
        print("\ndip-tracks-velocity (ST_dip / ST_vconv by depth):")
        for z, sv, sd in rows:
            ratio = sd / sv if sv > 1e-6 else float("nan")
            print(f"  z={z:4.0f} km  ST_v={sv:.3f}  ST_dip={sd:.3f}  ratio={ratio:.3f}")


if __name__ == "__main__":
    main()

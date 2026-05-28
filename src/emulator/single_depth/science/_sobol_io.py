#!/usr/bin/env python3
"""Shared I/O and helpers for single-depth Sobol sensitivity analysis.

Mirrors the dataset/model loading and prediction conventions used by the
one-at-a-time sensitivity script (``plot_single_depth_sensitivity.py``) so the
Sobol tooling reuses the exact same standardization handling.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


def default_label_map() -> dict[str, str]:
    """Pretty math labels for the model parameters (matches the OAT script)."""
    return {
        "v_conv": r"$v_{\mathrm{conv}}$ (cm/yr)",
        "age_SP": r"$\mathrm{age}_{\mathrm{SP}}$ (Myr)",
        "age_OP": r"$\mathrm{age}_{\mathrm{OP}}$ (Myr)",
        "dip_int": r"$\theta_{\mathrm{slab}}$ ($^\circ$)",
        "eta_UM": r"$\eta_{\mathrm{UM}}$ ($\log_{10}$ Pa s)",
        "t_conv": r"$t_{\mathrm{conv}}$ (Myr)",
        "v_conv_over_tconv": r"$v_{\mathrm{conv}}/t_{\mathrm{conv}}$",
    }


def target_label(target_col: str) -> str:
    if target_col == "dTdt_C_per_Myr":
        return r"Cooling rate ($^\circ$C / Myr)"
    return target_col


def load_dataset_bundle(data_dir: Path) -> dict[str, object]:
    """Load metadata, raw inputs, the train split, and the X/Y scalers."""
    meta = json.loads((data_dir / "metadata.json").read_text())
    x_raw = np.load(data_dir / "X_raw.npy")
    train_idx = np.load(data_dir / "train_idx.npy")
    val_idx = np.load(data_dir / "val_idx.npy")

    return {
        "meta": meta,
        "x_raw": x_raw,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "x_mean": np.asarray(meta["scalers"]["X"]["mean"], dtype=float),
        "x_std": np.asarray(meta["scalers"]["X"]["std"], dtype=float),
        "y_mean": np.asarray(meta["scalers"]["Y"]["mean"], dtype=float),
        "y_std": np.asarray(meta["scalers"]["Y"]["std"], dtype=float),
    }


def predict_raw(
    model: object,
    x_raw: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
) -> np.ndarray:
    """Predict in raw target units for a single-target model.

    Standardizes inputs, predicts in standardized space, then inverts the Y
    scaler. Returns a 1-D array of length ``len(x_raw)``.
    """
    x_std_space = (x_raw - x_mean) / x_std
    yhat_std = np.asarray(model.predict(x_std_space))
    if yhat_std.ndim == 2:
        yhat_std = yhat_std[:, 0]
    y0 = float(np.atleast_1d(y_mean)[0])
    s0 = float(np.atleast_1d(y_std)[0])
    return yhat_std * s0 + y0


def build_problem(
    feature_cols: list[str],
    x_train: np.ndarray,
    q_low: float,
    q_high: float,
) -> dict[str, object]:
    """Build a SALib problem definition from per-feature training quantiles.

    Uses the central ``[q_low, q_high]`` quantile range of each feature in the
    training split as the sampling box, keeping Saltelli samples inside
    well-supported regions of the emulator. Constant/degenerate features are
    widened slightly so SALib does not see a zero-width bound.
    """
    bounds: list[list[float]] = []
    for j in range(x_train.shape[1]):
        lo, hi = np.quantile(x_train[:, j], [q_low, q_high])
        lo = float(lo)
        hi = float(hi)
        if hi <= lo:
            eps = abs(lo) * 1e-6 + 1e-9
            lo, hi = lo - eps, hi + eps
        bounds.append([lo, hi])
    return {
        "num_vars": len(feature_cols),
        "names": list(feature_cols),
        "bounds": bounds,
    }


def depth_from_data_name(data_name: str, meta: dict | None = None) -> float | None:
    """Extract the observation depth (km) from a dataset name like ``40km_dTdt``.

    Falls back to the first ``obs_depth_km`` entry in metadata if the name does
    not carry a leading ``<n>km`` token.
    """
    m = re.match(r"^(\d+(?:\.\d+)?)km", data_name)
    if m:
        return float(m.group(1))
    if meta is not None and "obs_depth_km" in meta:
        vals = np.atleast_1d(np.asarray(meta["obs_depth_km"], dtype=float))
        if vals.size:
            return float(vals[0])
    return None

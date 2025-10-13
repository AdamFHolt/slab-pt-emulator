#!/usr/bin/env python3

"""
Preprocess training data.

- Reads params-list.csv and one or more master_*.csv files.
- Merges on run_id.
- Builds X from known parameter columns (only those present).
- Builds Y automatically:
    * If ['dT_C', 'dTdt_C_per_Myr'] exist: uses those (target_kind='summary_dT')
    * Else: requires --targets=<col1,col2,...>
- Drops rows with NaNs in X or Y.
- Standardizes X and Y (z-score) and saves raw + standardized arrays.
- Saves split indices (train/val) and metadata.json (columns, scalers, depths, etc).

Usage examples:
    # Typical: combine all master_* CSVs, auto-detect targets
    python preprocess_training.py \
        --masters "../../subd-model-runs/run-outputs/analysis/master_*km_DT1-6.csv"

    # Explicit targets:
    python preprocess_training.py \
        --masters "../../subd-model-runs/run-outputs/analysis/master_*km_DT1-6.csv" \
        --targets "dTdt_C_per_Myr"
"""

from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Preferred parameter columns (will use intersection with actual columns)
DEFAULT_PARAM_COLS = ["v_conv", "age_SP", "age_OP", "dip_int", "eta_int", "eta_UM", "eps_trans"]

# Default relative locations (resolved from this file)
THIS_FILE = Path(__file__).resolve()

# Repo root assumed to be two levels above this file: .../src/emulator/preprocess_training.py
SLABPT_ROOT_DEFAULT = THIS_FILE.parents[2]


def _resolve_root(env_var: str | None) -> Path:
    if env_var and len(env_var.strip()) > 0:
        return Path(env_var).expanduser().resolve()
    return SLABPT_ROOT_DEFAULT


def _load_params(params_path: Path) -> pd.DataFrame:
    dfp = pd.read_csv(params_path)
    # Ensure a string run_id exists and is zero-padded if not already provided.
    if "run_id" not in dfp.columns:
        dfp = dfp.copy()
        dfp["run_id"] = [f"{i:03d}" for i in range(len(dfp))]
    else:
        # Coerce to zero-padded strings if they look numeric
        if np.issubdtype(dfp["run_id"].dtype, np.number):
            dfp["run_id"] = dfp["run_id"].astype(int).map(lambda i: f"{i:03d}")
        else:
            dfp["run_id"] = dfp["run_id"].astype(str).str.zfill(3)
    return dfp


def _load_masters(masters_glob: str | List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Accepts a glob string or list of explicit paths.
    Returns concatenated dataframe and list of source file paths (str).
    """
    if isinstance(masters_glob, list):
        files = [Path(p).resolve() for p in masters_glob]
    else:
        files = sorted(Path().resolve().glob(masters_glob))
        if len(files) == 0:
            base = (THIS_FILE.parent / Path(masters_glob)).resolve()
            files = sorted(base.parent.glob(base.name))

    if len(files) == 0:
        raise FileNotFoundError(f"No master files found for pattern/paths: {masters_glob}")

    dfs = []
    for f in files:
        df = pd.read_csv(f, dtype={"run_id": str})
        # Standardize run_id formatting
        df["run_id"] = df["run_id"].astype(str).str.zfill(3)
        # Annotate a depth if embedded in filename like 'master_25km_...'
        m = re.search(r"master_(\d+)\s*km", f.name.replace("-", ""))
        depth_km = int(m.group(1)) if m else None
        if depth_km is not None:
            df = df.copy()
            df["obs_depth_km"] = depth_km
        df["source_file"] = f.as_posix()
        dfs.append(df)

    dfm = pd.concat(dfs, ignore_index=True)
    return dfm, [f.as_posix() for f in files]


def _pick_features(df_merged: pd.DataFrame, preferred: List[str]) -> List[str]:
    available = [c for c in preferred if c in df_merged.columns]
    if not available:
        raise ValueError(f"None of the expected parameter columns were found. "
                         f"Tried: {preferred}\nAvailable columns: {list(df_merged.columns)}")
    return available


def _prepare_features(df: pd.DataFrame,feat_cols: List[str],log_cols: List[str], base: str = "10",) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Coerce feature columns to numeric and (optionally) log-transform a subset.
    """
    # 1) coerce all features to numeric (handles '1.23e20' strings)
    for c in feat_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 2) apply logs to the intersection (present & requested)
    tmap: Dict[str, str] = {}
    log_set = {c for c in log_cols if c in df.columns}
    if log_set:
        with np.errstate(divide="ignore", invalid="ignore"):
            for c in log_set:
                vals = df[c].astype(float)
                # non-positive -> NaN (to be dropped by mask later)
                vals = vals.where(vals > 0)
                if base == "10":
                    df[c] = np.log10(vals)
                    tmap[c] = "log10"
                else:
                    df[c] = np.log(vals)
                    tmap[c] = "loge"

    return df, tmap


def _auto_pick_targets(dfm: pd.DataFrame, explicit_targets: List[str] | None) -> Tuple[np.ndarray, Dict]:
    """
    Returns (Y, target_meta)
    - If explicit_targets provided: use those columns in that order.
    - Else: try ['dT_C', 'dTdt_C_per_Myr'].
    """
    meta: Dict = {"target_kind": None}

    # 1) Explicit targets provided by user
    if explicit_targets:
        missing = [t for t in explicit_targets if t not in dfm.columns]
        if missing:
            raise ValueError(f"Requested target columns not found: {missing}")
        Y = dfm[explicit_targets].to_numpy(dtype=float)
        meta["target_kind"] = "explicit"
        meta["target_cols"] = explicit_targets
        return Y, meta

    # 2) Default: use ΔT and ΔT/Δt if they exist
    if ("dT_C" in dfm.columns) and ("dTdt_C_per_Myr" in dfm.columns):
        cols = ["dT_C", "dTdt_C_per_Myr"]
        Y = dfm[cols].to_numpy(dtype=float)
        meta["target_kind"] = "summary_dT"
        meta["target_cols"] = cols
        return Y, meta

    # 3) Otherwise, raise an error
    raise ValueError(
        "Could not auto-detect targets. Pass --targets col1,col2,... "
        "or ensure [dT_C, dTdt_C_per_Myr] exist."
    )

def _standardize(arr: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    mu = np.nanmean(arr, axis=0)
    sigma = np.nanstd(arr, axis=0, ddof=0)
    # Avoid division by zero
    sigma_safe = np.where(sigma == 0, 1.0, sigma)
    arr_std = (arr - mu) / sigma_safe
    return arr_std, {"mean": mu.tolist(), "std": sigma_safe.tolist()}


def _train_val_split(n: int, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(round(val_frac * n))
    val_idx = np.sort(idx[:n_val])
    train_idx = np.sort(idx[n_val:])
    return train_idx, val_idx


def main():
    parser = argparse.ArgumentParser(description="Preprocess Slab P–T training data.")
    parser.add_argument("--params", default="../../data/params-list.csv",
                        help="Path to params-list.csv (relative or absolute).")
    parser.add_argument("--masters", nargs="+", default=["../../subd-model-runs/run-outputs/analysis/master_*km_DT1-6.csv"],
                        help="One or more glob patterns or explicit paths to master_*.csv.")
    parser.add_argument("--targets", default=None,
                        help="Comma-separated target columns to use (overrides auto-detect).")
    parser.add_argument("--max-dTdt", type=float, default=None,
                        help="Drop rows w/ dTdt_C_per_Myr > this val.")
    parser.add_argument("--max-dT", type=float, default=None,
                        help="Drop rows with dT_C > this val.")
    parser.add_argument("--outdir", default=str(THIS_FILE.parent / "data"),
                        help="Output directory for npy/json artifacts.")
    parser.add_argument("--val-frac", type=float, default=0.15,
                        help="Validation fraction (0–1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument("--root-env", default="SLABPT_ROOT",
                        help="Environment variable to override repo root (default: SLABPT_ROOT).")
    parser.add_argument("--log-cols",default="eta_int,eta_UM,eps_trans",help="feature columns to log10-transform.")
    parser.add_argument("--log-base",choices=["10", "e"],default="10")
    args = parser.parse_args()

    # Resolve root and important paths
    repo_root = _resolve_root(os.environ.get(args.root_env))
    params_path = (Path(args.params) if Path(args.params).is_absolute()
                   else (Path.cwd() / args.params)).resolve()
    
    if not params_path.exists():
        raise FileNotFoundError(f"params-list.csv not found at {params_path}")

    # Load run params
    df_params = _load_params(params_path)

    # Load and concat masters
    concat_master_df_list = []
    master_sources_all = []
    for m in args.masters:
        dfm, sources = _load_masters(m)
        concat_master_df_list.append(dfm)
        master_sources_all.extend(sources)
    df_master = pd.concat(concat_master_df_list, ignore_index=True)

    # Merge
    df = pd.merge(df_params, df_master, on="run_id", how="inner").drop_duplicates()

    # Pull out features, coerce to numeric, log-transform some
    feat_cols = _pick_features(df, DEFAULT_PARAM_COLS)
    log_cols = [c.strip() for c in (args.log_cols.split(",") if args.log_cols else [])]
    df, transform_map = _prepare_features(df=df,feat_cols=feat_cols,log_cols=log_cols,base=args.log_base,)
    X = df[feat_cols].to_numpy(dtype=float)

    # Pull out targets
    explicit_targets = [t.strip() for t in args.targets.split(",")] if args.targets else None
    Y, target_meta = _auto_pick_targets(df, explicit_targets)

    # Apply optional cuts on dTdt, dT (create mask)
    mask_cuts = np.ones(len(df), dtype=bool)

    if args.max_dTdt is not None and "dTdt_C_per_Myr" in df.columns:
        v = df["dTdt_C_per_Myr"]
        mask_cuts &= (v.isna() | (v <= args.max_dTdt))

    if args.max_dT is not None and "dT_C" in df.columns:
        v = df["dT_C"]
        mask_cuts &= (v.isna() | (v <= args.max_dT))

    n_cut_only = int((~mask_cuts).sum())

    # Also cut out rows with NaNs (create mask)
    nan_mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    n_nan_only = int((~nan_mask).sum())

    # Apply masks to cut and eliminate NaN rows
    full_mask = mask_cuts & nan_mask
    n_before = len(df)
    X = X[full_mask]; Y = Y[full_mask]
    df_clean = df.loc[full_mask].reset_index(drop=True)
    n_after = X.shape[0]
    dropped_total = n_before - n_after
    print(f"[OK] Dropped by NaNs (post-cut): {n_nan_only}")
    print(f"[OK] Total dropped (cuts + NaNs): {dropped_total}")

    # Standardize
    X_std, X_scaler = _standardize(X)
    Y_std, Y_scaler = _standardize(Y)

    # Split
    train_idx, val_idx = _train_val_split(n_after, args.val_frac, args.seed)

    # Prepare outdir
    outdir = (Path(args.outdir) if Path(args.outdir).is_absolute()
              else (THIS_FILE.parent / args.outdir)).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Save arrays
    np.save(outdir / "X_raw.npy", X)
    np.save(outdir / "Y_raw.npy", Y)
    np.save(outdir / "X_std.npy", X_std)
    np.save(outdir / "Y_std.npy", Y_std)
    np.save(outdir / "train_idx.npy", train_idx)
    np.save(outdir / "val_idx.npy", val_idx)

    # Metadata
    meta = {
        "repo_root": repo_root.as_posix(),
        "params_path": params_path.as_posix(),
        "master_sources": sorted(list(set(master_sources_all))),
        "dropped": {
            "by_cuts": n_cut_only,
            "by_nans": n_nan_only,
            "total": dropped_total
        },
        "feature_cols": feat_cols,
        "feature_transforms": { "log": transform_map },
        "target": target_meta,
        "cuts_max": {
            "dTdt_C_per_Myr": args.max_dTdt,
            "dT_C": args.max_dT,
        },
        "scalers": {
            "X": X_scaler,
            "Y": Y_scaler,
        },
        "split": {
            "val_frac": args.val_frac,
            "seed": args.seed,
            "n_train": int(train_idx.size),
            "n_val": int(val_idx.size),
        },
        "run_ids": df_clean["run_id"].tolist(),
        "cli_args": vars(args)
    }

    dropped_run_ids = df.loc[~full_mask, "run_id"].tolist()
    meta["dropped_run_ids"] = dropped_run_ids

    # If available, carry through obs_depth_km (useful for multi-depth data)
    if "obs_depth_km" in df_clean.columns:
        meta["obs_depth_km"] = [None if pd.isna(x) else float(x) for x in df_clean["obs_depth_km"].tolist()]

    # Save metadata
    with open(outdir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Console summary
    print(f"[OK] Merged params+masters: {n_before} rows → {n_after} after NaN drop and thresh cut.")
    print(f"[OK] Features: {feat_cols}")
    print(f"[OK] Targets ({meta['target']['target_kind']}): {meta['target']['target_cols']}")
    if transform_map:
        print(f"[OK] Log-transformed features ({args.log_base}): {sorted(transform_map.keys())}")
    print(f"[OK] Saved arrays to: {outdir}")
    print(f"[OK] Train/Val sizes: {meta['split']['n_train']}/{meta['split']['n_val']}")
    print(f"[OK] Wrote metadata: {outdir / 'metadata.json'}")


if __name__ == "__main__":
    main()

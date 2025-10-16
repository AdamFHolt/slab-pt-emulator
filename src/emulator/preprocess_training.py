#!/usr/bin/env python3

"""
Preprocess a master file for emulator training.

- Reads params-list.csv and one master_*.csv file.
- Merges on run_id.
- Builds X from known parameter columns (only those present).
- Builds Y automatically:
    * If ['dT_C','dTdt_C_per_Myr'] exist → uses both (target_kind='summary_dT')
    * Else: requires --targets col1,col2,...
- Drops NaNs, applies optional max thresholds, standardizes (z-score),
  and saves arrays + metadata.

Usage:
    python preprocess_training_single.py \
        --params ../../data/params-list.csv \
        --master ../../subd-model-runs/run-outputs/analysis/master_25km_DT1-6.csv \
        --targets dTdt_C_per_Myr \
        --outdir src/emulator/data
"""

from __future__ import annotations
import argparse, json, os, re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np, pandas as pd

# Preferred parameter columns (will use intersection with actual columns)
DEFAULT_PARAM_COLS = ["v_conv", "age_SP", "age_OP", "dip_int", "eta_int", "eta_UM", "eps_trans"]
THIS_FILE = Path(__file__).resolve()
# Repo root two levels above this file: 
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
        if np.issubdtype(dfp["run_id"].dtype, np.number):
            dfp["run_id"] = dfp["run_id"].astype(int).map(lambda i: f"{i:03d}")
        else:
            dfp["run_id"] = dfp["run_id"].astype(str).str.zfill(3)
    return dfp


def _load_master(master_path: str) -> pd.DataFrame:
    """Read a master CSV and annotate depth if found in filename."""
    f = Path(master_path).resolve()
    if not f.exists():
        raise FileNotFoundError(f"Master file not found: {f}")

    df = pd.read_csv(f, dtype={"run_id": str})
    df["run_id"] = df["run_id"].astype(str).str.zfill(3)

    # Annotate a depth if embedded in filename like 'master_25km_...'
    m = re.search(r"master_(\d+)\s*km", f.name.replace("-", ""))
    depth_km = int(m.group(1)) if m else None
    if depth_km is not None:
        df["obs_depth_km"] = depth_km

    df["source_file"] = f.as_posix()
    return df


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

    # 1) Explicit targets provided 
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

    raise ValueError("Could not auto-detect targets. Pass --targets col1,col2,... or ensure [dT_C, dTdt_C_per_Myr] exist.")


def _standardize(arr: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    mu = np.nanmean(arr, axis=0)
    sigma = np.nanstd(arr, axis=0, ddof=0)
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
    parser.add_argument("--master", required=True,
                        help="Path to a single master_*.csv file.")
    parser.add_argument("--targets", default=None,
                        help="Comma-separated target columns to use (overrides auto-detect).")
    parser.add_argument("--max-dTdt", type=float, default=None,
                        help="Drop rows w/ dTdt_C_per_Myr > this val.")
    parser.add_argument("--max-dT", type=float, default=None,
                        help="Drop rows with dT_C > this val.")
    parser.add_argument("--add-thermal-param", action="store_true",
                        help="Include thermal parameter as a feature.")
    parser.add_argument("--add-eta-ratio", action="store_true",
                        help="Include interface viscosity ratio as a feature.")
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

    # Load params and master
    df_params = _load_params(params_path)
    df_master = _load_master(args.master)

    # Merge
    df = pd.merge(df_params, df_master, on="run_id", how="inner").drop_duplicates()

    # Feature prep
    feat_cols = _pick_features(df, DEFAULT_PARAM_COLS)
    log_cols = [c.strip() for c in (args.log_cols.split(",") if args.log_cols else [])]
    df, transform_map = _prepare_features(df=df,feat_cols=feat_cols,log_cols=log_cols,base=args.log_base,)
    # Optional extra features
    extra_feats = []
    if args.add_thermal_param:
        if all(c in df.columns for c in ["v_conv", "age_SP", "dip_int"]):
            df["thermal_param"] = df["v_conv"] * df["age_SP"] * np.sin(np.radians(df["dip_int"]))
            feat_cols.append("thermal_param")
            extra_feats.append("thermalParam")
        else:
            print("[WARN] Missing one or more columns for slab thermal parameter; skipping.")
    if args.add_eta_ratio:
        if all(c in df.columns for c in ["eta_int", "eta_UM"]):
            df["eta_ratio"] = df["eta_int"] / df["eta_UM"]
            feat_cols.append("eta_ratio")
            extra_feats.append("etaRatio")
    # Final feature set
    X = df[feat_cols].to_numpy(dtype=float)

    # Targets
    explicit_targets = [t.strip() for t in args.targets.split(",")] if args.targets else None
    Y, target_meta = _auto_pick_targets(df, explicit_targets)

    # Apply optional cuts 
    mask_cuts = np.ones(len(df), dtype=bool)
    if args.max_dTdt is not None and "dTdt_C_per_Myr" in df.columns:
        v = df["dTdt_C_per_Myr"]
        mask_cuts &= (v.isna() | (v <= args.max_dTdt))
    if args.max_dT is not None and "dT_C" in df.columns:
        v = df["dT_C"]
        mask_cuts &= (v.isna() | (v <= args.max_dT))
    n_cut_only = int((~mask_cuts).sum())

    # Also drop NaNs 
    nan_mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    n_nan_only = int((~nan_mask).sum())
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

    # Save outputs
    base_outdir = Path(args.outdir).resolve()
    m = re.search(r"master_(\d+)\s*km", Path(args.master).name.replace("-", ""))
    depth_str = f"{m.group(1)}km" if m else "unknown_depth"
    # short labels for targets
    def _shorten_target(name: str) -> str:
        name = name.replace("dTdt_C_per_Myr", "dTdt")
        name = name.replace("dT_C", "dT")
        return name
    if args.targets:
        target_list = [_shorten_target(t.strip()) for t in args.targets.split(",")]
    else:
        target_list = [_shorten_target(t) for t in target_meta["target_cols"]]
    target_label = "_".join(target_list)
    if extra_feats:
        target_label += "_" + "_".join(extra_feats)
    subdir_name = f"{depth_str}_{target_label}"
    outdir = base_outdir / subdir_name
    outdir.mkdir(parents=True, exist_ok=True)
    # save
    print(f"[OK] Output directory set to: {outdir}")
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
        "master_source": Path(args.master).resolve().as_posix(),
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

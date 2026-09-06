#!/usr/bin/env python3
"""Collate the per-time profile-PCA quality reports of the 0.5-10 Myr series.

Reads every ``profile_pca_quality.json`` under a models root and writes one
table: held-out (val) reconstructed-profile RMSE in degrees C per output time,
the PCA-truncation floor at the same time, and the depth at which the held-out
RMSE is worst.

Usage:
    summarize_10myr_quality.py \
        --models-root src/emulator/models/profile_pca_10myr/const-vc/runs \
        --out plots/qc-emulator/profile-pca/10myr/const-vc_profile_rmse_by_time.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]


def time_from_name(name: str) -> float:
    m = re.search(r"_t([0-9p]+)Myr", name)
    if not m:
        return float("nan")
    return float(m.group(1).replace("p", "."))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-root", required=True)
    ap.add_argument("--model-tag", default="gp_m25")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.models_root)
    if not root.is_absolute():
        root = REPO_ROOT / root

    rows = []
    for ds_dir in sorted(root.iterdir()):
        qpath = ds_dir / args.model_tag / "profile_pca_quality.json"
        if not qpath.exists():
            continue
        q = json.loads(qpath.read_text())
        depth = np.asarray(q["depth_grid_km"], float)
        rec: dict[str, object] = {
            "dataset": ds_dir.name,
            "time_myr": time_from_name(ds_dir.name),
        }
        for split in ("train", "val"):
            m = q["metrics"].get(split)
            if not m:
                continue
            emu = m["profile_space"]["emulator_reconstruction"]
            pca = m["profile_space"]["pca_truncation_baseline"]
            rbd = np.asarray(emu["rmse_by_depth"], float)
            j = int(np.argmax(rbd))
            rec[f"n_{split}"] = m["n_rows"]
            rec[f"{split}_profile_rmse_C"] = emu["rmse"]
            rec[f"{split}_profile_mae_C"] = emu["mae"]
            rec[f"{split}_profile_r2"] = emu["r2"]
            rec[f"{split}_per_run_rmse_p95_C"] = emu["per_run_rmse"]["p95"]
            rec[f"{split}_per_run_rmse_max_C"] = emu["per_run_rmse"]["max"]
            rec[f"{split}_worst_depth_km"] = float(depth[j])
            rec[f"{split}_worst_depth_rmse_C"] = float(rbd[j])
            rec[f"{split}_pca_floor_rmse_C"] = pca["rmse"]
        rows.append(rec)

    df = pd.DataFrame(rows).sort_values("time_myr").reset_index(drop=True)
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[OK] wrote {out}")

    cols = ["time_myr", "n_train", "n_val", "val_profile_rmse_C", "val_worst_depth_km",
            "val_worst_depth_rmse_C", "val_pca_floor_rmse_C", "val_profile_r2"]
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(df[[c for c in cols if c in df.columns]].to_string(index=False,
              float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

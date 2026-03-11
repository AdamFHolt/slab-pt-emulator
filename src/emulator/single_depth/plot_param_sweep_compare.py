#!/usr/bin/env python3
"""
Compare param-sweep model fits (GP/RF variants) for ONE dataset (depth + variant)
using metrics in report.json, and make quick comparison plots + a CSV table.

Expected layout (your example):
  models/param-sweep/<suite>/<data_name>/<run_tag>/report.json

Outputs (default):
  /home/holt/Projects/SlabPT-emulator/plots/qc-emulator/<suite>/param-sweep/<data_name>/
    sweep_bar_val_rmse.png
    sweep_bar_val_r2.png
    sweep_scatter_val_rmse_vs_r2.png
    sweep_table.csv

Example:
  python plot_param_sweep_compare.py --suite const-vc --data-name 40km_dTdt_thermalParam

Options:
  --metric-split val|train  (default val)
  --sort-by val_rmse|val_r2|name  (default val_rmse)
  --top N to limit number of bars for readability (default 24)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PLOTS_DIR_DEFAULT = Path("/home/holt/Projects/SlabPT-emulator/plots/qc-emulator")


# -------------------------- helpers

def _load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        return json.load(f)

def _safe_get(d: Dict[str, Any], keys: List[str], default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _discover_runs(dataset_dir: Path) -> List[Path]:
    if not dataset_dir.exists():
        return []
    return sorted([p for p in dataset_dir.iterdir() if p.is_dir()])

def _shorten_run_tag(tag: str) -> str:
    """
    Make long directory names readable on plots.
    Examples:
      gp_rbf_r15_ls1.0_b1e-3-1e3_n3e-3_nb1e-6-1.0_a1e-6  -> gp_rbf r15 n3e-3
      rf_t600_dNone -> rf t600 dNone
    """
    if tag.startswith("rf_"):
        # rf_t600_dNone
        m = re.match(r"rf_t(?P<t>\d+)_d(?P<d>.+)$", tag)
        if m:
            return f"rf t{m.group('t')} d{m.group('d')}"
        return tag.replace("_", " ")

    if tag.startswith("gp_"):
        # gp_rbf_r15_ls1.0_b1e-3-1e3_n3e-3_nb1e-6-1.0_a1e-6
        # keep: kernel + r + n (and maybe alpha)
        parts = tag.split("_")
        # parts like: ['gp','rbf','r15','ls1.0','b1e-3-1e3','n3e-3','nb1e-6-1.0','a1e-6']
        kernel = parts[1] if len(parts) > 1 else "gp"
        r = next((p for p in parts if p.startswith("r")), None)
        n = next((p for p in parts if p.startswith("n") and not p.startswith("nb")), None)
        a = next((p for p in parts if p.startswith("a")), None)
        keep = [f"gp_{kernel}"]
        if r: keep.append(r)
        if n: keep.append(n)
        if a: keep.append(a)
        return " ".join(keep).replace("gp_", "gp ")

    return tag.replace("_", " ")

def _read_metrics(report_path: Path) -> Dict[str, Any]:
    """
    Returns:
      {
        "train_rmse": float|nan, "train_r2": float|nan,
        "val_rmse": float|nan,   "val_r2": float|nan,
        "model_type": str|None
      }
    """
    rep = _load_json(report_path)

    tr = _safe_get(rep, ["metrics", "train", "_macro_avg"], default={}) or {}
    va = _safe_get(rep, ["metrics", "val", "_macro_avg"], default={}) or {}

    return dict(
        model_type=_safe_get(rep, ["model_type"], default=None),
        train_rmse=float(tr.get("rmse", np.nan)) if tr else np.nan,
        train_r2=float(tr.get("r2", np.nan)) if tr else np.nan,
        val_rmse=float(va.get("rmse", np.nan)) if va else np.nan,
        val_r2=float(va.get("r2", np.nan)) if va else np.nan,
    )

def _ensure_outdir(base: Path, suite: str, data_name: str) -> Path:
    outdir = base / suite / "param-sweep" / data_name
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def _barplot(df: pd.DataFrame, x_col: str, y_col: str, title: str, outpath: Path, top: int):
    dff = df.copy()

    # Select top N by ordering already applied outside, but keep NaNs out
    dff = dff[np.isfinite(dff[y_col].to_numpy(float))]
    if top is not None and top > 0:
        dff = dff.head(top)

    if dff.empty:
        print(f"[WARN] Nothing to plot for {y_col} (no finite values).")
        return

    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.35 * len(dff))))
    ax.barh(dff[x_col], dff[y_col])
    ax.invert_yaxis()
    ax.grid(True, axis="x", ls=":", alpha=0.35)
    ax.set_xlabel(y_col)
    ax.set_ylabel("model run")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {outpath}")

def _scatter(df: pd.DataFrame, title: str, outpath: Path):
    dff = df.copy()
    dff = dff[np.isfinite(dff["val_rmse"].to_numpy(float)) & np.isfinite(dff["val_r2"].to_numpy(float))]
    if dff.empty:
        print("[WARN] Nothing to plot for scatter (need finite val_rmse and val_r2).")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(dff["val_rmse"], dff["val_r2"], s=35, alpha=0.85)

    # annotate best few (lowest rmse)
    best = dff.sort_values("val_rmse", ascending=True).head(6)
    for _, row in best.iterrows():
        ax.text(row["val_rmse"], row["val_r2"], str(row["label"]), fontsize=8, ha="left", va="bottom")

    ax.grid(True, ls=":", alpha=0.35)
    ax.set_xlabel("val_rmse")
    ax.set_ylabel("val_r2")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {outpath}")

def main():
    p = argparse.ArgumentParser(description="Compare param-sweep fits for one dataset.")
    p.add_argument("--suite", required=True, choices=["const-vc", "ramped-vc"])
    p.add_argument("--data-name", required=True,
                   help="Dataset folder name, e.g. 40km_dTdt or 40km_dTdt_thermalParam")
    p.add_argument("--sweep-root", default=str(Path("models/param-sweep")),
                   help="Root for sweep models (default: models/param-sweep)")
    p.add_argument("--outdir", default=str(PLOTS_DIR_DEFAULT),
                   help=f"Base plots directory (default: {PLOTS_DIR_DEFAULT})")

    p.add_argument("--metric-split", choices=["val", "train"], default="val",
                   help="Which split to emphasize in sorting and titles.")
    p.add_argument("--sort-by", choices=["val_rmse", "val_r2", "train_rmse", "train_r2", "name"], default="val_rmse",
                   help="How to sort runs for bar plots.")
    p.add_argument("--top", type=int, default=24,
                   help="Max number of runs to show in bar charts (for readability). Use 0 for all.")
    args = p.parse_args()

    suite = args.suite
    data_name = args.data_name
    sweep_root = Path(args.sweep_root).resolve()
    dataset_dir = sweep_root / suite / data_name

    runs = _discover_runs(dataset_dir)
    if not runs:
        raise SystemExit(f"[ERR] No sweep runs found under: {dataset_dir}")

    rows: List[Dict[str, Any]] = []
    for run_dir in runs:
        report_path = run_dir / "report.json"
        if not report_path.exists():
            print(f"[WARN] Missing report.json: {report_path}")
            continue

        m = _read_metrics(report_path)
        rows.append(dict(
            name=run_dir.name,
            label=_shorten_run_tag(run_dir.name),
            path=str(run_dir),
            **m
        ))

    if not rows:
        raise SystemExit(f"[ERR] Found run dirs, but no readable report.json files under: {dataset_dir}")

    df = pd.DataFrame(rows)

    # sorting
    if args.sort_by == "name":
        df = df.sort_values("name")
    else:
        asc = True
        if args.sort_by.endswith("_r2"):
            asc = False
        df = df.sort_values(args.sort_by, ascending=asc, na_position="last")

    out_base = Path(args.outdir).resolve()
    outdir = _ensure_outdir(out_base, suite, data_name)

    # save table (full)
    table_path = outdir / "sweep_table.csv"
    df.to_csv(table_path, index=False)
    print(f"[OK] Wrote: {table_path}")

    # bar plots (val)
    top = None if args.top == 0 else args.top

    _barplot(
        df=df,
        x_col="label",
        y_col="val_rmse",
        title=f"{suite} | {data_name} | sweep comparison (VAL RMSE) | sorted by {args.sort_by}",
        outpath=outdir / "sweep_bar_val_rmse.png",
        top=top,
    )

    _barplot(
        df=df,
        x_col="label",
        y_col="val_r2",
        title=f"{suite} | {data_name} | sweep comparison (VAL R²) | sorted by {args.sort_by}",
        outpath=outdir / "sweep_bar_val_r2.png",
        top=top,
    )

    # scatter (val rmse vs r2)
    _scatter(
        df=df,
        title=f"{suite} | {data_name} | VAL: RMSE vs R²",
        outpath=outdir / "sweep_scatter_val_rmse_vs_r2.png",
    )


if __name__ == "__main__":
    main()

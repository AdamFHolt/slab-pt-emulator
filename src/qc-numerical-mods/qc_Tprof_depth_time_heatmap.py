#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TPROF_RE = re.compile(r"^Tprof_(\d+)\.csv$")
RUN_RE = re.compile(r"run_(\d+)$")


def collect_run_dirs(glob_pattern: str) -> list[Path]:
    parents = {Path(p).resolve().parent for p in glob_pattern and map(str, Path().glob(glob_pattern))}
    if parents:
        return sorted(parents)

    paths = [Path(p).resolve() for p in map(str, Path().glob(glob_pattern))]
    if not paths:
        return []
    return sorted({p.parent for p in paths})


def list_tprof_files(run_dir: Path) -> list[Path]:
    files = []
    for p in run_dir.glob("Tprof_*.csv"):
        m = TPROF_RE.match(p.name)
        if m:
            files.append((int(m.group(1)), p))
    return [p for _, p in sorted(files, key=lambda x: x[0])]


def run_label(run_dir: Path) -> str:
    m = RUN_RE.search(run_dir.name)
    if m:
        return f"run_{int(m.group(1)):03d}"
    return run_dir.name


def build_grid(tprof_files: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    for p in tprof_files:
        df = pd.read_csv(p)
        need = {"time_Myr", "depth_km", "T_C"}
        if not need.issubset(df.columns):
            raise ValueError(f"{p} missing required columns {sorted(need)}")
        rows.append(df[["time_Myr", "depth_km", "T_C"]].copy())

    all_df = pd.concat(rows, ignore_index=True)
    all_df["time_Myr"] = pd.to_numeric(all_df["time_Myr"], errors="coerce")
    all_df["depth_km"] = pd.to_numeric(all_df["depth_km"], errors="coerce")
    all_df["T_C"] = pd.to_numeric(all_df["T_C"], errors="coerce")
    all_df = all_df.dropna(subset=["time_Myr", "depth_km"])

    times = np.sort(all_df["time_Myr"].unique())
    depths = np.sort(all_df["depth_km"].unique())

    grid = (
        all_df.pivot_table(index="depth_km", columns="time_Myr", values="T_C", aggfunc="mean")
        .reindex(index=depths, columns=times)
        .to_numpy()
    )
    return times, depths, grid


def plot_heatmap(times: np.ndarray, depths: np.ndarray, grid: np.ndarray, out_png: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(
        grid,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        extent=[times.min(), times.max(), depths.max(), depths.min()],
        cmap="coolwarm",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("T (C)")
    ax.set_xlabel("time (Myr)")
    ax.set_ylabel("depth (km)")
    ax.set_title(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-run depth-time heatmaps from Tprof_*.csv files.")
    ap.add_argument(
        "--glob",
        required=True,
        help='Glob to run Tprof files, e.g. "subd-model-runs/const-vc/analysis/run_*/Tprof_*.csv"',
    )
    ap.add_argument("--out-dir", required=True, help="Directory to write one heatmap PNG per run.")
    ap.add_argument("--max-runs", type=int, default=None, help="Optional limit for quick test runs.")
    args = ap.parse_args()

    files = [Path(p).resolve() for p in map(str, Path().glob(args.glob))]
    if not files:
        raise SystemExit(f"No files matched glob: {args.glob}")

    run_dirs = sorted({p.parent for p in files})
    if args.max_runs is not None and args.max_runs > 0:
        run_dirs = run_dirs[: args.max_runs]

    out_dir = Path(args.out_dir).resolve()
    made = 0
    skipped = 0
    for run_dir in run_dirs:
        tprof_files = list_tprof_files(run_dir)
        if len(tprof_files) < 2:
            skipped += 1
            continue
        times, depths, grid = build_grid(tprof_files)
        if grid.size == 0:
            skipped += 1
            continue
        label = run_label(run_dir)
        out_png = out_dir / f"{label}_Tprof_depth-time.png"
        plot_heatmap(times, depths, grid, out_png, title=f"{label}: T(depth, time)")
        made += 1

    print(f"[OK] wrote {made} heatmaps to {out_dir}")
    if skipped:
        print(f"[WARN] skipped {skipped} runs (insufficient/invalid Tprof files)")


if __name__ == "__main__":
    main()

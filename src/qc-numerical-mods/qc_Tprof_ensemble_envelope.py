#!/usr/bin/env python3
import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_RE = re.compile(r"run_(\d+)")


def run_id_from_path(p: Path) -> str:
    m = RUN_RE.search(str(p))
    if m:
        return f"run_{int(m.group(1)):03d}"
    return p.parent.name


def read_tprof(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"time_Myr", "depth_km", "T_C"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} missing required columns {sorted(need)}")
    out = df[["time_Myr", "depth_km", "T_C"]].copy()
    out["time_Myr"] = pd.to_numeric(out["time_Myr"], errors="coerce")
    out["depth_km"] = pd.to_numeric(out["depth_km"], errors="coerce")
    out["T_C"] = pd.to_numeric(out["T_C"], errors="coerce")
    out = out.dropna(subset=["time_Myr", "depth_km", "T_C"])
    return out


def choose_profile_for_time(run_files: list[Path], time_req: float, tol_myr: float) -> tuple[Path, float] | None:
    best = None
    for p in run_files:
        df = read_tprof(p)
        if df.empty:
            continue
        t = float(df["time_Myr"].iloc[0])
        dt = abs(t - time_req)
        if best is None or dt < best[2]:
            best = (p, t, dt)
    if best is None or best[2] > tol_myr:
        return None
    return best[0], best[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Ensemble T(depth) envelopes across runs at selected times.")
    ap.add_argument(
        "--glob",
        required=True,
        help='Glob for Tprof CSVs, e.g. "subd-model-runs/const-vc/analysis/run_*/Tprof_*.csv"',
    )
    ap.add_argument("--times", type=float, nargs="+", default=[0.5, 2.5, 5.0], help="Requested times (Myr).")
    ap.add_argument("--tol-myr", type=float, default=0.2, help="Max time mismatch to accept per run.")
    ap.add_argument("--out", required=True, help="Output prefix, e.g. plots/.../Tprof_ensemble_envelope")
    ap.add_argument("--dpi", type=int, default=220)
    args = ap.parse_args()

    files = [Path(p).resolve() for p in glob.glob(args.glob)]
    if not files:
        raise SystemExit(f"No files matched: {args.glob}")

    run_to_files: dict[str, list[Path]] = {}
    for p in files:
        run = run_id_from_path(p)
        run_to_files.setdefault(run, []).append(p)

    for run in run_to_files:
        run_to_files[run] = sorted(run_to_files[run])

    rows = []
    for time_req in args.times:
        for run_id, run_files in run_to_files.items():
            chosen = choose_profile_for_time(run_files, float(time_req), args.tol_myr)
            if chosen is None:
                continue
            p, t_sel = chosen
            df = read_tprof(p)
            df["run_id"] = run_id
            df["time_req_Myr"] = float(time_req)
            df["time_sel_Myr"] = float(t_sel)
            rows.append(df[["run_id", "time_req_Myr", "time_sel_Myr", "depth_km", "T_C"]])

    if not rows:
        raise SystemExit("No profiles matched requested times within tolerance.")

    all_df = pd.concat(rows, ignore_index=True)
    out_prefix = Path(args.out).resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    times_req = [float(t) for t in args.times]
    fig, axes = plt.subplots(1, len(times_req), figsize=(4.7 * len(times_req), 5.0), constrained_layout=True)
    if len(times_req) == 1:
        axes = [axes]

    for ax, t in zip(axes, times_req):
        sub = all_df[np.isclose(all_df["time_req_Myr"], t)].copy()
        if sub.empty:
            ax.set_title(f"~{t:g} Myr (no data)")
            ax.axis("off")
            continue

        stats = (
            sub.groupby("depth_km")["T_C"]
            .agg(
                t05=lambda x: np.nanpercentile(x, 5),
                t50=lambda x: np.nanpercentile(x, 50),
                t95=lambda x: np.nanpercentile(x, 95),
            )
            .reset_index()
            .sort_values("depth_km")
        )

        z = stats["depth_km"].to_numpy()
        p05 = stats["t05"].to_numpy()
        p50 = stats["t50"].to_numpy()
        p95 = stats["t95"].to_numpy()

        ax.fill_betweenx(z, p05, p95, alpha=0.25, label="5-95%")
        ax.plot(p50, z, linewidth=2.2, color="black", label="median")

        n_runs = int(sub["run_id"].nunique())
        t_med = float(np.nanmedian(sub["time_sel_Myr"].to_numpy()))
        ax.set_title(f"time={t_med:.2f} Myr (n={n_runs})")
        ax.set_xlabel("Temperature (C)")
        ax.set_ylabel("Depth (km)")
        ax.grid(alpha=0.25, linestyle=":")
        ax.invert_yaxis()

    axes[0].legend(loc="best")
    fig.savefig(f"{out_prefix}.png", dpi=args.dpi, bbox_inches="tight")
    print(f"[OK] wrote {out_prefix}.png")


if __name__ == "__main__":
    main()

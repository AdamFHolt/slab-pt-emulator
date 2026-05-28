#!/usr/bin/env python3
"""Plot Sobol indices versus depth for one suite.

Reads every per-emulator Sobol JSON for a suite (written by
``compute_sobol_sensitivity.py``), sorts them by observation depth, and draws
two stacked panels — first-order S1 vs depth and total-effect ST vs depth — with
one line per parameter. This is the headline "which parameter controls cooling
at which depth" figure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _sobol_io import default_label_map

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[4]

# Stable per-parameter colors (Paul Tol bright/medium-contrast palette).
PARAM_COLORS = {
    "v_conv": "#4477AA",
    "age_SP": "#EE6677",
    "age_OP": "#228833",
    "dip_int": "#CCBB44",
    "eta_UM": "#AA3377",
    "t_conv": "#66CCEE",
    "v_conv_over_tconv": "#BBBBBB",
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot Sobol S1/ST vs depth for one suite.")
    p.add_argument("--suite", required=True, help="Suite name, e.g. const-vc.")
    p.add_argument("--model-tag", default="gp_m25", help="Model subdirectory tag.")
    p.add_argument(
        "--data-suffix",
        default="dTdt",
        help="Dataset suffix selecting the emulator family (e.g. dTdt).",
    )
    p.add_argument(
        "--sobol-dir",
        default=None,
        help="Directory holding the Sobol JSONs. Defaults to "
        "plots/science-emulator/single_depth/<suite>/sobol.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    sobol_dir = (
        Path(args.sobol_dir).resolve()
        if args.sobol_dir
        else (REPO_ROOT / "plots" / "science-emulator" / "single_depth" / args.suite / "sobol")
    )
    if not sobol_dir.exists():
        raise FileNotFoundError(f"Sobol directory not found: {sobol_dir}")

    # Match exactly "<n>km_<suffix>_<tag>_sobol.json"; exclude thermalParam etc.
    pattern = f"*km_{args.data_suffix}_{args.model_tag}_sobol.json"
    json_paths = sorted(sobol_dir.glob(pattern))
    if not json_paths:
        raise FileNotFoundError(f"No Sobol JSONs matching {pattern} in {sobol_dir}")

    records = []
    feature_cols: list[str] | None = None
    for jp in json_paths:
        data = json.loads(jp.read_text())
        if data.get("depth_km") is None:
            continue
        if feature_cols is None:
            feature_cols = list(data["feature_cols"])
        elif list(data["feature_cols"]) != feature_cols:
            # Skip emulators whose feature set differs from the rest of the family.
            continue
        records.append(data)

    if not records or feature_cols is None:
        raise RuntimeError("No usable Sobol JSON records with consistent features.")

    records.sort(key=lambda d: d["depth_km"])
    depths = np.array([d["depth_km"] for d in records], dtype=float)
    s1 = np.array([d["S1"] for d in records], dtype=float)  # (n_depth, D)
    st = np.array([d["ST"] for d in records], dtype=float)

    label_map = default_label_map()

    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
    for ax, mat, title in (
        (axes[0], s1, "First-order $S_1$ (main effect)"),
        (axes[1], st, "Total-effect $S_T$ (incl. interactions)"),
    ):
        for j, feat in enumerate(feature_cols):
            ax.plot(
                depths, mat[:, j], marker="o", ms=4, lw=1.8,
                color=PARAM_COLORS.get(feat, None), label=label_map.get(feat, feat),
            )
        ax.set_ylabel("Sobol index", fontsize=11)
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=12)

    axes[1].set_xlabel("Depth (km)", fontsize=11)
    axes[0].legend(loc="upper right", fontsize=9, ncol=2, framealpha=0.9)
    fig.suptitle(
        f"{args.suite}  —  Sobol sensitivity of cooling rate vs depth "
        f"({args.model_tag})",
        fontsize=13,
    )

    out_path = sobol_dir / f"{args.suite}_sobol_vs_depth_{args.model_tag}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}  ({len(records)} depths)")


if __name__ == "__main__":
    main()

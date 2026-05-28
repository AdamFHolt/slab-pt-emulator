#!/usr/bin/env python3
"""Plot per-emulator Sobol sensitivity from a computed JSON.

Produces a two-panel figure for one single-depth emulator:
  - top: ranked first-order (S1) and total-effect (ST) bars with conf whiskers
  - bottom: second-order (S2) pairwise-interaction heatmap

Reads the JSON written by ``compute_sobol_sensitivity.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _sobol_io import default_label_map, target_label

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[4]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot per-emulator Sobol indices from JSON.")
    p.add_argument("--suite", required=True, help="Suite name, e.g. const-vc.")
    p.add_argument("--data-name", required=True, help="Dataset name, e.g. 40km_dTdt.")
    p.add_argument("--model-tag", default="gp_m25", help="Model subdirectory tag.")
    p.add_argument(
        "--sobol-dir",
        default=None,
        help="Directory holding the Sobol JSON. Defaults to "
        "plots/science-emulator/single_depth/<suite>/sobol.",
    )
    return p


def _short_label(name: str, label_map: dict[str, str]) -> str:
    # Reuse the full math label for axes; the S2 heatmap ticks use it too.
    return label_map.get(name, name)


def main() -> None:
    args = _build_parser().parse_args()

    sobol_dir = (
        Path(args.sobol_dir).resolve()
        if args.sobol_dir
        else (REPO_ROOT / "plots" / "science-emulator" / "single_depth" / args.suite / "sobol")
    )
    json_path = sobol_dir / f"{args.data_name}_{args.model_tag}_sobol.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Sobol JSON not found: {json_path}")

    data = json.loads(json_path.read_text())
    feats = list(data["feature_cols"])
    label_map = default_label_map()
    labels = [_short_label(f, label_map) for f in feats]

    s1 = np.asarray(data["S1"], dtype=float)
    st = np.asarray(data["ST"], dtype=float)
    s1_conf = np.asarray(data["S1_conf"], dtype=float)
    st_conf = np.asarray(data["ST_conf"], dtype=float)
    s2 = np.asarray(data["S2"], dtype=float)

    # Rank by total effect (descending).
    order = np.argsort(st)[::-1]
    ord_labels = [labels[i] for i in order]
    y = np.arange(len(order))
    h = 0.38

    fig = plt.figure(figsize=(11, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15])

    # --- Top: grouped horizontal bars (S1 and ST) ---
    ax = fig.add_subplot(gs[0, 0])
    ax.barh(
        y + h / 2, st[order], height=h, color="#4477AA", label="ST (total)",
        xerr=st_conf[order], error_kw={"elinewidth": 1.0, "ecolor": "#22334d"}, zorder=3,
    )
    ax.barh(
        y - h / 2, s1[order], height=h, color="#99CCEE", label="S1 (first-order)",
        xerr=s1_conf[order], error_kw={"elinewidth": 1.0, "ecolor": "#3b6680"}, zorder=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(ord_labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Sobol index (fraction of output variance)", fontsize=11)
    ax.set_xlim(left=0.0)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    tgt = target_label(data["target_col"])
    ax.set_title(
        f"{args.suite}  {args.data_name}  —  Sobol sensitivity of {tgt}", fontsize=13,
    )
    info = (
        f"depth = {data['depth_km']:.0f} km\n"
        f"emulator val. $R^2$ = {data['val_r2']:.3f}\n"
        f"$N$ evals = {data['n_evals']}"
    )
    # Anchor the info box in the mid-right region (empty for all but the top
    # bar) so it never collides with the lower-right legend.
    ax.text(
        0.985, 0.5, info, transform=ax.transAxes, ha="right", va="center", fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
    )

    # --- Bottom: second-order interaction heatmap ---
    ax2 = fig.add_subplot(gs[1, 0])
    s2_sym = np.array(s2, dtype=float)
    # SALib returns S2 in the upper triangle (lower triangle + diagonal NaN).
    # Symmetrize for display and blank the diagonal.
    iu = np.triu_indices_from(s2_sym, k=1)
    full = np.full_like(s2_sym, np.nan)
    full[iu] = s2_sym[iu]
    full.T[iu] = s2_sym[iu]
    vmax = np.nanmax(np.abs(full)) if np.isfinite(full).any() else 1.0
    vmax = max(vmax, 1e-6)
    masked = np.ma.masked_invalid(full)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#dddddd")
    im = ax2.imshow(masked, cmap=cmap, vmin=-vmax, vmax=vmax)
    ax2.set_xticks(range(len(feats)))
    ax2.set_yticks(range(len(feats)))
    ax2.set_xticklabels(labels, fontsize=10, rotation=20, ha="right")
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_title("Second-order interactions $S_2$", fontsize=12)
    for i in range(len(feats)):
        for j in range(len(feats)):
            if np.isfinite(full[i, j]):
                ax2.text(
                    j, i, f"{full[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="black",
                )
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("$S_2$", fontsize=10)

    out_path = sobol_dir / f"{args.data_name}_{args.model_tag}_sobol.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()

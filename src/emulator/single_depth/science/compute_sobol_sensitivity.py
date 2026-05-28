#!/usr/bin/env python3
"""Compute Sobol sensitivity indices for one single-depth emulator.

Samples a trained single-depth GP (or RF) emulator heavily over the central
quantile range of its training inputs and computes variance-based Sobol indices
(first-order S1, total-effect ST, and second-order S2). Results are written to a
JSON file consumed by the Sobol plotting scripts.

Example:
    python3 compute_sobol_sensitivity.py --suite const-vc --data-name 40km_dTdt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from SALib import ProblemSpec

from _sobol_io import (
    build_problem,
    depth_from_data_name,
    load_dataset_bundle,
    predict_raw,
)

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[4]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compute Sobol sensitivity indices (S1, ST, S2) for one single-depth "
            "emulator and write them to JSON."
        )
    )
    p.add_argument("--suite", required=True, help="Suite name, e.g. const-vc or ramped-vc.")
    p.add_argument("--data-name", required=True, help="Dataset name, e.g. 40km_dTdt.")
    p.add_argument("--model-tag", default="gp_m25", help="Model subdirectory tag.")
    p.add_argument(
        "--data-root",
        default=str(REPO_ROOT / "src" / "emulator" / "data" / "single_depth"),
        help="Root containing single-depth dataset bundles grouped by suite.",
    )
    p.add_argument(
        "--models-root",
        default=str(REPO_ROOT / "src" / "emulator" / "models" / "single_depth"),
        help="Root containing single-depth trained models grouped by suite.",
    )
    p.add_argument(
        "--outdir",
        default=None,
        help=(
            "Output directory. Defaults to "
            "plots/science-emulator/single_depth/<suite>/sobol."
        ),
    )
    p.add_argument(
        "--n-base",
        type=int,
        default=1024,
        help="Saltelli base sample size N (use a power of two). Total model evals "
        "= N*(2D+2) with second-order indices enabled.",
    )
    p.add_argument(
        "--range-quantiles",
        type=float,
        nargs=2,
        default=(0.01, 0.99),
        metavar=("QLOW", "QHIGH"),
        help="Central quantile range of the training inputs used as the Sobol box.",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed for sampling/analysis.")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    q_low, q_high = args.range_quantiles
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError("--range-quantiles must satisfy 0 <= low < high <= 1")
    if args.n_base < 16:
        raise ValueError("--n-base should be at least 16 (use a power of two).")

    data_dir = Path(args.data_root).resolve() / args.suite / "runs" / args.data_name
    model_dir = (
        Path(args.models_root).resolve()
        / args.suite
        / "runs"
        / args.data_name
        / args.model_tag
    )
    outdir = (
        Path(args.outdir).resolve()
        if args.outdir
        else (REPO_ROOT / "plots" / "science-emulator" / "single_depth" / args.suite / "sobol")
    )
    outdir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    bundle = load_dataset_bundle(data_dir)
    report = json.loads((model_dir / "report.json").read_text())
    model = joblib.load(model_dir / "model.joblib")

    x_train = bundle["x_raw"][bundle["train_idx"]]
    x_mean, x_std = bundle["x_mean"], bundle["x_std"]
    y_mean, y_std = bundle["y_mean"], bundle["y_std"]
    feature_cols = list(bundle["meta"]["feature_cols"])
    target_col = report["target_cols"][0]

    problem = build_problem(feature_cols, x_train, q_low, q_high)

    sp = ProblemSpec(problem)
    sp.sample_sobol(args.n_base, calc_second_order=True, seed=args.seed)
    sp.evaluate(
        lambda X: predict_raw(model, X, x_mean, x_std, y_mean, y_std)
    )
    sp.analyze_sobol(calc_second_order=True, seed=args.seed)
    res = sp.analysis

    val_macro = report["metrics"]["val"]["_macro_avg"]
    depth_km = depth_from_data_name(args.data_name, bundle["meta"])

    payload = {
        "suite": args.suite,
        "data_name": args.data_name,
        "model_tag": args.model_tag,
        "target_col": target_col,
        "depth_km": depth_km,
        "feature_cols": feature_cols,
        "bounds": problem["bounds"],
        "n_base": args.n_base,
        "n_evals": int(sp.samples.shape[0]),
        "range_quantiles": [q_low, q_high],
        "seed": args.seed,
        "val_r2": float(val_macro["r2"]),
        "val_rmse": float(val_macro["rmse"]),
        "S1": np.asarray(res["S1"], dtype=float).tolist(),
        "S1_conf": np.asarray(res["S1_conf"], dtype=float).tolist(),
        "ST": np.asarray(res["ST"], dtype=float).tolist(),
        "ST_conf": np.asarray(res["ST_conf"], dtype=float).tolist(),
        "S2": np.asarray(res["S2"], dtype=float).tolist(),
        "S2_conf": np.asarray(res["S2_conf"], dtype=float).tolist(),
    }

    out_path = outdir / f"{args.data_name}_{args.model_tag}_sobol.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[OK] wrote {out_path}")
    order = np.argsort(payload["ST"])[::-1]
    ranked = ", ".join(
        f"{feature_cols[i]}: ST={payload['ST'][i]:.3f} S1={payload['S1'][i]:.3f}"
        for i in order
    )
    print(f"[INFO] {args.suite}/{args.data_name} (n_evals={payload['n_evals']}) {ranked}")


if __name__ == "__main__":
    main()

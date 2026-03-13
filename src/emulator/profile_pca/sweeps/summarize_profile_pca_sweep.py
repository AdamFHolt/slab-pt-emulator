#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLOTS_ROOT_DEFAULT = REPO_ROOT / "plots" / "qc-emulator" / "profile-pca" / "pca-sweep"


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return obj


def _parse_dataset_name(dataset_name: str) -> dict[str, Any]:
    # Sweep dataset names are intentionally structured so the summary table can
    # recover the sweep knobs directly from the directory name.
    #
    # Example:
    #   profileT_pca_t3Myr_k8_raw
    #   profileT_pca_t3Myr_k8_whitened
    parts = dataset_name.split("_")
    out: dict[str, Any] = {"dataset_name": dataset_name}
    for part in parts:
        if part.startswith("t") and part.endswith("Myr") and len(part) > 4:
            out["time_label"] = part[1:-3]
        elif part.startswith("k") and part[1:].isdigit():
            out["k"] = int(part[1:])
    if parts:
        out["score_space_tag"] = parts[-1]
    return out


def _discover_reports(models_root: Path, suites: list[str]) -> list[Path]:
    out: list[Path] = []
    for suite in suites:
        suite_root = models_root / suite / "pca_sweep"
        if not suite_root.exists():
            continue
        out.extend(sorted(suite_root.glob("*/gp_m25/profile_pca_quality.json")))
    return out


def _row_from_report(path: Path) -> dict[str, Any]:
    report = _load_json(path)
    meta = _parse_dataset_name(str(report["dataset_name"]))
    val = report["metrics"]["val"]

    return {
        "suite": report["suite"],
        "dataset_name": report["dataset_name"],
        "time_label": meta.get("time_label"),
        "k": meta.get("k"),
        "score_space": report.get("score_space"),
        "model_type": report.get("model_type"),
        "model_kernel": report.get("model_kernel"),
        "val_profile_rmse": float(val["profile_space"]["emulator_reconstruction"]["rmse"]),
        "val_profile_p95_rmse": float(val["profile_space"]["emulator_reconstruction"]["per_run_rmse"]["p95"]),
        "val_score_rmse": float(val["score_space"]["_macro_avg"]["rmse"]),
        "val_score_r2": float(val["score_space"]["_macro_avg"]["r2"]),
        "val_pca_only_profile_rmse": float(val["profile_space"]["pca_truncation_baseline"]["rmse"]),
        "report_path": path.as_posix(),
    }


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    cols = [
        "suite",
        "dataset_name",
        "k",
        "score_space",
        "val_profile_rmse",
        "val_profile_p95_rmse",
        "val_score_rmse",
        "val_score_r2",
        "val_pca_only_profile_rmse",
    ]
    dff = df.loc[:, cols].copy()
    headers = list(dff.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in dff.iterrows():
        values = [str(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize profile-PCA sweep reports into ranked tables.")
    ap.add_argument("--models-root", default="src/emulator/models/profile_pca")
    ap.add_argument("--suites", default="const-vc,ramped-vc", help="Comma-separated suite list.")
    ap.add_argument("--dataset-pattern", default="profileT_pca_t3Myr", help="Keep only dataset names containing this text.")
    ap.add_argument("--outdir", default=str(PLOTS_ROOT_DEFAULT), help="Output directory for summary tables.")
    args = ap.parse_args()

    suites = [x.strip() for x in args.suites.replace(",", " ").split() if x.strip()]
    models_root = (REPO_ROOT / args.models_root).resolve() if not Path(args.models_root).is_absolute() else Path(args.models_root)
    outdir = (REPO_ROOT / args.outdir).resolve() if not Path(args.outdir).is_absolute() else Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for report_path in _discover_reports(models_root, suites):
        row = _row_from_report(report_path)
        if args.dataset_pattern and args.dataset_pattern not in row["dataset_name"]:
            continue
        rows.append(row)

    if not rows:
        raise SystemExit("[ERR] No matching profile-PCA quality reports found.")

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["val_profile_rmse", "val_profile_p95_rmse", "val_score_rmse"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    combined_csv = outdir / "profile_pca_sweep_summary.csv"
    combined_md = outdir / "profile_pca_sweep_summary.md"
    df.to_csv(combined_csv, index=False)
    _write_markdown_table(df, combined_md)
    print(f"[OK] Saved: {combined_csv}")
    print(f"[OK] Saved: {combined_md}")

    for suite, dff in df.groupby("suite", sort=False):
        suite_csv = outdir / f"{suite}_profile_pca_sweep_summary.csv"
        suite_md = outdir / f"{suite}_profile_pca_sweep_summary.md"
        dff = dff.sort_values(
            by=["val_profile_rmse", "val_profile_p95_rmse", "val_score_rmse"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
        dff.to_csv(suite_csv, index=False)
        _write_markdown_table(dff, suite_md)
        print(f"[OK] Saved: {suite_csv}")
        print(f"[OK] Saved: {suite_md}")

    best = df.iloc[0]
    print(
        "[OK] Best overall: "
        f"suite={best['suite']} dataset={best['dataset_name']} "
        f"val_profile_rmse={best['val_profile_rmse']:.6f} "
        f"val_profile_p95_rmse={best['val_profile_p95_rmse']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

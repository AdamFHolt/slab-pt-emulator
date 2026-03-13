#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
PLOTS_ROOT_DEFAULT = REPO_ROOT / "plots" / "qc-emulator" / "profile-pca" / "gp-tuning"


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return obj


def _items(text: str) -> list[str]:
    normalized = text.replace(",", " ")
    return [x.strip() for x in normalized.split() if x.strip()]


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_tag(tag: str) -> dict[str, Any]:
    out: dict[str, Any] = {"tag": tag}
    parts = tag.split("_")
    if len(parts) >= 5:
        out["kernel"] = parts[1]
        out["restarts"] = int(parts[2][1:]) if parts[2].startswith("r") else None
        out["ls_high_tag"] = parts[3][3:] if parts[3].startswith("lsu") else None
        out["noise_low_tag"] = parts[4][4:] if parts[4].startswith("nlow") else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize profile-PCA GP tuning sweep results.")
    ap.add_argument(
        "--sweep-root",
        default="src/emulator/models/profile_pca",
        help="Workflow root containing suite/gp_tuning/dataset/tag/profile_pca_quality.json",
    )
    ap.add_argument("--suites", default="ramped-vc", help="Comma- or space-separated suite list.")
    ap.add_argument(
        "--dataset-pattern",
        default="profileT_pca_t3Myr_k10_whitened",
        help="Keep only dataset names containing this text.",
    )
    ap.add_argument("--outdir", default=str(PLOTS_ROOT_DEFAULT))
    args = ap.parse_args()

    sweep_root = (REPO_ROOT / args.sweep_root).resolve() if not Path(args.sweep_root).is_absolute() else Path(args.sweep_root)
    suites = _items(args.suites)
    outdir = (REPO_ROOT / args.outdir).resolve() if not Path(args.outdir).is_absolute() else Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for suite in suites:
        suite_root = sweep_root / suite / "gp_tuning"
        if not suite_root.exists():
            continue
        for report_path in sorted(suite_root.glob("*/*/profile_pca_quality.json")):
            report = _load_json(report_path)
            dataset_name = str(report["dataset_name"])
            if args.dataset_pattern and args.dataset_pattern not in dataset_name:
                continue
            tag = report_path.parent.name
            meta = _parse_tag(tag)
            val = report["metrics"]["val"]
            rows.append(
                {
                    "suite": suite,
                    "dataset_name": dataset_name,
                    "tag": tag,
                    "kernel": meta.get("kernel"),
                    "restarts": meta.get("restarts"),
                    "ls_high_tag": meta.get("ls_high_tag"),
                    "noise_low_tag": meta.get("noise_low_tag"),
                    "val_profile_rmse": float(val["profile_space"]["emulator_reconstruction"]["rmse"]),
                    "val_profile_p95_rmse": float(val["profile_space"]["emulator_reconstruction"]["per_run_rmse"]["p95"]),
                    "val_score_rmse": float(val["score_space"]["_macro_avg"]["rmse"]),
                    "val_score_r2": float(val["score_space"]["_macro_avg"]["r2"]),
                    "val_pca_only_profile_rmse": float(val["profile_space"]["pca_truncation_baseline"]["rmse"]),
                    "report_path": report_path.as_posix(),
                }
            )

    if not rows:
        raise SystemExit("[ERR] No matching profile-PCA GP tuning reports found.")

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["val_profile_rmse", "val_profile_p95_rmse", "val_score_rmse"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    combined_csv = outdir / "profile_pca_gp_tuning_summary.csv"
    combined_md = outdir / "profile_pca_gp_tuning_summary.md"
    df.to_csv(combined_csv, index=False)
    _write_markdown_table(df, combined_md)
    print(f"[OK] Saved: {combined_csv}")
    print(f"[OK] Saved: {combined_md}")

    for suite, dff in df.groupby("suite", sort=False):
        suite_csv = outdir / f"{suite}_profile_pca_gp_tuning_summary.csv"
        suite_md = outdir / f"{suite}_profile_pca_gp_tuning_summary.md"
        dff.to_csv(suite_csv, index=False)
        _write_markdown_table(dff, suite_md)
        print(f"[OK] Saved: {suite_csv}")
        print(f"[OK] Saved: {suite_md}")

    best = df.iloc[0]
    print(
        "[OK] Best overall: "
        f"suite={best['suite']} dataset={best['dataset_name']} tag={best['tag']} "
        f"val_profile_rmse={best['val_profile_rmse']:.6f} "
        f"val_profile_p95_rmse={best['val_profile_p95_rmse']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping/object: {path}")
    return obj


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return obj


def _fmt(x: float) -> str:
    return f"{x:.6f}"


def _validate_one_report(report: dict[str, Any], limits: dict[str, Any]) -> tuple[bool, dict[str, Any], list[str]]:
    # We intentionally validate a small, interpretable set of metrics:
    # - score-space macro R2 / RMSE
    # - reconstructed profile RMSE
    # - reconstructed profile per-run p95 RMSE
    #
    # This keeps the first PCA gate aligned with the user’s goal:
    # check both the compressed target space and the physical profile space.
    val_metrics = report.get("metrics", {}).get("val", {})
    score_macro = val_metrics.get("score_space", {}).get("_macro_avg", {})
    emu_profile = val_metrics.get("profile_space", {}).get("emulator_reconstruction", {})
    emu_p95 = emu_profile.get("per_run_rmse", {}).get("p95")

    observed = {
        "score_r2": float(score_macro["r2"]),
        "score_rmse": float(score_macro["rmse"]),
        "profile_rmse": float(emu_profile["rmse"]),
        "profile_p95_rmse": float(emu_p95),
    }

    expected = {
        "score_r2_min": float(limits["score_r2_min"]),
        "score_rmse_max": float(limits["score_rmse_max"]),
        "profile_rmse_max": float(limits["profile_rmse_max"]),
        "profile_p95_rmse_max": float(limits["profile_p95_rmse_max"]),
    }

    checks = {
        "score_r2": observed["score_r2"] >= expected["score_r2_min"],
        "score_rmse": observed["score_rmse"] <= expected["score_rmse_max"],
        "profile_rmse": observed["profile_rmse"] <= expected["profile_rmse_max"],
        "profile_p95_rmse": observed["profile_p95_rmse"] <= expected["profile_p95_rmse_max"],
    }
    ok = all(checks.values())

    messages = [
        f"score_r2={_fmt(observed['score_r2'])}>={_fmt(expected['score_r2_min'])}({checks['score_r2']})",
        f"score_rmse={_fmt(observed['score_rmse'])}<={_fmt(expected['score_rmse_max'])}({checks['score_rmse']})",
        f"profile_rmse={_fmt(observed['profile_rmse'])}<={_fmt(expected['profile_rmse_max'])}({checks['profile_rmse']})",
        f"profile_p95_rmse={_fmt(observed['profile_p95_rmse'])}<={_fmt(expected['profile_p95_rmse_max'])}({checks['profile_p95_rmse']})",
    ]
    return ok, observed | expected, messages


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate profile-PCA quality reports against threshold spec.")
    ap.add_argument("--thresholds", required=True, help="YAML threshold config.")
    ap.add_argument(
        "--models-root",
        default="src/emulator/models/profile_pca",
        help="Root directory containing suite/runs/dataset/model_tag/profile_pca_quality.json",
    )
    ap.add_argument(
        "--model-tag",
        default=None,
        help="Model subdir tag (for example gp_m25). Defaults to model_tag in thresholds YAML.",
    )
    ap.add_argument("--suites", default=None, help="Optional comma-separated suite filter.")
    ap.add_argument("--datasets", default=None, help="Optional comma-separated dataset filter.")
    ap.add_argument("--allow-missing", action="store_true", help="Do not fail when reports are missing.")
    ap.add_argument("--json-out", default=None, help="Optional path to write machine-readable validation summary.")
    args = ap.parse_args()

    cfg_path = Path(args.thresholds).resolve()
    cfg = _load_yaml(cfg_path)

    model_tag = args.model_tag or str(cfg.get("model_tag", "")).strip()
    if not model_tag:
        raise ValueError("model_tag missing; set in YAML or via --model-tag.")

    thresholds = cfg.get("thresholds", {})
    if not isinstance(thresholds, dict):
        raise ValueError("thresholds must be a mapping of suite -> dataset -> limits.")

    models_root = Path(args.models_root).resolve()
    suites_filter = {x.strip() for x in args.suites.split(",") if x.strip()} if args.suites else None
    datasets_filter = {x.strip() for x in args.datasets.split(",") if x.strip()} if args.datasets else None

    checked = 0
    passed = 0
    failed = 0
    missing = 0
    rows: list[dict[str, Any]] = []

    print(f"[INFO] thresholds={cfg_path}")
    print(f"[INFO] models_root={models_root}")
    print(f"[INFO] model_tag={model_tag}")

    for suite, suite_cfg in sorted(thresholds.items()):
        if suites_filter and suite not in suites_filter:
            continue
        if not isinstance(suite_cfg, dict):
            raise ValueError(f"thresholds.{suite} must be a mapping.")

        for dataset_name, limits in sorted(suite_cfg.items()):
            if datasets_filter and dataset_name not in datasets_filter:
                continue
            if not isinstance(limits, dict):
                raise ValueError(f"thresholds.{suite}.{dataset_name} must be a mapping.")

            report_path = models_root / suite / "runs" / dataset_name / model_tag / "profile_pca_quality.json"
            if not report_path.exists():
                missing += 1
                rows.append(
                    {
                        "suite": suite,
                        "dataset": dataset_name,
                        "status": "MISSING",
                        "report": str(report_path),
                    }
                )
                print(f"[MISS] {suite}/{dataset_name} -> {report_path}")
                continue

            report = _load_json(report_path)
            checked += 1
            ok, values, messages = _validate_one_report(report, limits)
            status = "PASS" if ok else "FAIL"

            if ok:
                passed += 1
            else:
                failed += 1

            print(f"[{status}] {suite}/{dataset_name} " + " ".join(messages))

            rows.append(
                {
                    "suite": suite,
                    "dataset": dataset_name,
                    "status": status,
                    "report": str(report_path),
                    **values,
                }
            )

    print("------")
    print(f"[SUM] checked={checked} pass={passed} fail={failed} missing={missing}")

    if args.json_out:
        out = Path(args.json_out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checked": checked,
            "pass": passed,
            "fail": failed,
            "missing": missing,
            "model_tag": model_tag,
            "rows": rows,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[OK] wrote summary json: {out}")

    if failed > 0:
        return 1
    if missing > 0 and not args.allow_missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

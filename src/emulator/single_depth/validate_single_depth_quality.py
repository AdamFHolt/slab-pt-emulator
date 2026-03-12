#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Threshold config must be a mapping/object.")
    return cfg


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return obj


def _get_nested(d: dict[str, Any], dotted_key: str) -> Any:
    cur: Any = d
    for key in dotted_key.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(dotted_key)
        cur = cur[key]
    return cur


def _fmt(x: float) -> str:
    return f"{x:.6f}"


def main() -> int:
    p = argparse.ArgumentParser(description="Validate emulator report metrics against threshold spec.")
    p.add_argument(
        "--thresholds",
        default="configs/emulator-quality.gp_m25.yaml",
        help="YAML threshold spec file.",
    )
    p.add_argument(
        "--models-root",
        default="src/emulator/models/single_depth",
        help="Root directory containing suite/runs/data_name/model_tag/report.json",
    )
    p.add_argument(
        "--model-tag",
        default=None,
        help="Model subdir tag (e.g., gp_m25). Defaults to model_tag in thresholds YAML.",
    )
    p.add_argument(
        "--suites",
        default=None,
        help="Optional comma-separated suite filter (e.g., const-vc,ramped-vc).",
    )
    p.add_argument(
        "--datasets",
        default=None,
        help="Optional comma-separated dataset filter (e.g., 10km_dTdt,20km_dTdt).",
    )
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="Do not fail on missing report.json files; skip them with warning.",
    )
    p.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write machine-readable validation summary JSON.",
    )
    args = p.parse_args()

    cfg_path = Path(args.thresholds).resolve()
    cfg = _load_yaml(cfg_path)

    metric_source = str(cfg.get("metric_source", "metrics.val._macro_avg"))
    thresholds = cfg.get("thresholds", {})
    if not isinstance(thresholds, dict):
        raise ValueError("thresholds must be a mapping of suite -> dataset -> limits.")

    model_tag = args.model_tag or str(cfg.get("model_tag", "")).strip()
    if not model_tag:
        raise ValueError("model_tag missing; set in YAML or via --model-tag.")

    models_root = Path(args.models_root).resolve()
    suites_filter = None
    if args.suites:
        suites_filter = {s.strip() for s in args.suites.split(",") if s.strip()}
    datasets_filter = None
    if args.datasets:
        datasets_filter = {d.strip() for d in args.datasets.split(",") if d.strip()}

    checked = 0
    passed = 0
    failed = 0
    missing = 0
    rows: list[dict[str, Any]] = []

    print(f"[INFO] thresholds={cfg_path}")
    print(f"[INFO] models_root={models_root}")
    print(f"[INFO] model_tag={model_tag}")
    print(f"[INFO] metric_source={metric_source}")

    for suite, suite_cfg in sorted(thresholds.items()):
        if suites_filter and suite not in suites_filter:
            continue
        if not isinstance(suite_cfg, dict):
            raise ValueError(f"thresholds.{suite} must be mapping.")
        for data_name, limits in sorted(suite_cfg.items()):
            if datasets_filter and data_name not in datasets_filter:
                continue
            if not isinstance(limits, dict):
                raise ValueError(f"thresholds.{suite}.{data_name} must be mapping.")

            report_path = models_root / suite / "runs" / data_name / model_tag / "report.json"
            if not report_path.exists():
                missing += 1
                rows.append(
                    dict(
                        suite=suite,
                        dataset=data_name,
                        status="MISSING",
                        report=str(report_path),
                    )
                )
                print(f"[MISS] {suite}/{data_name} -> {report_path}")
                continue

            rep = _load_json(report_path)
            try:
                metrics = _get_nested(rep, metric_source)
            except KeyError:
                missing += 1
                rows.append(
                    dict(
                        suite=suite,
                        dataset=data_name,
                        status="MISSING_METRICS",
                        report=str(report_path),
                    )
                )
                print(f"[MISS] {suite}/{data_name} -> metric path not found: {metric_source}")
                continue

            if not isinstance(metrics, dict):
                raise ValueError(f"Expected dict at metric path for {report_path}: {metric_source}")

            checked += 1
            r2 = float(metrics["r2"])
            rmse = float(metrics["rmse"])
            mae = float(metrics["mae"])
            r2_min = float(limits["r2_min"])
            rmse_max = float(limits["rmse_max"])
            mae_max = float(limits["mae_max"])

            ok_r2 = r2 >= r2_min
            ok_rmse = rmse <= rmse_max
            ok_mae = mae <= mae_max
            ok = ok_r2 and ok_rmse and ok_mae

            if ok:
                passed += 1
                status = "PASS"
                print(
                    f"[PASS] {suite}/{data_name} "
                    f"r2={_fmt(r2)}>={_fmt(r2_min)} "
                    f"rmse={_fmt(rmse)}<={_fmt(rmse_max)} "
                    f"mae={_fmt(mae)}<={_fmt(mae_max)}"
                )
            else:
                failed += 1
                status = "FAIL"
                print(
                    f"[FAIL] {suite}/{data_name} "
                    f"r2={_fmt(r2)}>={_fmt(r2_min)}({ok_r2}) "
                    f"rmse={_fmt(rmse)}<={_fmt(rmse_max)}({ok_rmse}) "
                    f"mae={_fmt(mae)}<={_fmt(mae_max)}({ok_mae})"
                )

            rows.append(
                dict(
                    suite=suite,
                    dataset=data_name,
                    status=status,
                    report=str(report_path),
                    r2=r2,
                    rmse=rmse,
                    mae=mae,
                    r2_min=r2_min,
                    rmse_max=rmse_max,
                    mae_max=mae_max,
                )
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
            "metric_source": metric_source,
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

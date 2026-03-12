#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent
TRAINER = REPO_ROOT / "src" / "emulator" / "train_emulator.py"


def _depth_sort_key(name: str) -> tuple[float, str]:
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*km_", name)
    if not m:
        return (1e12, name)
    return (float(m.group(1)), name)


def _load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping/object.")
    return cfg


def _discover_datasets(suite_dir: Path, ds_cfg: dict[str, Any]) -> list[str]:
    mode = str(ds_cfg.get("mode", "auto")).strip().lower()
    if mode not in {"auto", "list", "profile-pca"}:
        raise ValueError("dataset.mode must be 'auto', 'list', or 'profile-pca'.")

    if mode == "list":
        names = ds_cfg.get("names", [])
        if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
            raise ValueError("dataset.names must be a list of dataset folder names.")
        return sorted(names, key=_depth_sort_key)

    if not suite_dir.exists():
        raise FileNotFoundError(f"Suite data directory not found: {suite_dir}")

    names = [p.name for p in suite_dir.iterdir() if p.is_dir()]

    if mode == "profile-pca":
        # For profile-PCA datasets, default to *_pca* folders unless a prefix is given.
        prefix = str(ds_cfg.get("prefix", "")).strip()
        if prefix:
            names = [n for n in names if n.startswith(prefix)]
        else:
            names = [n for n in names if "_pca" in n.lower()]

        pattern = ds_cfg.get("pattern")
        if pattern:
            rgx = re.compile(str(pattern))
            names = [n for n in names if rgx.search(n)]

        return sorted(names)

    variant = ds_cfg.get("variant")
    if variant:
        pat = re.compile(r"^\d+(?:\.\d+)?km_" + re.escape(str(variant)) + r"$")
        names = [n for n in names if pat.match(n)]

    pattern = ds_cfg.get("pattern")
    if pattern:
        rgx = re.compile(str(pattern))
        names = [n for n in names if rgx.search(n)]

    return sorted(names, key=_depth_sort_key)


def _suite_data_dir(data_root: Path, suite: str, ds_cfg: dict[str, Any]) -> Path:
    subdir = str(ds_cfg.get("subdir", "runs")).strip()
    return data_root / suite / subdir


def _suite_out_dir(out_root: Path, suite: str, model_cfg: dict[str, Any]) -> Path:
    subdir = str(model_cfg.get("subdir", "runs")).strip()
    return out_root / suite / subdir


def _as_cli_float_pair(vals: Any, key: str) -> tuple[str, str]:
    if not isinstance(vals, (list, tuple)) or len(vals) != 2:
        raise ValueError(f"{key} must be a list of exactly two numbers.")
    return (str(float(vals[0])), str(float(vals[1])))


def _build_train_cmd(cfg: dict[str, Any], data_root_suite: Path, out_root_suite: Path, data_name: str) -> list[str]:
    model_cfg = dict(cfg.get("model", {}))
    model_type = str(model_cfg.get("type", "gp")).strip().lower()
    if model_type not in {"gp", "rf"}:
        raise ValueError("model.type must be 'gp' or 'rf'.")

    cmd = [
        sys.executable,
        str(TRAINER),
        "--data-root", str(data_root_suite),
        "--data-name", data_name,
        "--model", model_type,
        "--out", str(out_root_suite),
        "--seed", str(int(model_cfg.get("seed", 42))),
    ]

    if model_type == "gp":
        kernel = str(model_cfg.get("kernel", "matern25")).strip().lower()
        if kernel not in {"rbf", "matern15", "matern25"}:
            raise ValueError("model.kernel must be one of: rbf, matern15, matern25.")

        ls_bounds = _as_cli_float_pair(model_cfg.get("ls_bounds", [1e-3, 1e3]), "model.ls_bounds")
        noise_bounds = _as_cli_float_pair(model_cfg.get("noise_bounds", [1e-6, 1.0]), "model.noise_bounds")
        cmd += [
            "--kernel", kernel,
            "--ls-init", str(float(model_cfg.get("ls_init", 1.0))),
            "--ls-bounds", ls_bounds[0], ls_bounds[1],
            "--noise-init", str(float(model_cfg.get("noise_init", 3e-3))),
            "--noise-bounds", noise_bounds[0], noise_bounds[1],
            "--alpha", str(float(model_cfg.get("alpha", 1e-6))),
            "--gp-restarts", str(int(model_cfg.get("gp_restarts", 25))),
        ]
    else:
        cmd += [
            "--rf-trees", str(int(model_cfg.get("rf_trees", 400))),
            "--rf-jobs", str(int(model_cfg.get("rf_jobs", -1))),
        ]
        rf_max_depth = model_cfg.get("rf_max_depth")
        if rf_max_depth is not None:
            cmd += ["--rf-max-depth", str(int(rf_max_depth))]

    return cmd


def main() -> int:
    p = argparse.ArgumentParser(description="Unified training entry point for emulator runs.")
    p.add_argument("--config", required=True, help="YAML config path, e.g. configs/gp.yaml")
    p.add_argument("--dry-run", action="store_true", help="Print commands only, do not execute training.")
    p.add_argument("--datasets", default=None,
                   help="Optional comma-separated dataset override (e.g., 10km_dTdt,20km_dTdt).")
    args = p.parse_args()

    cfg = _load_config((REPO_ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config))

    suite = str(cfg.get("suite", "const-vc")).strip()
    if suite not in {"const-vc", "ramped-vc"}:
        raise ValueError("suite must be 'const-vc' or 'ramped-vc'.")

    data_root = Path(cfg.get("data_root", "src/emulator/data/single_depth")).resolve()
    out_root = Path(cfg.get("out_root", "src/emulator/models/single_depth")).resolve()

    ds_cfg = dict(cfg.get("dataset", {}))
    model_cfg = dict(cfg.get("model", {}))

    data_root_suite = _suite_data_dir(data_root, suite, ds_cfg)
    out_root_suite = _suite_out_dir(out_root, suite, model_cfg)
    out_root_suite.mkdir(parents=True, exist_ok=True)
    dataset_names = _discover_datasets(data_root_suite, ds_cfg)

    if args.datasets:
        wanted = [x.strip() for x in args.datasets.split(",") if x.strip()]
        dataset_names = wanted

    if not dataset_names:
        raise RuntimeError(f"No datasets resolved for suite '{suite}' under {data_root_suite}.")

    execution = dict(cfg.get("execution", {}))
    fail_fast = bool(execution.get("fail_fast", True))
    dry_run = bool(execution.get("dry_run", False)) or args.dry_run

    print(f"[INFO] suite={suite}")
    print(f"[INFO] data_root={data_root_suite}")
    print(f"[INFO] out_root={out_root_suite}")
    print(f"[INFO] datasets={len(dataset_names)}")
    for d in dataset_names:
        print(f"  - {d}")

    failures: list[str] = []
    for name in dataset_names:
        cmd = _build_train_cmd(cfg, data_root_suite, out_root_suite, name)
        print("------")
        print("[RUN]", " ".join(cmd))

        if dry_run:
            continue

        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if proc.returncode != 0:
            failures.append(name)
            print(f"[ERR] training failed for dataset: {name}")
            if fail_fast:
                return proc.returncode

    if failures:
        print(f"[ERR] completed with failures ({len(failures)}): {', '.join(failures)}")
        return 1

    if dry_run:
        print("[OK] dry-run complete.")
    else:
        print("[OK] all trainings complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

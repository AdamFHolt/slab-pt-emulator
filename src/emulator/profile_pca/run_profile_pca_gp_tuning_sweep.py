#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent
EMULATOR_DIR = SCRIPT_DIR.parent
REPO_ROOT = THIS_FILE.parents[3]
TRAIN_PY = EMULATOR_DIR / "train_emulator.py"
EVAL_PY = SCRIPT_DIR / "evaluate_profile_pca_quality.py"


def _items(text: str) -> list[str]:
    normalized = text.replace(",", " ")
    return [x.strip() for x in normalized.split() if x.strip()]


def _run(cmd: list[str], dry_run: bool) -> None:
    print("[RUN]", " ".join(cmd))
    if dry_run:
        return
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _sci_tag(value: float) -> str:
    # Keep tags filesystem-friendly and stable across runs.
    text = f"{value:.0e}"
    mantissa, exponent = text.split("e", maxsplit=1)
    exponent = exponent.replace("+", "")
    sign = ""
    if exponent.startswith("-"):
        sign = "-"
        exponent = exponent[1:]
    exponent = exponent.lstrip("0") or "0"
    return f"{mantissa}e{sign}{exponent}"


def _tag(kernel: str, restarts: int, ls_high: float, noise_low: float) -> str:
    return f"gp_{kernel}_r{restarts}_lsu{_sci_tag(ls_high)}_nlow{_sci_tag(noise_low)}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run a GP hyperparameter sweep on a fixed profile-PCA dataset."
    )
    ap.add_argument("--suites", default="ramped-vc", help="Comma- or space-separated suite list.")
    ap.add_argument(
        "--datasets",
        default="profileT_pca_t3Myr_k10_whitened",
        help="Comma- or space-separated profile-PCA dataset names.",
    )
    ap.add_argument("--kernels", default="matern25 matern15 rbf", help="Kernel choices.")
    ap.add_argument("--restarts", default="10 25", help="GP restart counts.")
    ap.add_argument("--ls-highs", default="1e3 1e4", help="Upper bounds for GP length scales.")
    ap.add_argument("--noise-lows", default="1e-6 1e-8", help="Lower bounds for GP noise level.")
    ap.add_argument("--ls-init", type=float, default=1.0)
    ap.add_argument("--noise-init", type=float, default=3e-3)
    ap.add_argument("--alpha", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--sweep-root",
        default=str(EMULATOR_DIR / "models" / "profile-pca-gp-sweep"),
        help="Root directory for copied sweep artifacts.",
    )
    ap.add_argument(
        "--base-model-root",
        default=str(EMULATOR_DIR / "models"),
        help="Temporary training root passed to train_emulator.py before copying artifacts.",
    )
    ap.add_argument(
        "--skip-evaluate",
        action="store_true",
        help="Only train/copy models; do not compute profile_pca_quality.json.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Rerun jobs even when the sweep output already has profile_pca_quality.json.",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    suites = _items(args.suites)
    datasets = _items(args.datasets)
    kernels = _items(args.kernels)
    restarts = [int(x) for x in _items(args.restarts)]
    ls_highs = [float(x) for x in _items(args.ls_highs)]
    noise_lows = [float(x) for x in _items(args.noise_lows)]

    for suite in suites:
        if suite not in {"const-vc", "ramped-vc"}:
            raise ValueError("suites must contain only 'const-vc' or 'ramped-vc'")
    for kernel in kernels:
        if kernel not in {"matern25", "matern15", "rbf"}:
            raise ValueError("kernels must contain only 'matern25', 'matern15', or 'rbf'")

    base_model_root = Path(args.base_model_root).resolve()
    sweep_root = Path(args.sweep_root).resolve()

    for suite in suites:
        data_root = (EMULATOR_DIR / "data" / suite).resolve()
        for dataset_name in datasets:
            dataset_dir = data_root / dataset_name
            if not args.dry_run and not dataset_dir.exists():
                raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

            for kernel in kernels:
                for restarts_n in restarts:
                    for ls_high in ls_highs:
                        for noise_low in noise_lows:
                            tag = _tag(kernel, restarts_n, ls_high, noise_low)
                            print("------")
                            print(
                                f"[JOB] suite={suite} dataset={dataset_name} "
                                f"kernel={kernel} restarts={restarts_n} "
                                f"ls_high={ls_high:g} noise_low={noise_low:g} tag={tag}"
                            )

                            model_dir_name = {
                                "matern25": "gp_m25",
                                "matern15": "gp_m15",
                                "rbf": "gp_rbf",
                            }[kernel]
                            trained_dir = base_model_root / suite / dataset_name / model_dir_name
                            sweep_dir = sweep_root / suite / dataset_name / tag
                            quality_json = sweep_dir / "profile_pca_quality.json"

                            if quality_json.exists() and not args.force:
                                print(f"[SKIP] existing quality report: {quality_json}")
                                continue

                            train_cmd = [
                                sys.executable,
                                str(TRAIN_PY),
                                "--data-root",
                                str(data_root),
                                "--data-name",
                                dataset_name,
                                "--model",
                                "gp",
                                "--kernel",
                                kernel,
                                "--gp-restarts",
                                str(restarts_n),
                                "--ls-init",
                                str(args.ls_init),
                                "--ls-bounds",
                                "0.001",
                                str(ls_high),
                                "--noise-init",
                                str(args.noise_init),
                                "--noise-bounds",
                                str(noise_low),
                                "1.0",
                                "--alpha",
                                str(args.alpha),
                                "--seed",
                                str(args.seed),
                                "--out",
                                str(base_model_root / suite),
                            ]
                            _run(train_cmd, dry_run=args.dry_run)

                            if args.dry_run:
                                print(f"[COPY] {trained_dir} -> {sweep_dir}")
                            else:
                                if sweep_dir.exists():
                                    shutil.rmtree(sweep_dir)
                                sweep_dir.mkdir(parents=True, exist_ok=True)
                                shutil.copytree(trained_dir, sweep_dir, dirs_exist_ok=True)
                                print(f"[OK] copied sweep artifacts -> {sweep_dir}")

                            if not args.skip_evaluate:
                                eval_cmd = [
                                    sys.executable,
                                    str(EVAL_PY),
                                    "--dataset-dir",
                                    str(dataset_dir),
                                    "--model-dir",
                                    str(sweep_dir if not args.dry_run else sweep_dir),
                                ]
                                _run(eval_cmd, dry_run=args.dry_run)

    if args.dry_run:
        print("[OK] dry-run complete.")
    else:
        print("[OK] GP tuning sweep complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable


THIS_FILE = Path(__file__).resolve()
SCRIPT_DIR = THIS_FILE.parent
EMULATOR_DIR = SCRIPT_DIR.parent
REPO_ROOT = THIS_FILE.parents[3]


def _csv_items(text: str) -> list[str]:
    # Accept both comma-separated and space-separated lists so the script is
    # easy to call from either the shell or a Makefile variable.
    normalized = text.replace(",", " ")
    return [x.strip() for x in normalized.split() if x.strip()]


def _time_label(time_text: str) -> str:
    return time_text.replace(".", "p")


def _dataset_name(time_text: str, k: int, score_space: str) -> str:
    # The sweep must encode score space into the dataset name, otherwise the
    # raw and whitened variants would overwrite each other on disk.
    return f"profileT_pca_t{_time_label(time_text)}Myr_k{k}_{score_space}"


def _run(cmd: list[str], dry_run: bool) -> None:
    print("[RUN]", " ".join(cmd))
    if dry_run:
        return
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _iter_jobs(
    suites: Iterable[str],
    times: Iterable[str],
    ks: Iterable[int],
    score_spaces: Iterable[str],
) -> Iterable[tuple[str, str, int, str, str]]:
    for suite in suites:
        for time_text in times:
            for k in ks:
                for score_space in score_spaces:
                    yield suite, time_text, k, score_space, _dataset_name(time_text, k, score_space)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a profile-PCA sweep over k and score-space choices.")
    ap.add_argument("--suites", default="const-vc,ramped-vc", help="Comma-separated suite list.")
    ap.add_argument("--times", default="3", help="Comma-separated target times in Myr.")
    ap.add_argument("--ks", default="4,6,8,10", help="Comma-separated PCA component counts.")
    ap.add_argument("--score-spaces", default="raw,whitened", help="Comma-separated score-space choices.")
    ap.add_argument(
        "--steps",
        default="preprocess,train,evaluate",
        help="Comma-separated steps to run: preprocess,train,evaluate",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = ap.parse_args()

    suites = _csv_items(args.suites)
    times = _csv_items(args.times)
    ks = [int(x) for x in _csv_items(args.ks)]
    score_spaces = _csv_items(args.score_spaces)
    steps = set(_csv_items(args.steps))

    for step in steps:
        if step not in {"preprocess", "train", "evaluate"}:
            raise ValueError("steps must be chosen from: preprocess, train, evaluate")
    for suite in suites:
        if suite not in {"const-vc", "ramped-vc"}:
            raise ValueError("suites must contain only 'const-vc' or 'ramped-vc'")
    for score_space in score_spaces:
        if score_space not in {"raw", "whitened"}:
            raise ValueError("score-spaces must contain only 'raw' or 'whitened'")
    for k in ks:
        if k < 1:
            raise ValueError("all k values must be >= 1")

    for suite, time_text, k, score_space, dataset_name in _iter_jobs(suites, times, ks, score_spaces):
        print("------")
        print(
            f"[JOB] suite={suite} time_myr={time_text} "
            f"k={k} score_space={score_space} dataset={dataset_name}"
        )

        if "preprocess" in steps:
            _run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "preprocess_profile_pca.py"),
                    "--suite",
                    suite,
                    "--target-time-myr",
                    time_text,
                    "--k",
                    str(k),
                    "--score-space",
                    score_space,
                    "--dataset-name",
                    dataset_name,
                ],
                dry_run=args.dry_run,
            )

        if "train" in steps:
            _run(
                [
                    sys.executable,
                    "train.py",
                    "--config",
                    f"configs/gp.{suite}.profile-pca.yaml",
                    "--datasets",
                    dataset_name,
                ],
                dry_run=args.dry_run,
            )

        if "evaluate" in steps:
            _run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "evaluate_profile_pca_quality.py"),
                    "--dataset-dir",
                    str(EMULATOR_DIR / "data" / suite / dataset_name),
                    "--model-dir",
                    str(EMULATOR_DIR / "models" / suite / dataset_name / "gp_m25"),
                ],
                dry_run=args.dry_run,
            )

    if args.dry_run:
        print("[OK] dry-run complete.")
    else:
        print("[OK] sweep complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

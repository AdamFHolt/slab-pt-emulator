#!/usr/bin/env python3
"""Seed the gp_m25 quality gates of a suite from its current report baselines.

Appends a `<suite>:` block to configs/emulator-quality.gp_m25.yaml (single-depth: every
<depth>km_dTdt[_thermalParam] model under src/emulator/models/single_depth/<suite>/runs) and to
configs/profile-pca-quality.gp_m25.yaml (profile-PCA: profileT_pca_t3Myr_k10), with the slack rule the
existing entries were made with (see the notes in each yaml):

  single-depth : r2_min = r2 - 0.01,          rmse_max = 1.25 rmse,        mae_max = 1.25 mae
  profile-PCA  : score_r2_min = r2 - 0.02,    score_rmse_max = 1.25 rmse,
                 profile_rmse_max = 1.25 rmse, profile_p95_rmse_max = 1.25 p95

Refuses to touch a suite that already has a block (edit by hand instead). The block is appended as text so
the comments in the yaml survive; both files end inside the top-level `thresholds:` mapping.

Usage:  env/bin/python src/emulator/seed_quality_thresholds.py const-vc-sh [--dry-run]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
SD_YAML = ROOT / "configs" / "emulator-quality.gp_m25.yaml"
PCA_YAML = ROOT / "configs" / "profile-pca-quality.gp_m25.yaml"
TAG = "gp_m25"


def _r(x: float, nd: int) -> float:
    return float(f"{x:.{nd}f}")


def sd_block(suite: str) -> str:
    root = ROOT / "src" / "emulator" / "models" / "single_depth" / suite / "runs"
    reports = sorted(root.glob(f"*km_dTdt*/{TAG}/report.json"),
                     key=lambda p: (int(p.parents[1].name.split("km")[0]), p.parents[1].name))
    if not reports:
        sys.exit(f"no single-depth reports under {root}")
    lines = [f"  {suite}:"]
    for rp in reports:
        m = json.load(open(rp))["metrics"]["val"]["_macro_avg"]
        lines.append(f"    {rp.parents[1].name}: {{r2_min: {_r(m['r2'] - 0.01, 3)}, "
                     f"rmse_max: {_r(1.25 * m['rmse'], 2)}, mae_max: {_r(1.25 * m['mae'], 2)}}}")
    return "\n".join(lines) + "\n"


def pca_block(suite: str, dataset: str = "profileT_pca_t3Myr_k10") -> str:
    q = ROOT / "src" / "emulator" / "models" / "profile_pca" / suite / "runs" / dataset / TAG / "profile_pca_quality.json"
    if not q.exists():
        sys.exit(f"no profile-PCA quality report at {q}")
    val = json.load(open(q))["metrics"]["val"]
    sc = val["score_space"]["_macro_avg"]
    emu = val["profile_space"]["emulator_reconstruction"]
    return (f"  {suite}:\n    {dataset}:\n"
            f"      score_r2_min: {_r(sc['r2'] - 0.02, 3)}\n"
            f"      score_rmse_max: {_r(1.25 * sc['rmse'], 2)}\n"
            f"      profile_rmse_max: {_r(1.25 * emu['rmse'], 2)}\n"
            f"      profile_p95_rmse_max: {_r(1.25 * emu['per_run_rmse']['p95'], 2)}\n")


def append(path: Path, suite: str, block: str, dry: bool) -> None:
    cfg = yaml.safe_load(path.read_text())
    if suite in cfg.get("thresholds", {}):
        sys.exit(f"{path.name}: thresholds.{suite} already exists; edit it by hand")
    text = path.read_text()
    last = [l for l in text.splitlines() if l.strip()][-1]
    if not last.startswith("  "):
        sys.exit(f"{path.name} does not end inside the thresholds mapping; append by hand")
    new = text.rstrip("\n") + "\n\n" + block
    if dry:
        print(f"--- would append to {path.name}:\n{block}")
        return
    path.write_text(new)
    check = yaml.safe_load(new)["thresholds"][suite]
    print(f"[OK] {path.name}: thresholds.{suite} added ({len(check)} datasets)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    append(SD_YAML, a.suite, sd_block(a.suite), a.dry_run)
    append(PCA_YAML, a.suite, pca_block(a.suite), a.dry_run)

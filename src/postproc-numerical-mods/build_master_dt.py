#!/usr/bin/env python3
"""Assemble a hierarchical master cooling table from per-run DT_<a>_<b>.csv files.

Byte-for-byte equivalent to the awk aggregation loop inside
``extract_cooling-rates_all-mods.sh`` (values are copied verbatim from the
per-run CSV text; missing runs/depths become NaN rows), but it does not
re-run any extraction, so an existing master table can never be clobbered by
an accidental re-extraction.

Usage:
    build_master_dt.py --suite const-vc --t1 1 --t2 20 --depths 0:80:1
    build_master_dt.py --suite const-vc --t1 1 --t2 20 --out /path/master.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

COLS = ["T1_C", "T2_C", "dT_C", "dt_Myr", "dTdt_C_per_Myr"]
NAN_ROW = ["NaN"] * len(COLS)


def parse_depths(spec: str) -> list[str]:
    spec = spec.strip()
    if ":" in spec:
        parts = [int(p) for p in spec.split(":")]
        step = parts[2] if len(parts) == 3 else 1
        return [str(d) for d in range(parts[0], parts[1] + 1, step)]
    return [p.strip() for p in spec.split(",") if p.strip()]


def read_dt_file(path: Path) -> dict[float, list[str]]:
    """depth (float) -> the five master columns, as raw text."""
    out: dict[float, list[str]] = {}
    lines = path.read_text().splitlines()
    if not lines:
        return out
    header = [h.strip() for h in lines[0].split(",")]
    idx = {name: header.index(name) for name in COLS if name in header}
    if len(idx) != len(COLS):
        return out
    dcol = header.index("depth_km")
    for line in lines[1:]:
        if not line.strip():
            continue
        f = line.split(",")
        try:
            d = float(f[dcol])
        except (ValueError, IndexError):
            continue
        vals = [f[idx[c]].strip() for c in COLS]
        out[d] = NAN_ROW if any(v == "" for v in vals) else vals
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="const-vc")
    ap.add_argument("--t1", type=int, required=True)
    ap.add_argument("--t2", type=int, required=True)
    ap.add_argument("--depths", default="0:80:1")
    ap.add_argument("--out", default=None, help="Default: <suite>/analysis/master_DT<t1>-<t2>.csv")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing master table.")
    args = ap.parse_args()

    suite_dir = REPO_ROOT / "subd-model-runs" / args.suite
    analysis = suite_dir / "analysis"
    run_root = suite_dir / "run-outputs"
    out = Path(args.out).resolve() if args.out else (analysis / f"master_DT{args.t1}-{args.t2}.csv")

    if out.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing {out} (use --force)")

    depths = parse_depths(args.depths)
    run_names = sorted(p.name for p in run_root.glob("run_*") if p.is_dir())

    n_present = 0
    lines = ["depth_km,run_id,T1_C,T2_C,dT_C,dt_Myr,dTdt_C_per_Myr"]
    missing: list[str] = []
    for run_name in run_names:
        run_num = run_name[len("run_"):]
        dt_path = analysis / run_name / f"DT_{args.t1}_{args.t2}.csv"
        table = read_dt_file(dt_path) if dt_path.exists() else {}
        if table:
            n_present += 1
        else:
            missing.append(run_num)
        for d in depths:
            vals = table.get(float(d), NAN_ROW)
            lines.append(f"{d},{run_num},{','.join(vals)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"[OK] wrote {out}")
    print(f"[OK] runs total={len(run_names)} with DT_{args.t1}_{args.t2}={n_present} missing={len(missing)}")
    if missing:
        print("[OK] missing run ids: " + ",".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

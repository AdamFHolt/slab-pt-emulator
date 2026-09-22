#!/usr/bin/env python3
"""Write subd-model-runs/const-vc-sh/run-inputs/full-list.txt: all 400 runs in submission order.

Order = the 8 pilot runs first (pilot-list.txt, unchanged order), then the remaining 392 by
greedy maximin in the normalised 5-D design space (log10 eta_UM), seeded with the pilot.  The
design is a Latin hypercube, whose row order is arbitrary, so this makes every prefix of the
list roughly space-filling: if the feeder is stopped early (a problem seen in the first
finishers, an allocation limit) the runs already done still cover the parameter space instead of
a corner of it.  The pilot at the head means the first finishers are exactly the runs the README's
pilot checks were written for.

Usage:  python src/build-numerical-mods/make_full_list.const-vc-sh.py
Deterministic; no dependencies beyond numpy.  run-inputs/ is gitignored, so rerun after a fresh
checkout.  push_runs_to_tacc.sh rsyncs the whole run-inputs/, so the list travels with the suite.
"""
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_INPUTS = ROOT / "subd-model-runs" / "const-vc-sh" / "run-inputs"
PARAMS = ROOT / "data" / "params" / "params-list.const-vc-sh.csv"
PILOT = RUN_INPUTS / "pilot-list.txt"
OUT = RUN_INPUTS / "full-list.txt"

X = np.genfromtxt(PARAMS, delimiter=",", names=True)
cols = list(X.dtype.names)
A = np.column_stack([np.log10(X[c]) if c == "eta_UM" else X[c] for c in cols])
A = (A - A.min(0)) / (A.max(0) - A.min(0))          # unit hypercube
n = len(A)
assert n == 400, n

pilot = [int(l) for l in PILOT.read_text().split() if l.strip() and not l.startswith("#")]
assert len(set(pilot)) == len(pilot) == 8 and all(0 <= p < n for p in pilot), pilot

order = list(pilot)
chosen = np.zeros(n, bool); chosen[pilot] = True
# min distance from every point to the chosen set, updated incrementally
dmin = np.min(np.linalg.norm(A[:, None, :] - A[None, pilot, :], axis=2), axis=1)
dmin[chosen] = -1
while len(order) < n:
    i = int(np.argmax(dmin))               # farthest point from everything chosen so far
    order.append(i); chosen[i] = True
    dmin = np.minimum(dmin, np.linalg.norm(A - A[i], axis=1)); dmin[chosen] = -1
assert sorted(order) == list(range(n))

lines = ["# const-vc-sh: all 400 runs in submission order (make_full_list.const-vc-sh.py).",
         "# First 8 = pilot-list.txt; then greedy maximin in normalised (v_conv, age_SP, age_OP, dip, log10 eta_UM),",
         "# so any prefix of this list is roughly space-filling.  Submit on spr with:",
         "#   export asp3_skx=...; SLURM_FILE=run_one.spr.slurm nohup ./submit_from_list.sh full-list.txt > submit_full.log 2>&1 &",
         "# (THROTTLE defaults to 24 = spr per-user queue limit.)"]
lines += [f"{i:03d}" for i in order]
OUT.write_text("\n".join(lines) + "\n")

# report: coverage of the first k runs = fraction of design points within r of some chosen point
print(f"wrote {OUT} ({n} runs)")
print("prefix   min pairwise dist   mean dist of all 400 to nearest chosen")
for k in (8, 24, 48, 100, 200, 400):
    S = A[order[:k]]
    D = np.linalg.norm(A[:, None, :] - S[None, :, :], axis=2)
    pd = np.linalg.norm(S[:, None, :] - S[None, :, :], axis=2); pd[np.eye(k, dtype=bool)] = np.inf
    print(f"{k:5d}   {pd.min():.3f}               {D.min(1).mean():.3f}")

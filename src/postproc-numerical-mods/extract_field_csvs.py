#!/home/holt/software/ParaView-5.11.1-MPI-Linux-Python3.9-x86_64/bin/pvpython
"""Write analysis/run_XXX/t{k}.csv for a list of output timesteps.

Thin pvpython wrapper around ``utils.model_processing.extract_csv`` so the
field-CSV stage can be driven independently of the profile stage (see
``extend_profiles_all-mods.sh``).  Timesteps whose CSV already exists are
skipped by ``extract_csv`` itself.

Usage:
    extract_field_csvs.py RUN_NUM "1,10,11,...,20" [SUITE]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from utils.model_processing import extract_csv

if len(sys.argv) < 3:
    raise SystemExit("Usage: extract_field_csvs.py RUN_NUM STEPS_CSV [SUITE]")

MOD_NAME = str(sys.argv[1]).replace("run_", "")
STEPS = [int(s) for s in str(sys.argv[2]).replace(" ", "").split(",") if s != ""]
SUITE = str(sys.argv[3]) if len(sys.argv) > 3 else "const-vc"

SUITE_ROOT = ROOT / "subd-model-runs" / SUITE
IN_DIR = SUITE_ROOT / "run-outputs"
RUN_ANALYSIS = SUITE_ROOT / "analysis" / f"run_{MOD_NAME}"
RUN_ANALYSIS.mkdir(parents=True, exist_ok=True)

for k in sorted(set(STEPS)):
    extract_csv(IN_DIR, RUN_ANALYSIS, MOD_NAME, k)

print("Done.")

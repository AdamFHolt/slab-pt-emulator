# pvpython driver: write t{k}.csv field CSVs for one dd100 run (args: RUN STEP [STEP...])
import sys
from pathlib import Path
sys.path.insert(0, "/home/holt/Projects/SlabPT-emulator/src")
from utils.model_processing import extract_csv
ROOT = Path("/home/holt/Projects/SlabPT-emulator/subd-model-runs/const-vc-dd100")
run = sys.argv[1]
out = ROOT / "analysis" / f"run_{run}"; out.mkdir(parents=True, exist_ok=True)
for k in sys.argv[2:]:
    extract_csv(str(ROOT / "run-outputs"), out, run, int(k))

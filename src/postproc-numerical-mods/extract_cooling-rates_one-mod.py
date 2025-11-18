#!/home/holt/software/ParaView-5.11.1-MPI-Linux-Python3.9-x86_64/bin/pvpython

import sys
from pathlib import Path
import subprocess
import pandas as pd

# Repo root:  .../SlabPT-emulator
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))  # allow `utils.*` imports

from utils.model_processing import extract_csv

# ---- CLI args ----
MOD_NAME   = str(sys.argv[1])          # e.g. "000"
TIMESTEP1  = int(sys.argv[2])
TIMESTEP2  = int(sys.argv[3])
DEPTHS_ARG = str(sys.argv[4])          # e.g. "5,10,15,..."
SUITE      = str(sys.argv[5]) if len(sys.argv) > 5 else "const-vc"

DEPTHS_KM  = [float(s) for s in DEPTHS_ARG.split(",")]

# ---- Paths to utility scripts ----
DT_script   = ROOT / "src" / "utils" / "compute_slab_cooling.py"
plot_script = ROOT / "src" / "utils" / "plot_T_and_C_fields.py"

# ---- Suite / I/O layout ----
SUITE_ROOT    = ROOT / "subd-model-runs" / SUITE
IN_DIR        = SUITE_ROOT / "run-outputs"
ANALYSIS_ROOT = SUITE_ROOT / "analysis"
ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)

run_name     = f"run_{MOD_NAME}"
RUN_ANALYSIS = ANALYSIS_ROOT / run_name
RUN_ANALYSIS.mkdir(parents=True, exist_ok=True)

# 1) Extract CSV snapshots from pvtu
print("1: CSVs----------------------")
ofull1, t_yr1 = extract_csv(IN_DIR, RUN_ANALYSIS, MOD_NAME, TIMESTEP1)
ofull2, t_yr2 = extract_csv(IN_DIR, RUN_ANALYSIS, MOD_NAME, TIMESTEP2)

# 2) Compute slab cooling ΔT and dT/dt along trench
print("2: Extracting DT-------------")
template = str(RUN_ANALYSIS / "t{}.csv")
OUTDT    = str(RUN_ANALYSIS / f"DT_{TIMESTEP1}_{TIMESTEP2}.csv")

cmd_slab_DT = [
    "python3", str(DT_script),
    "--template", template,
    "--t1", str(TIMESTEP1),
    "--t2", str(TIMESTEP2),
    "--t1-yr", str(t_yr1),
    "--t2-yr", str(t_yr2),
    "--depths-km", *[str(d) for d in DEPTHS_KM],
    "--out", OUTDT,
    "--grid-res-km", "1",
    "--c-thresh", "0.5",
    "--x-min-km", "1600",
]
subprocess.run(cmd_slab_DT, check=True)

# 3) Plot fields + markers
print("3: Plotting-------------------")
field1 = str(RUN_ANALYSIS / f"t{TIMESTEP1}.csv")
field2 = str(RUN_ANALYSIS / f"t{TIMESTEP2}.csv")

FIG_DIR = ANALYSIS_ROOT / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
png_out = str(FIG_DIR / f"{run_name}.DT_{TIMESTEP1}_{TIMESTEP2}.png")
pdf_out = str(FIG_DIR / f"{run_name}.DT_{TIMESTEP1}_{TIMESTEP2}.pdf")

# Params for annotation
params_csv = ROOT / "data" / "params" / (f"params-list.{SUITE}.csv")

try:
    dfp = pd.read_csv(params_csv)
    idx = int(MOD_NAME)  # run_000 -> row 0, etc.
    row = dfp.iloc[idx]
    annot = (
        f"v={row['v_conv']:.2f} cm/yr, "
        f"age_SP={row['age_SP']:.1f} Ma, "
        f"age_OP={row['age_OP']:.1f} Ma, "
        f"dip={row['dip_int']:.1f}°, "
        f"η_UM={float(row['eta_UM']):.2e} Pa·s"
    )
except Exception:
    annot = f"{run_name} (params unavailable)"

cmd_plot = [
    "python3", str(plot_script),
    "--field-csv", field1,
    "--field2-csv", field2,
    "--out", png_out,
    "--markers", OUTDT,
    "--grid-res-km", "1",
    "--xmin-km", "1700", "--xmax-km", "2300", "--ymax-km", "1000",
    "--depth-max-km", "170",
    "--cmap", "coolwarm",
    "--interp", "nearest",
    "--y-origin", "bottom",
    "--annot", annot,
]
subprocess.run(cmd_plot, check=True)

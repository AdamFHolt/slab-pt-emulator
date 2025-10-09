#!/home/holt/software/ParaView-5.11.1-MPI-Linux-Python3.9-x86_64/bin/pvpython

import sys, pathlib, subprocess, csv
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))  

from utils.model_processing import extract_csv

MOD_NAME   = str(sys.argv[1])
TIMESTEP1  = int(sys.argv[2])
TIMESTEP2  = int(sys.argv[3])
DEPTHS_ARG = str(sys.argv[4])              # e.g. "25,50,75"
DEPTHS_KM  = [float(s) for s in DEPTHS_ARG.split(",")]

DT_script   = str(ROOT / "utils" / "compute_slab_cooling.py")
plot_script = str(ROOT / "utils" / "plot_T_and_C_fields.py")

IN_DIR  = '../subd-model-runs/run-outputs/'
OUTCSV_DIR = pathlib.Path(f'../subd-model-runs/run-outputs/run_{MOD_NAME}')
OUTCSV_DIR.mkdir(parents=True, exist_ok=True)

print("1: CSVs----------------------")
ofull1, t_yr1 = extract_csv(IN_DIR, OUTCSV_DIR, MOD_NAME, TIMESTEP1)
ofull2, t_yr2 = extract_csv(IN_DIR, OUTCSV_DIR, MOD_NAME, TIMESTEP2)

print("2: Extracting DT-------------")
template    = str(OUTCSV_DIR / "t{}.csv")
OUTDT_DIR   = pathlib.Path(f'../subd-model-runs/run-outputs/analysis')
OUTDT_DIR.mkdir(parents=True, exist_ok=True)
OUTDT        = str(OUTDT_DIR / f"run_{MOD_NAME}.DT_{TIMESTEP1}_{TIMESTEP2}.csv")

cmd_slab_DT = [
    "python3", DT_script,
    "--template", template,
    "--t1", str(TIMESTEP1), "--t2", str(TIMESTEP2),
    "--t1-yr", str(t_yr1), "--t2-yr", str(t_yr2),
    "--depths-km", *[str(d) for d in DEPTHS_KM],
    "--out", OUTDT,
    "--grid-res-km", "1",
    "--c-thresh", "0.5",
    "--x-min-km", "1600"
]
subprocess.run(cmd_slab_DT, check=True)

# 4) Plot fields + markers
print("3: Plotting-------------------")
field1 = str(OUTCSV_DIR / f"t{TIMESTEP1}.csv")
field2 = str(OUTCSV_DIR / f"t{TIMESTEP2}.csv")
png_out = str(OUTDT_DIR / f"figs/run_{MOD_NAME}.DT_{TIMESTEP1}_{TIMESTEP2}.png")
pdf_out = str(OUTDT_DIR / f"figs/run_{MOD_NAME}.DT_{TIMESTEP1}_{TIMESTEP2}.pdf")


# read params and build annotation string for this run
params_csv = ROOT / "../data/params-list.csv"
try:
    dfp = pd.read_csv(params_csv)
    idx = int(MOD_NAME)                       # run_000 -> row 0, etc.
    row = dfp.iloc[idx]
    annot = (
        f"run {MOD_NAME}  |  "
        f"v={row['v_conv']:.2f} cm/yr, "
        f"age_SP={row['age_SP']:.1f} Ma, "
        f"age_OP={row['age_OP']:.1f} Ma, "
        f"dip={row['dip_int']:.1f}°\n"
        f"η_int={float(row['eta_int']):.2e} Pa·s, "
        f"η_UM={float(row['eta_UM']):.2e} Pa·s, "
        f"ε̇_trans={float(row['eps_trans']):.2e} s⁻¹"
    )
except Exception as e:
    annot = f"run {MOD_NAME} (params unavailable)"



cmd_plot = [
    "python3", plot_script,
    "--field-csv", field1,
    "--field2-csv", field2,
    "--out", png_out,
    # "--out2", pdf_out,
    "--markers", OUTDT,
    "--grid-res-km", "1",
    "--xmin-km", "1700", "--xmax-km", "2300", "--ymax-km", "1000",
    "--depth-max-km", "180",
    "--cmap", "coolwarm",
    "--interp", "nearest",
    "--y-origin", "bottom",
    "--annot", annot,
]

subprocess.run(cmd_plot, check=True)

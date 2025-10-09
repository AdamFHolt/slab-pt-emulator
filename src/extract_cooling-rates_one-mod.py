#!/home/holt/software/ParaView-5.11.1-MPI-Linux-Python3.9-x86_64/bin/pvpython

import sys, pathlib, subprocess, csv

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

# print("(depth_km, x1_km, T1_C, x2_km, T2_C, dT_C, dt_Myr, dTdt_C_per_Myr)")
# with open(OUTDT, "r", newline="") as f:
#     r = csv.DictReader(f)
#     for row in r:
#         d   = float(row["depth_km"])
#         x1  = float(row["x1_km"])
#         t1  = float(row["T1_C"])
#         x2  = float(row["x2_km"])
#         t2  = float(row["T2_C"])
#         dT  = float(row["dT_C"])
#         dt_Myr = float(row["dt_Myr"])
#         rate = float(row["dTdt_C_per_Myr"])
#         print(f"{d:6.1f}, {x1:8.1f}, {t1:8.1f}, {x2:8.1f}, {t2:8.1f}, {dT:8.1f}, {dt_Myr:6.3f}, {rate:9.3f}")


# 4) Plot fields + markers
print("3: Plotting-------------------")
field1 = str(OUTCSV_DIR / f"t{TIMESTEP1}.csv")
field2 = str(OUTCSV_DIR / f"t{TIMESTEP2}.csv")
png_out = str(OUTDT_DIR / f"figs/run_{MOD_NAME}.DT_{TIMESTEP1}_{TIMESTEP2}.png")
pdf_out = str(OUTDT_DIR / f"figs/run_{MOD_NAME}.DT_{TIMESTEP1}_{TIMESTEP2}.pdf")

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
]
subprocess.run(cmd_plot, check=True)

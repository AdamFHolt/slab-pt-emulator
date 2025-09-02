#!/home/holt/software/ParaView-5.11.1-MPI-Linux-Python3.9-x86_64/bin/pvpython
# Extract two CSVs via ParaView, then run slab-top P–T extraction for both, plot.
import sys, pathlib, subprocess
from utils_postprocessing import extract_csv

MOD_NAME   = str(sys.argv[1])
TIMESTEP1  = int(sys.argv[2])
TIMESTEP2  = int(sys.argv[3])
DEPTH_DT   = float(sys.argv[4])  # depth [km] for cooling rate

# 1) CSV extraction via ParaView
IN_DIR  = '../subd-model-runs/run-outputs/'
OUT_DIR = pathlib.Path(f'../subd-model-runs/run-outputs/analysis/run_{MOD_NAME}')
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("CSVs-------------------------------------------------")
ofull1, t_yr1 = extract_csv(IN_DIR, OUT_DIR, MOD_NAME, TIMESTEP1)
ofull2, t_yr2 = extract_csv(IN_DIR, OUT_DIR, MOD_NAME, TIMESTEP2)

# 2) Slab-top P–T extraction (pure Python script)
template = str(OUT_DIR / "{}.csv")
slabtop_pt = str(pathlib.Path(__file__).with_name("slabtop_pt.py"))  # robust path
print("PT paths---------------------------------------------")

def run_pt_once(tstep: int):
    pt_csv = OUT_DIR / f"slabtop_PT_{tstep}.csv"
    if pt_csv.exists():
        print(f"[pt] exists, skipping: {pt_csv}")
        return

    cmd = [
        "python3", slabtop_pt,
        "--template", template,
        "--tmin", str(tstep), "--tmax", str(tstep),
        "--outdir", str(OUT_DIR),
        "--grid-res-m", "1000",
        "--xy-filter", "11",
        "--pt-filter", "21",
        "--pt-xmin-km", "1800",   # only slab-top points with x >= this (km)
    ]

    subprocess.run(cmd, check=True)  

    # verify it wrote the output
    if not pt_csv.exists():
        raise RuntimeError(f"[pt][fail] expected output missing: {pt_csv}")

for t in sorted({TIMESTEP1, TIMESTEP2}):
    run_pt_once(t)

# 3) Compute cooling rate at certain depth, DEPTH_DT

# 4) Make a simple plot
print("Plotting---------------------------------------------")
plot_script = str(pathlib.Path(__file__).with_name("plot_slabtop_and_field.py"))
field1 = str(OUT_DIR / f"{TIMESTEP1}.csv")
field2 = str(OUT_DIR / f"{TIMESTEP2}.csv")
png_out = str(OUT_DIR / f"slabtop_PT_compare_{TIMESTEP1}_{TIMESTEP2}.png")

cmd_plot = [
    "python3", plot_script,
    "--field-csv", field1,
    "--field2-csv", field2,
    "--pt1", str(OUT_DIR / f"slabtop_PT_{TIMESTEP1}.csv"),
    "--pt2", str(OUT_DIR / f"slabtop_PT_{TIMESTEP2}.csv"),
    "--out", png_out,
    "--grid-res-km", "1",
    "--xmin-km", "1600", "--xmax-km", "2300", "--ymax-km", "1000",
    "--depth-max-km", "180",
    "--cmap", "coolwarm",
    "--interp", "nearest",
    "--y-origin", "bottom",
    # "--show-sample",  # uncomment to sanity-check points
]
subprocess.run(cmd_plot, check=True)
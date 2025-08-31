#!/home/holt/software/ParaView-5.11.1-MPI-Linux-Python3.9-x86_64/bin/pvpython
# Extract two CSVs via ParaView, then run slab-top P–T extraction for both.
import sys, pathlib, subprocess
from utils_postprocessing import extract_csv

MOD_NAME   = str(sys.argv[1])
TIMESTEP1  = int(sys.argv[2])
TIMESTEP2  = int(sys.argv[3])

# 1) CSV extraction via ParaView
IN_DIR  = '../subd-model-runs/run-outputs/'
OUT_DIR = pathlib.Path(f'../subd-model-runs/run-outputs/analysis/run_{MOD_NAME}')
OUT_DIR.mkdir(parents=True, exist_ok=True)

ofull1, t_yr1 = extract_csv(IN_DIR, OUT_DIR, MOD_NAME, TIMESTEP1)
ofull2, t_yr2 = extract_csv(IN_DIR, OUT_DIR, MOD_NAME, TIMESTEP2)
print(f"[csv] run_{MOD_NAME} t={TIMESTEP1} → {ofull1} (t={t_yr1/1e6:.3f} Myr)")
print(f"[csv] run_{MOD_NAME} t={TIMESTEP2} → {ofull2} (t={t_yr2/1e6:.3f} Myr)")

# 2) Slab-top P–T extraction (pure Python script)
template = str(OUT_DIR / "{}.csv")
slabtop_pt = str(pathlib.Path(__file__).with_name("slabtop_pt.py"))  # robust path

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
        "--xy-filter", "59", "--pt-filter", "101",
    ]
    print("[pt] Running:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[pt][fail] t={tstep} (exit {res.returncode})")
        if res.stdout: print(res.stdout)
        if res.stderr: print(res.stderr)
        sys.exit(res.returncode)
    print(f"[pt][ok] wrote {pt_csv}")

# de-dup in case TIMESTEP1==TIMESTEP2
for t in sorted({TIMESTEP1, TIMESTEP2}):
    run_pt_once(t)
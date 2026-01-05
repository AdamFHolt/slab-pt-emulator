#!/home/holt/software/ParaView-5.11.1-MPI-Linux-Python3.9-x86_64/bin/pvpython

import sys
from pathlib import Path
import subprocess
import pandas as pd

# Repo root:  .../SlabPT-emulator
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))  # allow `utils.*` imports

from utils.model_processing import extract_csv

# ----------------------------------------------------------------------
# CLI args (POSitional, matching extract_cooling-rates_all-mods.sh):
#
#   1) MOD_NAME   e.g. "000"
#   2) TIMESTEP1  (int)
#   3) TIMESTEP2  (int)
#   4) TRANGE     "start:end" or "start:end:step"  (e.g. "0:10")
#   5) DEPTHS_ARG "0,2,4,...", already expanded by the bash script
#   6) SUITE      "const-vc" or "ramped-vc"
# ----------------------------------------------------------------------
if len(sys.argv) < 6:
    raise SystemExit(
        "Usage: extract_cooling-rates_one-mod.py MOD_NAME TSTEP1 TSTEP2 TRANGE DEPTHS [SUITE]\n"
        "  MOD_NAME  e.g. 000\n"
        "  TSTEP1    e.g. 1\n"
        "  TSTEP2    e.g. 10\n"
        '  TRANGE    e.g. "0:10" or "0:10:1"\n'
        '  DEPTHS    comma-separated depths in km (already expanded), e.g. "0,1,2,...,80"\n'
        "  SUITE     const-vc or ramped-vc (default: const-vc)\n"
    )

MOD_NAME   = str(sys.argv[1])          # e.g. "000"
TIMESTEP1  = int(sys.argv[2])
TIMESTEP2  = int(sys.argv[3])
TRANGE_ARG = str(sys.argv[4])          # e.g. "0:10"
DEPTHS_ARG = str(sys.argv[5])          # e.g. "0,1,2,...,80"
SUITE      = str(sys.argv[6]) if len(sys.argv) > 6 else "const-vc"

# Parse TRANGE: "start:end" or "start:end:step"
parts = TRANGE_ARG.split(":")
if len(parts) == 2:
    t_start, t_end = map(int, parts)
    t_step = 1
elif len(parts) == 3:
    t_start, t_end, t_step = map(int, parts)
else:
    raise SystemExit(f"TRANGE must be start:end or start:end:step, got '{TRANGE_ARG}'")

TIMES_ALLT = list(range(t_start, t_end + 1, t_step))

# Depths (already expanded by the bash script; just split on commas)
DEPTHS_KM = [float(s) for s in DEPTHS_ARG.split(",")]

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

print("1: CSVs + AllT profiles ----------------------")

# Figure out all timesteps for which we need CSV snapshots
needed_timesteps = sorted(set(TIMES_ALLT + [TIMESTEP1, TIMESTEP2]))

# Extract CSVs and store physical times
t_years = {}  # timestep index -> time in years
for k in needed_timesteps:
    ofull, t_yr = extract_csv(IN_DIR, RUN_ANALYSIS, MOD_NAME, k)
    t_years[k] = t_yr

template = str(RUN_ANALYSIS / "t{}.csv")

# ----------------------------------------------------------------------
# T profiles: interface T vs depth at each time in TIMES_ALLT
# ----------------------------------------------------------------------
for k in TIMES_ALLT:
    tmp_out = RUN_ANALYSIS / f"_tmp_DT_{k}_{k}.csv"

    cmd_slab_allT = [
        "python3", str(DT_script),
        "--template", template,
        "--t1", str(k),
        "--t2", str(k),
        "--t1-yr", str(t_years[k]),
        "--t2-yr", str(t_years[k]),
        "--depths-km", *[str(d) for d in DEPTHS_KM],
        "--out", str(tmp_out),
        "--grid-res-km", "1",
        "--c-thresh", "0.5",
        "--x-min-km", "1600",
        "--smooth-x",
        "--smooth-window-km", "14",
        "--smooth-polyorder", "2",
        "--interp", "linear",
    ]
    subprocess.run(cmd_slab_allT, check=True)

    # Condense to interface-only AllT file: time_Myr, depth_km, T_C
    df_tmp = pd.read_csv(tmp_out)
    time_Myr = t_years[k] / 1e6
    df_allT = pd.DataFrame({
        "time_Myr": [time_Myr] * len(df_tmp),
        "depth_km": df_tmp["depth_km"].to_numpy(),
        "T_C":      df_tmp["T2_C"].to_numpy(),  # T2_C = interface T at this timestep
    })
    # confirm not accidentally using raw
    dmax = (df_tmp["T2_C"] - df_tmp["T2_C_raw"]).abs().max()
    print("max |T2_C - T2_C_raw| =", dmax)
    out_allT = RUN_ANALYSIS / f"Tprof_{k}.csv"
    df_allT.to_csv(out_allT, index=False)
    print(f"[DT] wrote: {out_allT}")

    # Optional: clean up the temporary full DT file
    try:
        tmp_out.unlink()
    except OSError:
        pass

# ----------------------------------------------------------------------
# 2) Compute slab cooling ΔT and dT/dt between TIMESTEP1 and TIMESTEP2
# ----------------------------------------------------------------------
print("2: Extracting DT (T1/T2 pair) -------------")

OUTDT = str(RUN_ANALYSIS / f"DT_{TIMESTEP1}_{TIMESTEP2}.csv")

cmd_slab_DT = [
    "python3", str(DT_script),
    "--template", template,
    "--t1", str(TIMESTEP1),
    "--t2", str(TIMESTEP2),
    "--t1-yr", str(t_years[TIMESTEP1]),
    "--t2-yr", str(t_years[TIMESTEP2]),
    "--depths-km", *[str(d) for d in DEPTHS_KM],
    "--out", OUTDT,
    "--grid-res-km", "1",
    "--c-thresh", "0.5",
    "--x-min-km", "1600",
    "--smooth-x",
    "--smooth-window-km", "14",
    "--smooth-polyorder", "2",
]
subprocess.run(cmd_slab_DT, check=True)

# ----------------------------------------------------------------------
# 3) Plot fields + markers at the T1/T2 pair 
# ----------------------------------------------------------------------
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

print("Done.")

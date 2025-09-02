from paraview.simple import *
import numpy as np, os, pathlib, re, sys

def extract_csv(IN_DIR, OUT_DIR, MOD_NAME, TIMESTEP):

    # load in the .pvd file
    soln = f'{IN_DIR}run_{MOD_NAME}/outputs/run_{MOD_NAME}/solution.pvd'
    if not os.path.exists(soln):
        print(f'ERROR: {soln} not found')
        sys.exit(1)

    reader = OpenDataFile(soln)
    times = list(getattr(reader, "TimestepValues", []))
    nt = len(times)

    if TIMESTEP < 0 or TIMESTEP >= nt:
        print(f'ERROR: timestep {TIMESTEP} out of range [0..{nt-1}]')
        sys.exit(1)

    t_yr = times[TIMESTEP]
    ofull = OUT_DIR / f't{TIMESTEP}.csv'

    if ofull.exists():
        print(f"[csv] exists, skipping: {ofull}")
        return str(ofull), t_yr

    w = CreateWriter( str(ofull), reader)
    w.FieldAssociation = "Point Data" # or "Cells"
    w.UpdatePipeline(t_yr)
    del w

    print(f"[csv] wrote {ofull} (t = {t_yr/1e6:.3f} Myr)")
    return str(ofull), t_yr

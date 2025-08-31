from paraview.simple import *
import os,sys
import numpy as np, os, pathlib, re

def extract_csv(IN_DIR, OUT_DIR, MOD_NAME, TIMESTEP):

    # load in the .pvd file
    soln = f'{IN_DIR}run_{MOD_NAME}/outputs/run_{MOD_NAME}/solution.pvd'
    if not os.path.exists(soln):
        print(f'ERROR: {soln} not found')
        sys.exit(1)

    reader = OpenDataFile(soln)
    times = list(getattr(reader, "TimestepValues", []))
    nt = len(times)
    print("----------------------------------------------------")
    print(f'Number of model timesteps for run_{MOD_NAME} = {nt}')

    if TIMESTEP < 0 or TIMESTEP >= nt:
        print(f'ERROR: timestep {TIMESTEP} out of range [0..{nt-1}]')
        sys.exit(1)

    t_yr = times[TIMESTEP]
    ofull = OUT_DIR / f'{TIMESTEP}.csv'

    if ofull.exists():
        print(f'SKIP: {ofull} already exists; not regenerating.')
        print("----------------------------------------------------")
        return  # exit the function if output is already present


    w = CreateWriter( str(ofull), reader)
    w.FieldAssociation = "Point Data" # or "Cells"
    w.UpdatePipeline(t_yr)
    del w

    print(f'Wrote {ofull} for run_{MOD_NAME} at timestep {TIMESTEP} (t = {t_yr/1e6:.3f} Myr)')
    print("----------------------------------------------------")

    return ofull, t_yr

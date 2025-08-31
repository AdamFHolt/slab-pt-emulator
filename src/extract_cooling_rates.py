#!/home/holt/software/ParaView-5.11.1-MPI-Linux-Python3.9-x86_64/bin/pvpython
#
# extract csv files for specified timestep of a .pvd file
#
from paraview.simple import *
import os,sys
import numpy as np, os, pathlib, re

MOD_NAME=str(sys.argv[1])
TIMESTEP=int(sys.argv[2])

# set input/output directories
IN_DIR  = '../subd-model-runs/run-outputs/'
OUT_DIR = pathlib.Path('../subd-model-runs/run-outputs/csv_outputs/run_'+str(MOD_NAME))
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    sys.exit(0)   # exit cleanly if output is already present


w = CreateWriter( str(ofull), reader)
w.FieldAssociation = "Point Data" # or "Cells"
w.UpdatePipeline(t_yr)
del w

print(f'Wrote {ofull} for run_{MOD_NAME} at timestep {TIMESTEP} (t = {t_yr/1e6:.3f} Myr)')
print("----------------------------------------------------")

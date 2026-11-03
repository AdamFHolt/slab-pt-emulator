#!/usr/bin/env python3
"""Velocity divergence inside the incoming-plate crust (x 1100-1600 km, z < 15 km), sh vs shp, at t = 0.5 / 1 / 2 Myr.

Written 2026-11-03. ASPECT advects compositional fields in advective form (dc/dt + u.grad c = 0), so the column
crust content changes at d/dt int c dz = int c div(u) dz (no flux through the top, vy = 0). A discrete velocity
that is not divergence-free inside the crust is a crust sink (div < 0) or source. Prints, per suite and step:
mean crust thickness H, int c div dz (km/Myr, compare with dH/dt from crust_budget.py), the crust-weighted
divergence, the divergence in the top 1 km, and the crust-weighted horizontal strain rate exx (stretching).
Velocities are gridded from point output at 0.1 km, so values are approximate; const-vc-sh is the control.

Usage: crust_divergence.py <run>   (e.g. 072)
"""
import pandas as pd, numpy as np, sys
from scipy.interpolate import griddata
run=sys.argv[1]
for suite in ['const-vc-sh','const-vc-shp']:
  for k in [1,2,4]:
    d=pd.read_csv(f'subd-model-runs/{suite}/analysis/run_{run}/t{k}.csv',usecols=['velocity:0','velocity:1','ocrust','Points:0','Points:1'])
    d['x']=d['Points:0']/1e3; d['z']=1000-d['Points:1']/1e3
    d=d[(d.x>1000)&(d.x<1700)&(d.z<25)]
    d=d.groupby([d.x.round(3),d.z.round(3)]).mean().reset_index(drop=True)
    dx=0.1
    xs=np.arange(1100,1600,dx); zs=np.arange(0,15,dx)
    X,Z=np.meshgrid(xs,zs)
    g={c:griddata((d.x,d.z),d[c],(X,Z)) for c in ['velocity:0','velocity:1','ocrust']}
    vx=g['velocity:0']; vz=-g['velocity:1']   # m/yr, z down
    h=dx*1e3
    div=np.gradient(vx,h,axis=1)+np.gradient(vz,h,axis=0)   # 1/yr
    c=np.clip(g['ocrust'],0,1)
    H=c.sum(0)*dx
    # rate of column crust change from c*div (advective form): d/dt int c dz = int c div dz (no boundary flux)
    src=(c*div).sum(0)*dx*1e6  # km/Myr
    exx=np.gradient(vx,h,axis=1)
    print(f"{suite} run_{run} t={k*0.5}: H {np.nanmean(H):.2f} km | int c*div dz {np.nanmean(src):+.3f} km/Myr | "
          f"mean div in crust {np.nansum(c*div)/np.nansum(c)*1e6:+.3f}/Myr, in top 1 km {np.nanmean(div[zs<1])*1e6:+.3f}/Myr, "
          f"exx(crust) {np.nansum(c*exx)/np.nansum(c)*1e6:+.4f}/Myr")

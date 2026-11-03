#!/usr/bin/env python3
"""Incoming-crust stress (tau_II = 2 eta eps_II from gridded output) by depth band, far field (trench-800..-150 km)
and bending zone (trench-150..-10 km), sh vs shp, against a DP yield with cohesion 5 MPa, phi = asin 0.05.
Written 2026-11-03 to answer "would 5 MPa work?". Needs crust_budget_run_XXX.csv for the trench position.
Usage: crust_stress.py <run> <step>
"""
import pandas as pd, numpy as np, sys
from scipy.interpolate import griddata
run=sys.argv[1]; k=int(sys.argv[2])
PHI=np.deg2rad(2.866)
out=[]
for suite in ['const-vc-sh','const-vc-shp']:
    d=pd.read_csv(f'subd-model-runs/{suite}/analysis/run_{run}/t{k}.csv',usecols=['velocity:0','velocity:1','p','ocrust','viscosity','Points:0','Points:1'])
    d['x']=d['Points:0']/1e3; d['z']=1000-d['Points:1']/1e3
    d=d[(d.x>900)&(d.x<2000)&(d.z<25)]
    d=d.groupby([d.x.round(3),d.z.round(3)]).mean().reset_index(drop=True)
    dx=0.1
    xs=np.arange(1000,1950,dx); zs=np.arange(0,12,dx)
    X,Z=np.meshgrid(xs,zs)
    g={c:griddata((d.x,d.z),d[c],(X,Z)) for c in ['velocity:0','velocity:1','ocrust','p']}
    g['eta']=10**griddata((d.x,d.z),np.log10(d.viscosity),(X,Z))
    h=dx*1e3; S=3.15576e7
    vx=g['velocity:0']/S; vy=g['velocity:1']/S
    a=np.gradient(vx,h,axis=1); b=-np.gradient(vy,h,axis=0); c1=-np.gradient(vx,h,axis=0); c2=np.gradient(vy,h,axis=1)
    div=(a+b)/3; exx,eyy,exy=a-div,b-div,0.5*(c1+c2)
    eps=np.sqrt(0.5*(exx**2+eyy**2+div**2)+exy**2)
    tau=2*g['eta']*eps/1e6
    # trench
    i10=np.argmin(abs(zs-10)); 
    dd=pd.read_csv(f'plots/science-numerical-mods/const-vc-shp/crust_budget_run_{run}.csv')
    xt=float(dd[(dd.suite==suite)&(dd.step==k)].x_trench.iloc[0])
    y5=5*np.cos(PHI)+np.maximum(g['p'],0)/1e6*np.sin(PHI)
    for reg,(x0,x1) in {'far (trench-800..-150)':(xt-800,xt-150),'bend (trench-150..-10)':(xt-150,xt-10)}.items():
        for z0,z1 in [(0,1),(1,2),(2,4),(4,6)]:
            m=(X>x0)&(X<x1)&(Z>=z0)&(Z<z1)&(g['ocrust']>0.5)
            if m.sum()<10: continue
            t=tau[m]
            out.append(dict(suite=suite.replace('const-vc-',''),region=reg,z=f'{z0}-{z1}',tau_med=np.nanmedian(t),tau_p90=np.nanpercentile(t,90),
                            yield5_med=np.nanmedian(y5[m]),frac_over5=np.nanmean(t>y5[m]),eta_med=np.nanmedian(np.log10(g['eta'][m]))))
print(f'run_{run} t={k*0.5} Myr'); print(pd.DataFrame(out).to_string(index=False,float_format='%.2f'))

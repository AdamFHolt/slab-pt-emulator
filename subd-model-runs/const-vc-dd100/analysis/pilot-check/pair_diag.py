#!/usr/bin/env python3
"""Paired coupling diagnostics: const-vc-dd100 run vs const-vc run at the same output step."""
import sys, numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool
from scipy.interpolate import griddata
ROOT=Path("/home/holt/Projects/SlabPT-emulator"); R=ROOT/"subd-model-runs"
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else Path(".")
STEPS=[2,4,10]; XMIN,XMAX,ZMAX,DX=1700.,2600.,260.,2.
P=pd.read_csv(ROOT/"data/params/params-list.const-vc.csv"); P["run"]=[f"{i:03d}" for i in range(400)]

def load(path):
    df=pd.read_csv(path); x=df["Points:0"].to_numpy()/1e3; z=1000-df["Points:1"].to_numpy()/1e3
    m=(x>=XMIN-20)&(x<=XMAX+20)&(z<=ZMAX+20)
    vx=df["velocity:0"].to_numpy()[m]*100; vy=df["velocity:1"].to_numpy()[m]*100
    X,Z=np.meshgrid(np.arange(XMIN,XMAX+DX,DX),np.arange(0,ZMAX+DX,DX)); pts=(x[m],z[m])
    G={k:griddata(pts,v,(X,Z),method="linear") for k,v in dict(T=df["T"].to_numpy()[m]-273.15,C=df["ocrust"].to_numpy()[m],
        OP=df["op"].to_numpy()[m],eta=np.log10(np.clip(df["viscosity"].to_numpy()[m],1e17,1e25)),vx=vx,vy=vy,v=np.hypot(vx,vy)).items()}
    return X,Z,G

def diag(X,Z,G,vc):
    xs,zs=X[0],Z[:,0]; rec={}
    def row(zk): return int(np.argmin(abs(zs-zk)))
    def col(xk): return int(np.clip(np.argmin(abs(xs-xk)),0,xs.size-1))
    xint={}
    for zk in (50,60,80,100,110,125,150,200,250):
        i=row(zk); ok=np.where(np.isfinite(G["C"][i])&(G["C"][i]>=0.5))[0]
        if not ok.size: xint[zk]=np.nan; continue
        j=ok.max(); xint[zk]=xs[j]; rec[f"xint{zk}"]=xs[j]; rec[f"Tint{zk}"]=G["T"][i,j]
        for d in (10,20):
            jj=col(xs[j]+d); rec[f"T{zk}_o{d}"]=G["T"][i,jj]; rec[f"e{zk}_o{d}"]=G["eta"][i,jj]; rec[f"v{zk}_o{d}"]=G["v"][i,jj]
    rec["slab_len_km"]=float(zs[np.where(np.isfinite(G["C"])&(G["C"]>=0.5))[0].max()]) if np.any(G["C"]>=0.5) else np.nan  # deepest crust
    # op-layer entrainment: deepest op>=0.5 within 60 km OP-side of the slab top
    mx=np.full(zs.size,np.nan)
    for i in range(zs.size):
        ok=np.where(np.isfinite(G["C"][i])&(G["C"][i]>=0.5))[0]
        if ok.size:
            j=ok.max(); seg=G["OP"][i,j:min(j+30,xs.size)]; mx[i]=np.nanmax(seg) if np.any(np.isfinite(seg)) else np.nan
    deep=np.where(mx>=0.5)[0]; rec["op_entrain_zmax"]=zs[deep.max()] if deep.size else np.nan
    # OP interior motion: op>=0.5, depth 30-90 km, 20-100 km OP-side of the slab top at 60 km depth
    if np.isfinite(xint.get(60,np.nan)):
        m=(G["OP"]>=0.5)&(Z>=30)&(Z<=90)&(X>=xint[60]+20)&(X<=xint[60]+100)
        if m.sum()>10:
            rec["OP_vx_mean"]=np.nanmean(G["vx"][m]); rec["OP_vy_mean"]=np.nanmean(G["vy"][m])
            rec["OP_v_max"]=np.nanmax(G["v"][m]); rec["OP_v_max_over_vc"]=rec["OP_v_max"]/vc
    # OP thickness (op>=0.5) 100 km and 300 km from the slab top at 60 km depth
    for tag,dx in (("near",100),("far",300)):
        if np.isfinite(xint.get(60,np.nan)):
            j=col(xint[60]+dx); colm=np.where(np.isfinite(G["OP"][:,j])&(G["OP"][:,j]>=0.5))[0]
            rec[f"OP_thick_{tag}"]=zs[colm.max()] if colm.size else np.nan
    return rec

def one(args):
    run,step=args; out=[]
    for suite in ("const-vc-dd100","const-vc"):
        p=R/suite/"analysis"/f"run_{run}"/f"t{step}.csv"
        if not p.is_file(): continue
        vc=float(P.loc[int(run),"v_conv"])
        try: X,Z,G=load(p); rec=diag(X,Z,G,vc)
        except Exception as e: rec=dict(err=str(e))
        rec.update(run=run,step=step,suite=suite); out.append(rec)
    return out

if __name__=="__main__":
    runs=sorted(p.name[4:] for p in (R/"const-vc-dd100/analysis").glob("run_[0-9]*") if p.is_dir())
    jobs=[(r,s) for r in runs for s in STEPS]
    with Pool(min(24,len(jobs))) as pool: recs=[x for lst in pool.map(one,jobs) for x in lst]
    df=pd.DataFrame(recs).merge(P,on="run",how="left").sort_values(["step","run","suite"])
    OUT.mkdir(parents=True,exist_ok=True); df.to_csv(OUT/"pair_diag.csv",index=False); print("wrote",OUT/"pair_diag.csv",df.shape)
    pd.set_option("display.width",300)
    for step in STEPS:
        d=df[df.step==step]; a=d[d.suite=="const-vc-dd100"].set_index("run"); b=d[d.suite=="const-vc"].set_index("run")
        common=a.index.intersection(b.index)
        if not len(common): continue
        print(f"\n===== step {step} ({step*0.5:.0f} Myr): dd100 vs const-vc pair, n={len(common)} =====")
        cols=["age_OP","dip_int","eta_UM","v_conv"]
        tab=a.loc[common,cols].round(1).copy()
        for c,lab in (("OP_v_max_over_vc","OPvmax/vc"),("op_entrain_zmax","op_zmax"),("Tint100","Tint100"),("Tint125","Tint125"),("T100_o10","T100_o10"),("T125_o10","T125_o10"),("e100_o10","e100_o10"),("xint100","xint100"),("xint200","xint200"),("OP_thick_near","OPthk_near")):
            tab[lab+"_dd"]=a.loc[common,c].round(1); tab[lab+"_cv"]=b.loc[common,c].round(1)
        print(tab.sort_values("age_OP",ascending=False).to_string())

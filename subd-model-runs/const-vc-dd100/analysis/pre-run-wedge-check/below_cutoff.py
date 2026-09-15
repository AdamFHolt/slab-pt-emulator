import numpy as np, pandas as pd, sys
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0,"/home/holt/Projects/SlabPT-emulator/subd-model-runs/const-vc-dd100/analysis/pre-run-wedge-check")
import wedge_nose_100km as W
W.DEPTHS=[100,125,150,160,175,190]; W.ZMAX=230.0
def one(run):
    X,Z,G=W.grid(W.load(W.AN/f"run_{run}"/"t20.csv")); xs,zs=X[0],Z[:,0]
    rec=dict(run=run)
    for zk in W.DEPTHS:
        i=int(np.argmin(abs(zs-zk))); ok=np.where(np.isfinite(G["C"][i])&(G["C"][i]>=0.5))[0]
        if not ok.size: continue
        j=ok.max()
        for d in (10,20,40):
            jj=min(j+int(d/W.DX),xs.size-1)
            rec[f"T{zk}_o{d}"]=G["T"][i,jj]; rec[f"v{zk}_o{d}"]=G["v"][i,jj]; rec[f"e{zk}_o{d}"]=G["eta"][i,jj]
    # op-field entrainment: deepest point with op>=0.5 within 60 km OP-side of the slab top
    mx=np.full(zs.size,np.nan)
    for i in range(zs.size):
        ok=np.where(np.isfinite(G["C"][i])&(G["C"][i]>=0.5))[0]
        if ok.size:
            j=ok.max(); seg=G["OP"][i,j:min(j+30,xs.size)]
            mx[i]=np.nanmax(seg) if np.any(np.isfinite(seg)) else np.nan
    deep=np.where(mx>=0.5)[0]; rec["op_entrain_zmax"]=zs[deep.max()] if deep.size else np.nan
    return rec
runs=["010","242","038","156","039","327","282","128","343","135","165","090","286","115","031"]
with Pool(15) as p: recs=p.map(one,runs)
df=pd.DataFrame(recs); P=pd.read_csv(W.ROOT/"data/params/params-list.const-vc.csv"); P["run"]=[f"{i:03d}" for i in range(400)]
df=df.merge(P,on="run")
pd.set_option("display.width",300)
print(df[["run","age_OP","eta_UM","dip_int","v_conv","T100_o10","T150_o10","T160_o10","T175_o10","T190_o10","e160_o10","e175_o10","v160_o10","v160_o20","v175_o10","v175_o40","op_entrain_zmax"]].round(1).to_string(index=False))

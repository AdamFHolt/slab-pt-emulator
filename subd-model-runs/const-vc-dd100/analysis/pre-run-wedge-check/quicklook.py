import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import griddata
ROOT=Path("/home/holt/Projects/SlabPT-emulator"); AN=ROOT/"subd-model-runs/const-vc/analysis"
P=pd.read_csv(ROOT/"data/params/params-list.const-vc.csv")
runs=sys.argv[1].split(","); step=int(sys.argv[2]); out=Path(sys.argv[3])
fig,axes=plt.subplots(len(runs),2,figsize=(14,4.2*len(runs)),squeeze=False)
for r,(axT,axE) in zip(runs,axes):
    df=pd.read_csv(AN/f"run_{r}"/f"t{step}.csv")
    x=df["Points:0"].to_numpy()/1e3; z=1000-df["Points:1"].to_numpy()/1e3
    m=(x>1750)&(x<2350)&(z<220)
    X,Z=np.meshgrid(np.arange(1750,2350,2.),np.arange(0,220,2.))
    T=griddata((x[m],z[m]),df["T"].to_numpy()[m]-273.15,(X,Z),method="linear")
    E=griddata((x[m],z[m]),np.log10(df["viscosity"].to_numpy()[m]),(X,Z),method="linear")
    C=griddata((x[m],z[m]),df["ocrust"].to_numpy()[m],(X,Z),method="linear")
    OP=griddata((x[m],z[m]),df["op"].to_numpy()[m],(X,Z),method="linear")
    p=P.iloc[int(r)]
    for ax,F,cm,lab,lv in ((axT,T,"coolwarm","T [C]",np.arange(0,1500,100)),(axE,E,"viridis","log10 eta",np.arange(18,24.1,0.5))):
        cs=ax.contourf(X,Z,F,levels=lv,cmap=cm,extend="both"); plt.colorbar(cs,ax=ax,label=lab)
        ax.contour(X,Z,C,levels=[0.5],colors="k",linewidths=1.2)
        ax.contour(X,Z,OP,levels=[0.5],colors="w",linewidths=0.8,linestyles="--")
        ax.contour(X,Z,T,levels=[1100,1200,1300],colors=["b","m","r"],linewidths=0.8)
        for zz in (100,125,150): ax.axhline(zz,color="0.4",ls=":",lw=0.8)
        ax.set_ylim(220,0); ax.set_xlabel("x [km]"); ax.set_ylabel("depth [km]"); ax.set_aspect("equal")
    axT.set_title(f"run_{r}  t-step {step}: vc={p.v_conv:.1f} cm/yr, ageSP={p.age_SP:.0f}, ageOP={p.age_OP:.0f} Myr, dip={p.dip_int:.0f}, etaUM={p.eta_UM:.1e}\n"
                  "black: ocrust=0.5 (slab top/crust); white dashed: op field; blue/magenta/red: 1100/1200/1300 C; dotted: 100/125/150 km",fontsize=9)
plt.tight_layout(); plt.savefig(out,dpi=110); print("wrote",out)

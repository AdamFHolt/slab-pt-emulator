import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path; from scipy.interpolate import griddata
ROOT=Path("/home/holt/Projects/SlabPT-emulator"); R=ROOT/"subd-model-runs"; P=pd.read_csv(ROOT/"data/params/params-list.const-vc.csv")
runs=sys.argv[1].split(","); step=int(sys.argv[2]); out=Path(sys.argv[3]); field=sys.argv[4] if len(sys.argv)>4 else "T"
fig,axes=plt.subplots(len(runs),2,figsize=(15,4.0*len(runs)),squeeze=False)
X,Z=np.meshgrid(np.arange(1750,2350,2.),np.arange(0,240,2.))
for r,axr in zip(runs,axes):
    for ax,suite in zip(axr,("const-vc","const-vc-dd100")):
        p=R/suite/"analysis"/f"run_{r}"/f"t{step}.csv"
        if not p.is_file(): ax.set_title(f"{suite} run_{r}: no t{step}.csv"); continue
        df=pd.read_csv(p); x=df["Points:0"].to_numpy()/1e3; z=1000-df["Points:1"].to_numpy()/1e3; m=(x>1730)&(x<2370)&(z<260)
        g=lambda v: griddata((x[m],z[m]),v[m],(X,Z),method="linear")
        T=g(df["T"].to_numpy()-273.15); C=g(df["ocrust"].to_numpy()); OP=g(df["op"].to_numpy()); E=g(np.log10(df["viscosity"].to_numpy()))
        V=g(np.hypot(df["velocity:0"].to_numpy(),df["velocity:1"].to_numpy())*100)
        if field=="T": cs=ax.contourf(X,Z,T,levels=np.arange(0,1500,100),cmap="coolwarm",extend="both"); lab="T [C]"
        elif field=="eta": cs=ax.contourf(X,Z,E,levels=np.arange(18,24.1,0.5),cmap="viridis",extend="both"); lab="log10 eta"
        else: cs=ax.contourf(X,Z,V,levels=np.linspace(0,P.iloc[int(r)].v_conv*1.2,13),cmap="magma",extend="max"); lab="|v| [cm/yr]"
        plt.colorbar(cs,ax=ax,label=lab)
        ax.contour(X,Z,C,levels=[0.5],colors="k",linewidths=1.2); ax.contour(X,Z,OP,levels=[0.5],colors="w",linewidths=0.9,linestyles="--")
        ax.contour(X,Z,T,levels=[1100,1200,1300],colors=["b","m","r"],linewidths=0.7)
        for zz in (100,150): ax.axhline(zz,color="0.3",ls=":",lw=0.8)
        ax.set_ylim(240,0); ax.set_aspect("equal"); ax.set_xlabel("x [km]"); ax.set_ylabel("depth [km]")
        p_=P.iloc[int(r)]; ax.set_title(f"{suite} run_{r} step {step} ({step/2:.0f} Myr): ageOP={p_.age_OP:.0f} dip={p_.dip_int:.0f} etaUM={p_.eta_UM:.1e} vc={p_.v_conv:.1f}",fontsize=9)
plt.tight_layout(); plt.savefig(out,dpi=100); print("wrote",out)

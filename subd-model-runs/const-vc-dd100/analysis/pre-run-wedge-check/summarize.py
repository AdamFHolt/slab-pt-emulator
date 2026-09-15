import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
S=Path("/home/holt/Projects/SlabPT-emulator/subd-model-runs/const-vc-dd100/analysis/pre-run-wedge-check")
df=pd.read_csv(S/"wedge_nose_100km.csv",dtype={"run":str})
pil=[f"{i:03d}" for i in [135,10,68,41,120,242,316,377,268,90,100,165]]; ctl=[f"{i:03d}" for i in [286,115,31]]
df["pilot"]=np.where(df.run.isin(pil),"at-risk",np.where(df.run.isin(ctl),"control",""))
t={0:"0 Myr",2:"1 Myr",10:"5 Myr",20:"10 Myr"}
pd.set_option("display.width",250)
for step in (2,10,20):
    d=df[df.step==step]
    print(f"\n=== step {step} ({t[step]}), n={len(d)} ===")
    for col in ["T_100_o10","T_110_o10","T_125_o10","Tint_100","Tint_125","Tmin_o10_100to125","etamax_o10_100to125","z1100_near","z1100_far","z1200_far","v_110_o10","v_140_o10"]:
        q=d[col].quantile([0,.1,.5,.9,1]).round(1).tolist(); print(f"  {col:22s} min/p10/med/p90/max = {q}")
    # counts
    print("  runs with Tmin_o10(100-125km) < 1000 C:",(d.Tmin_o10_100to125<1000).sum()," < 1100:",(d.Tmin_o10_100to125<1100).sum()," < 1200:",(d.Tmin_o10_100to125<1200).sum())
    print("  runs with z1100_far >= 100 km:",(d.z1100_far>=100).sum()," z1200_far>=100:",(d.z1200_far>=100).sum())
d=df[df.step==20].copy()
cols=["run","pilot","v_conv","age_SP","age_OP","dip_int","eta_UM","Tint_100","T_100_o10","T_110_o10","T_125_o10","etamax_o10_100to125","z1100_near","z1100_far"]
print("\n--- 20 coldest OP-side noses (10 km off slab top, 100-125 km) at 10 Myr ---")
print(d.sort_values("Tmin_o10_100to125")[cols].head(20).round(1).to_string(index=False))
print("\n--- pilot at-risk runs ranked within suite (rank 1 = coldest) ---")
d["rank"]=d.Tmin_o10_100to125.rank()
print(d[d.pilot!=""].sort_values("rank")[["run","pilot","rank","age_OP","eta_UM","dip_int","v_conv","Tmin_o10_100to125","etamax_o10_100to125","z1100_near"]].round(1).to_string(index=False))
# correlations with params
print("\nSpearman corr of Tmin_o10 (10 Myr) with params:")
print(d[["Tmin_o10_100to125","v_conv","age_SP","age_OP","dip_int","eta_UM"]].corr(method="spearman")["Tmin_o10_100to125"].round(2).to_string())
# early time: step 2
d2=df[df.step==2]
print("\n--- 10 coldest at 1 Myr ---")
print(d2.sort_values("Tmin_o10_100to125")[["run","age_OP","eta_UM","dip_int","v_conv","T_100_o10","T_110_o10","T_125_o10","Tmin_o10_100to125","etamax_o10_100to125","z1100_far"]].head(10).round(1).to_string(index=False))

fig,ax=plt.subplots(2,2,figsize=(13,9))
for a,(step,lab) in zip(ax.flat[:3],[(2,"1 Myr"),(10,"5 Myr"),(20,"10 Myr")]):
    dd=df[df.step==step]
    sc=a.scatter(dd.age_OP,dd.Tmin_o10_100to125,c=np.log10(dd.eta_UM),cmap="viridis",s=18)
    m=dd.pilot=="at-risk"; a.scatter(dd.age_OP[m],dd.Tmin_o10_100to125[m],facecolor="none",edgecolor="r",s=80,label="pilot at-risk")
    m=dd.pilot=="control"; a.scatter(dd.age_OP[m],dd.Tmin_o10_100to125[m],facecolor="none",edgecolor="k",s=80,label="pilot control")
    for T0,c in ((1000,"r"),(1100,"m"),(1200,"b")): a.axhline(T0,color=c,ls=":",lw=0.8)
    a.set_xlabel("age_OP [Myr]"); a.set_ylabel("min T, 10 km OP-side of slab top, 100-125 km [C]"); a.set_title(f"const-vc at {lab}"); a.legend(fontsize=8)
    plt.colorbar(sc,ax=a,label="log10 eta_UM")
a=ax.flat[3]; dd=df[df.step==20]
a.scatter(dd.age_OP,dd.z1100_far,s=14,label="1100 C far-field (x=2500 km), 10 Myr")
a.scatter(dd.age_OP,dd.z1100_near,s=14,label="1100 C, 30 km OP-side of slab top @100 km, 10 Myr")
d0=df[df.step==0]; a.scatter(d0.age_OP,d0.z1100_far,s=8,c="k",label="1100 C initial")
a.axhline(100,color="r",ls="--",lw=1,label="dd100 crust cutoff"); a.axhline(150,color="0.5",ls="--",lw=1,label="const-vc cutoff")
a.set_ylim(200,20); a.set_xlabel("age_OP [Myr]"); a.set_ylabel("depth of 1100 C isotherm [km]"); a.legend(fontsize=7); a.set_title("OP lid depth")
plt.tight_layout(); plt.savefig(S/"wedge_nose_summary.png",dpi=110); print("wrote fig")

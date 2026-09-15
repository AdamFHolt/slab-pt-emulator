#!/usr/bin/env python3
"""Wedge-nose state at 100-125 km depth in const-vc: how cold/stiff is the OP-side mantle
that the slab would be coupled to if the weak crust stops at 100 km (const-vc-dd100)?"""
import sys, numpy as np, pandas as pd
from pathlib import Path
from multiprocessing import Pool
from scipy.interpolate import griddata

ROOT = Path("/home/holt/Projects/SlabPT-emulator")
AN = ROOT / "subd-model-runs/const-vc/analysis"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
STEPS = [0, 2, 10, 20]                 # 0, 1, 5, 10 Myr
XMIN, XMAX, ZMAX, DX = 1700.0, 2600.0, 200.0, 2.0
DEPTHS = [80, 90, 100, 110, 120, 125, 140]
OFFS = [10, 20, 40]                    # km horizontally onto the OP side of the slab top
YMAX_KM = 1000.0

def load(path):
    df = pd.read_csv(path)
    x = df["Points:0"].to_numpy() / 1e3
    z = YMAX_KM - df["Points:1"].to_numpy() / 1e3
    m = (x >= XMIN - 20) & (x <= XMAX + 20) & (z <= ZMAX + 20)
    vx, vy = df["velocity:0"].to_numpy()[m], df["velocity:1"].to_numpy()[m]
    return dict(x=x[m], z=z[m], T=df["T"].to_numpy()[m] - 273.15,
                C=df["ocrust"].to_numpy()[m], OP=df["op"].to_numpy()[m],
                eta=np.log10(np.clip(df["viscosity"].to_numpy()[m], 1e17, 1e25)),
                v=np.hypot(vx, vy) * 100.0)   # m/yr -> cm/yr

def grid(f):
    X, Z = np.meshgrid(np.arange(XMIN, XMAX + DX, DX), np.arange(0, ZMAX + DX, DX))
    pts = (f["x"], f["z"])
    G = {k: griddata(pts, f[k], (X, Z), method="linear") for k in ("T", "C", "OP", "eta", "v")}
    return X, Z, G

def one(args):
    run, step = args
    p = AN / f"run_{run}" / f"t{step}.csv"
    if not p.is_file():
        return None
    try:
        X, Z, G = grid(load(p))
    except Exception as e:
        return dict(run=run, step=step, err=str(e))
    xs, zs = X[0, :], Z[:, 0]
    rec = dict(run=run, step=step)
    def row(zk): return int(np.argmin(np.abs(zs - zk)))
    def col(xk): return int(np.clip(np.argmin(np.abs(xs - xk)), 0, xs.size - 1))
    xint = {}
    for zk in DEPTHS:
        i = row(zk)
        ok = np.where(np.isfinite(G["C"][i]) & (G["C"][i] >= 0.5))[0]
        if ok.size == 0:
            xint[zk] = np.nan
            rec[f"xint_{zk}"] = np.nan
            continue
        j = int(ok.max()); xint[zk] = xs[j]
        rec[f"xint_{zk}"] = xs[j]
        rec[f"Tint_{zk}"] = G["T"][i, j]
        for d in OFFS:
            jj = col(xs[j] + d)
            rec[f"T_{zk}_o{d}"] = G["T"][i, jj]
            rec[f"eta_{zk}_o{d}"] = G["eta"][i, jj]
            rec[f"v_{zk}_o{d}"] = G["v"][i, jj]
            rec[f"op_{zk}_o{d}"] = G["OP"][i, jj]
    # lid depth next to the slab: at x = xint(100 km) + 30 / 60 km, and far-field (x=2500)
    for tag, xk in (("near", xint.get(100, np.nan) + 30), ("mid", xint.get(100, np.nan) + 60), ("far", 2500.0)):
        if not np.isfinite(xk):
            continue
        j = col(xk); Tcol = G["T"][:, j]
        for Tc in (1000, 1100, 1200, 1300):
            above = np.where(np.isfinite(Tcol) & (Tcol >= Tc))[0]
            rec[f"z{Tc}_{tag}"] = zs[above.min()] if above.size else np.nan
    # coldest OP-side point 10 km off the interface over 100-125 km
    Ts = [rec.get(f"T_{z}_o10", np.nan) for z in (100, 110, 120, 125)]
    rec["Tmin_o10_100to125"] = np.nanmin(Ts) if np.any(np.isfinite(Ts)) else np.nan
    etas = [rec.get(f"eta_{z}_o10", np.nan) for z in (100, 110, 120, 125)]
    rec["etamax_o10_100to125"] = np.nanmax(etas) if np.any(np.isfinite(etas)) else np.nan
    return rec

if __name__ == "__main__":
    runs = sorted(p.name[4:] for p in AN.glob("run_[0-9][0-9][0-9]") if p.is_dir())
    jobs = [(r, s) for r in runs for s in STEPS]
    with Pool(32) as pool:
        recs = [r for r in pool.imap_unordered(one, jobs, chunksize=4) if r]
    df = pd.DataFrame(recs).sort_values(["run", "step"])
    P = pd.read_csv(ROOT / "data/params/params-list.const-vc.csv")
    P["run"] = [f"{i:03d}" for i in range(len(P))]
    df = df.merge(P, on="run", how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "wedge_nose_100km.csv", index=False)
    print("wrote", OUT / "wedge_nose_100km.csv", df.shape, "errors:", df["err"].notna().sum() if "err" in df else 0)

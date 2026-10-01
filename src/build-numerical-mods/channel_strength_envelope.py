#!/usr/bin/env python3
"""Strength envelopes for the plate-interface channel (the 6 km ocrust layer, z < 150 km) along const-vc
slab-top P-T paths: a pre-build check of the channel rheology options for const-vc-shp.

Channel options compared (stress = second invariant sigma_II, simple shear across the channel at the
convergence rate, eps_II = v / (2 h)):
  sh            fixed 1e20 Pa s (viscosity window 0.995e20-1.005e20)
  shp as built  Drucker-Prager (C = 1 MPa, sin phi = 0.05) on the mantle's olivine composite law,
                clamped to 2.5e18-1e21 Pa s (each group's median olivine prefactors, they vary with eta_UM)
  shp-qtz       Drucker-Prager as above on wet-quartzite dislocation creep (Hirth et al. 2001:
                log10 A = -11.2 MPa^-n s^-1, n = 4, Q = 135 kJ/mol, water-fugacity exponent m = 1,
                fixed f_H2O), clamped to 2.5e18-2.5e23 Pa s (the ceiling is no longer a control)

ASPECT forms coded here (visco_plastic, 2D) -- confirm against the 3.0 source on TACC with
  grep -n "std::pow\\|exp" $WORK/software/aspect/source/material_model/rheology/dislocation_creep.cc
  grep -n "yield_stress" $WORK/software/aspect/source/material_model/rheology/drucker_prager.cc
  creep        eta = 0.5 A^(-1/n) eps_II^((1-n)/n) exp((E + P V) / (n R T')), T' = T + 9.24e-9 K/Pa * P
               (the .prm's 'Adiabat temperature gradient for viscosity'); composite = harmonic sum
  yield (2D)   tau_y = C cos(phi) + max(P, 0) sin(phi); if 2 eta eps_II > tau_y: eta = tau_y / (2 eps_II)
  clamp        eta -> min(max(eta, eta_min), eta_max) AFTER yielding
Lab -> ASPECT prefactor (uniaxial test, invariant model): eps_II = (sqrt3/2) eps_ax, sigma_II = dsigma / sqrt3,
so A_ASPECT = A_lab f^m 3^((n+1)/2) / 2 in Pa^-n s^-1. The script recomputes the quartzite stress in lab
form and in ASPECT form from the .prm-ready numbers and stops if they disagree.

P = rho g z (rho 3300, g 9.81; dynamic pressure ignored). T paths: median slab-top T(z) of the const-vc
runs in each v_conv tercile at 1 / 5 / 10 Myr (no shear heating -- heated paths are warmer, so the
creep branch is an upper bound on strength).

Output: plots/science-numerical-mods/const-vc-shp/channel_strength_envelope.{pdf,png}; prints the .prm
strings for the quartzite channel and the brittle-ductile transition per group and time.
Usage:  env/bin/python src/build-numerical-mods/channel_strength_envelope.py [--fH2O 1e9] [--h 6e3]
"""
import argparse
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "emulator", "science"))
import emu_style as S  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--fH2O", type=float, default=1e9, help="water fugacity, Pa (fixed; band 0.3x-3x drawn)")
ap.add_argument("--h", type=float, default=6e3, help="channel thickness, m")
ap.add_argument("--C", type=float, default=1e6)
ap.add_argument("--mu", type=float, default=0.05, help="sin(phi)")
args = ap.parse_args()
S.apply_style()

R, RHO, G, H = 8.314, 3300.0, 9.81, args.h
ADIA = 9.24e-9                                   # K/Pa, 'Adiabat temperature gradient for viscosity'
SPY = 3.15576e7
PHI = np.arcsin(args.mu)
ETA_MIN = 2.5e18
Z = np.arange(1.0, 100.01, 1.0)                  # km
P = RHO * G * Z * 1e3

# wet quartzite, Hirth et al. (2001)
QTZ = dict(logA=-11.2, n=4.0, Q=135e3, m=1.0)


def qtz_aspect_prefactor(f):
    a_lab_pa = 10 ** QTZ["logA"] * 1e-6 ** QTZ["n"] * (f / 1e6) ** QTZ["m"]   # MPa^-n s^-1 with f in MPa -> Pa^-n s^-1
    return a_lab_pa * 3 ** ((QTZ["n"] + 1) / 2) / 2


def eta_creep(A, n, E, V, eps, T_K, p):
    """ASPECT power-law creep viscosity."""
    Tv = T_K + ADIA * p
    return 0.5 * A ** (-1.0 / n) * eps ** ((1.0 - n) / n) * np.exp((E + p * V) / (n * R * Tv))


def sigma_qtz_lab(eps_II, T_K, f):
    """Same law in lab form: axial strain rate and differential stress, converted to invariants."""
    eps_ax = 2.0 * eps_II / np.sqrt(3.0)
    A = 10 ** QTZ["logA"] * (f / 1e6) ** QTZ["m"]                                 # MPa^-n s^-1
    dsig = (eps_ax / (A * np.exp(-QTZ["Q"] / (R * (T_K + ADIA * P))))) ** (1 / QTZ["n"])   # MPa
    return dsig * 1e6 / np.sqrt(3.0)


def tau_yield(p):
    return args.C * np.cos(PHI) + np.maximum(p, 0) * np.sin(PHI)


def channel(eta_visc, eps, eta_min, eta_max):
    """ASPECT order: viscous -> yield -> clamp. Returns (eta, stress, branch: 0 viscous, 1 yield, 2 floor, 3 ceiling)."""
    ty = tau_yield(P)
    yielded = 2 * eta_visc * eps > ty
    eta = np.where(yielded, ty / (2 * eps), eta_visc)
    br = np.where(yielded, 1, 0)
    br = np.where(eta < eta_min, 2, np.where(eta > eta_max, 3, br))
    eta = np.clip(eta, eta_min, eta_max)
    return eta, 2 * eta * eps, br


# ---------------------------------------------------------------- const-vc paths and olivine prefactors
c = np.load(os.path.join(ROOT, "subd-model-runs", "const-vc", "analysis", "slabtop_record_0-100km_nan.npz"))
ids, TREC, ZG = c["run_ids"], c["T"], c["depth_km"]
par = pd.read_csv(os.path.join(ROOT, "data", "params", "params-list.const-vc.csv"))
v_all = par.v_conv.values[ids]
edges = np.quantile(v_all, [0, 1 / 3, 2 / 3, 1])
GROUPS = []
for i in range(3):
    m = (v_all >= edges[i]) & (v_all <= edges[i + 1]) if i == 2 else (v_all >= edges[i]) & (v_all < edges[i + 1])
    GROUPS.append(dict(mask=m, v=np.median(v_all[m]), lo=edges[i], hi=edges[i + 1]))

rx = {k: re.compile(rf"set Prefactors for {k} creep\s*=\s*background:\s*([0-9.eE+-]+)") for k in ("dislocation", "diffusion")}
ol = {}
for f in glob.glob(os.path.join(ROOT, "subd-model-runs", "const-vc-sh", "run-inputs", "run_[0-9][0-9][0-9]", "run_*.prm")):
    txt = open(f).read()
    ol[int(os.path.basename(f)[4:7])] = tuple(float(rx[k].search(txt).group(1)) for k in ("dislocation", "diffusion"))
for g in GROUPS:
    pre = np.array([ol[i] for i in ids[g["mask"]] if i in ol])
    g["A_dis"], g["A_dif"] = np.median(pre[:, 0]), np.median(pre[:, 1])
    g["n_runs"] = g["mask"].sum()

TIMES = {1.0: 2, 5.0: 10, 10.0: 20}
for g in GROUPS:
    g["T"] = {t: np.interp(Z, ZG, np.nanmedian(TREC[g["mask"], k, :], axis=0)) for t, k in TIMES.items()}

# ---------------------------------------------------------------- implementation check: lab form == ASPECT form
A_Q = qtz_aspect_prefactor(args.fH2O)
for g in GROUPS:
    eps = g["v"] / 100 / SPY / (2 * H)
    Tk = g["T"][5.0] + 273.15
    s_asp = 2 * eta_creep(A_Q, QTZ["n"], QTZ["Q"], 0.0, eps, Tk, P) * eps
    s_lab = sigma_qtz_lab(eps, Tk, args.fH2O)
    rel = np.nanmax(np.abs(s_asp / s_lab - 1))
    assert rel < 1e-9, f"lab vs ASPECT quartzite stress disagree: {rel}"
print(f"check: quartzite stress, lab form == ASPECT form (max rel diff < 1e-9) for all groups")
print(f"\n.prm strings, ocrust phase 1 (z < 150 km), f_H2O = {args.fH2O / 1e9:g} GPa:")
print(f"  Prefactors for dislocation creep : {A_Q:.6e}   (Pa^-4 s^-1, incl. 3^2.5/2 invariant factor)")
print(f"  Stress exponents for dislocation creep : {QTZ['n']:g}    Activation energies : {QTZ['Q']:.0f}   Activation volumes : 0")
print(f"  Prefactors for diffusion creep : 1e-40 (off)    Minimum / Maximum viscosity : 2.5e18 / 2.5e23")
print(f"  Cohesions : {args.C:g}   Angles of internal friction : {np.degrees(PHI):.3f} deg")

# ---------------------------------------------------------------- evaluate
OPTS = ["sh", "shp as built", "shp-qtz"]
OPT_KW = {"sh": dict(color="0.55", lw=1.1), "shp as built": dict(color="#0072B2", lw=1.1, ls="--"),
          "shp-qtz": dict(color="#D55E00", lw=1.8)}
print("\nbrittle-ductile transition of shp-qtz (deepest yielded depth above the first creep depth), "
      "and stress at 40 / 80 km (MPa): sh | as built | qtz")
for g in GROUPS:
    eps = g["v"] / 100 / SPY / (2 * H)
    g["eps"] = eps
    g["res"] = {}
    for t in TIMES:
        Tk = g["T"][t] + 273.15
        r = {}
        r["sh"] = channel(np.full_like(Z, 1e20), eps, 0.995e20, 1.005e20)
        e_ol = 1 / (1 / eta_creep(g["A_dis"], 3.5, 530e3, 18e-6, eps, Tk, P) + 1 / eta_creep(g["A_dif"], 1.0, 375e3, 4e-6, eps, Tk, P))
        r["shp as built"] = channel(e_ol, eps, ETA_MIN, 1e21)
        r["shp-qtz"] = channel(eta_creep(A_Q, 4.0, QTZ["Q"], 0.0, eps, Tk, P), eps, ETA_MIN, 2.5e23)
        r["qtz_creep"] = 2 * eta_creep(A_Q, 4.0, QTZ["Q"], 0.0, eps, Tk, P) * eps
        r["qtz_band"] = [2 * eta_creep(qtz_aspect_prefactor(args.fH2O * s), 4.0, QTZ["Q"], 0.0, eps, Tk, P) * eps for s in (3.0, 1 / 3.0)]
        g["res"][t] = r
        br = r["shp-qtz"][2]
        creep = np.where(br == 0)[0]
        zbd = Z[creep[0]] if creep.size else np.nan
        Tbd = g["T"][t][creep[0]] if creep.size else np.nan
        i40, i80 = 39, 79
        print(f"  v {g['v']:.1f} cm/yr ({g['lo']:.1f}-{g['hi']:.1f}, n={g['n_runs']}), {t:>4g} Myr: BDT {zbd:4.0f} km at {Tbd:4.0f} C | "
              f"40 km {r['sh'][1][i40]/1e6:5.1f} {r['shp as built'][1][i40]/1e6:6.1f} {r['shp-qtz'][1][i40]/1e6:6.1f} | "
              f"80 km {r['sh'][1][i80]/1e6:5.1f} {r['shp as built'][1][i80]/1e6:6.1f} {r['shp-qtz'][1][i80]/1e6:6.1f}")

# ---------------------------------------------------------------- figure
fig, axs = plt.subplots(4, 3, figsize=(7.2, 9.0), sharey=True, facecolor="white")
for j, g in enumerate(GROUPS):
    a0, a1, a2, a3 = axs[:, j]
    for t in TIMES:
        a0.plot(g["T"][t], Z, color=S.time_color(t), lw=1.2, label=f"{t:g} Myr")
    a0.set_title(f"v$_{{conv}}$ {g['lo']:.1f}-{g['hi']:.1f} cm/yr (median {g['v']:.1f}, n={g['n_runs']})", fontsize=7)
    a0.set_xlim(0, 900)
    r = g["res"][5.0]
    a1.fill_betweenx(Z, r["qtz_band"][0] / 1e6, r["qtz_band"][1] / 1e6, color="#D55E00", alpha=0.12, lw=0)
    a1.plot(r["qtz_creep"] / 1e6, Z, color="#D55E00", lw=0.7, ls=":")
    a1.plot(tau_yield(P) / 1e6, Z, color="k", lw=0.7, ls=":")
    for o in OPTS:
        a1.plot(r[o][1] / 1e6, Z, **OPT_KW[o])
        a2.plot(np.log10(r[o][0]), Z, **OPT_KW[o])
        a3.plot(r[o][1] * g["v"] / 100 / SPY * 1e3, Z, **OPT_KW[o])                  # tau * v, mW/m^2
    for t in (1.0, 10.0):                                                          # T sensitivity of the qtz channel
        rr = g["res"][t]["shp-qtz"]
        a1.plot(rr[1] / 1e6, Z, color=S.time_color(t), lw=0.8)
        a3.plot(rr[1] * g["v"] / 100 / SPY * 1e3, Z, color=S.time_color(t), lw=0.8)
    a1.set_xlim(0, 160); a2.set_xlim(18, 23.5); a3.set_xlim(0, 400)
    for a in (a0, a1, a2, a3):
        a.axhspan(80, 100, color="0.93", lw=0, zorder=0)
for a, lab in zip(axs[:, 0], ["Depth (km)"] * 4):
    a.set_ylabel(lab)
axs[0, 0].set_ylim(100, 0)
axs[0, 1].set_xlabel("Slab-top T, const-vc median (" + S.DEG + "C)")
axs[1, 1].set_xlabel(r"Channel stress $\sigma_{II}$ (MPa), 5 Myr path")
axs[2, 1].set_xlabel(r"log$_{10}$ channel viscosity (Pa s), 5 Myr path")
axs[3, 1].set_xlabel(r"Shear-heating flux $\tau v$ (mW m$^{-2}$)")
from matplotlib.lines import Line2D  # noqa: E402
h = [Line2D([0], [0], label=o, **OPT_KW[o]) for o in OPTS]
h += [Line2D([0], [0], color="k", lw=0.7, ls=":", label="friction 1 MPa + 0.05 P"),
      Line2D([0], [0], color="#D55E00", lw=0.7, ls=":", label=f"wet qtz creep, f$_{{H2O}}$ {args.fH2O/1e9:g} GPa (band x3)")]
h += [Line2D([0], [0], color=S.time_color(t), lw=1.0, label=f"{t:g} Myr path") for t in TIMES]
fig.legend(handles=h, loc="lower center", ncol=4, fontsize=6.3, frameon=False)
fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.11, hspace=0.38, wspace=0.08)
out = os.path.join(ROOT, "plots", "science-numerical-mods", "const-vc-shp")
os.makedirs(out, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out, f"channel_strength_envelope.{ext}"), facecolor="white", **({"dpi": 300} if ext == "png" else {}))
print("\nwrote", os.path.join(out, "channel_strength_envelope.{pdf,png}"))

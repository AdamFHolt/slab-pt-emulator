#!/usr/bin/env python3
# build_runs.py  -----------------------------
# Expand LHS rows into template-based run dirs
# -------------------------------------------
import numpy as np, json, os, pathlib, re
from rheology_utils import prefactors
from input_geometry import make_plate_inputs     

LHS_FILE      = "../data/params-list.npy"
TEMPLATE_FILE = "../data/model_template.prm"   # tokenised .prm
OUT_DIR       = pathlib.Path("../subd-model-runs")
PLATE_THICK_M = 125000.0             # *** fixed plate thickness ***

COLS = [
    "v_conv","age_SP","age_OP","dip_int",
    "eta_int","mu_lith","eta_UM","eps_trans",
]

# -- load design -------------------------------------------------
X   = np.load(LHS_FILE)
rows = [dict(zip(COLS, row)) for row in X]

with open(TEMPLATE_FILE) as f:
    tmpl = f.read()
token_pat = re.compile(r"\$\$(\w+)\$\$")

# -- make folder for input structures (i.e. temp and comp text files)
input_dir = OUT_DIR / "initial-structures"
input_dir.mkdir(parents=True, exist_ok=True)

# -- loop & write ------------------------------------------------
for idx, row in enumerate(rows):

    run_dir = OUT_DIR / f"run_{idx:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # compute viscous flow prefactors
    visc_prefactors = prefactors(eta_um=row["eta_UM"], eps_trans=row["eps_trans"])

    # make plate inputs
    temp_name, comp_name = make_plate_inputs(
        dip=row["dip_int"],
        age_sp=row["age_SP"],
        age_op=row["age_OP"],
        plate_thick=PLATE_THICK_M,
        out_dir=input_dir,
    )

    text = tmpl
    for tok in token_pat.findall(tmpl):

        if tok == "MODELNAME":
            text = text.replace(f"$${tok}$$", "run_{idx:03d}")
        elif tok == "TEMPINPUTNAME":
            text = text.replace(f"$${tok}$$", temp_name)
        elif tok == "COMPINPUTNAME":
            text = text.replace(f"$${tok}$$", comp_name)
        elif tok == "CONVRATE":
            text = text.replace(f"$${tok}$$", str(row["v_conv"]))
        elif tok == "ANG_FRICTION":
            ang = np.arcsin(row["mu_lith"]) # rads
            text = text.replace(f"$${tok}$$", str(np.rad2deg(ang)))
        elif tok == "COHESION":
            ang = np.arcsin(row["mu_lith"]) # rads
            fixed_cohesion = 10e6  # Pa
            text = text.replace(f"$${tok}$$", str(fixed_cohesion/np.cos(ang)))
        elif tok == "ETAINT1":
            val = row["eta_int"] - 1.e17
            text = text.replace(f"$${tok}$$", str(val))
        elif tok == "ETAINT2":
            val = row["eta_int"] + 1.e17
            text = text.replace(f"$${tok}$$", str(val))
        elif tok == "ADISLCREEP":
            val = visc_prefactors["Adisl"]
            text = text.replace(f"$${tok}$$", str(val))
        elif tok == "ADIFFCREEP":
            val = visc_prefactors["Adiff"]
            text = text.replace(f"$${tok}$$", str(val))            
        elif tok == "ADIFFCREEP_LM":
            val = visc_prefactors["Adiff_lm"]
            text = text.replace(f"$${tok}$$", str(val))
        else:
            print(tok)
            raise KeyError(f"Token {tok} not in design columns")

    exit()

    # with open(run_dir/"model.prm", "w") as f:
    #     f.write(text)

#     # -- compose comp/temp filenames ----------------------------
#     age_sp = int(row["age_SP"])
#     age_op = int(row["age_OP"])
#     dip    = int(row["dip_deg"])
#     base   = (f"crust-dip{dip}_sp-age{age_sp}_op-age{age_op}_"
#               f"plate-thick{PLATE_THICK_M:.0f}")

#     for kind in ("comp", "temp"):
#         (run_dir/f"{kind}_{base}.txt").touch()

# print(f"Finished:  {len(rows)}  run directories under {OUT_DIR}")

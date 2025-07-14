#!/usr/bin/env python3
# build_runs.py  -----------------------------
# Expand LHS rows into template-based run dirs
# -------------------------------------------
import numpy as np, json, os, pathlib, re

LHS_FILE      = "../data/params_lhs.npy"
TEMPLATE_FILE = "../data/model_template.prm"   # tokenised .prm
OUT_DIR       = pathlib.Path("model_runs")
PLATE_THICK_M = 125_000.0             # *** fixed plate thickness ***

COLS = [
    "v_conv","age_SP","age_OP","dip_deg",
    "eta_int","mu_lith","eta_UM","eps_plaw"
]

# -- load design -------------------------------------------------
X   = np.load(LHS_FILE)
rows = [dict(zip(COLS, row)) for row in X]

with open(TEMPLATE_FILE) as f:
    tmpl = f.read()
token_pat = re.compile(r"\$\$(\w+)\$\$")

# -- loop & write ------------------------------------------------
for idx, row in enumerate(rows):
    run_dir = OUT_DIR / f"run_{idx:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # -- substitute tokens in .prm ------------------------------
    text = tmpl
    for tok in token_pat.findall(tmpl):
        if tok not in row:
            raise KeyError(f"Token {tok} not in design columns")
        val = int(row[tok]) if tok in ("age_SP","age_OP","dip_deg") else row[tok]
        text = text.replace(f"$${tok}$$", str(val))

    with open(run_dir/"model.prm", "w") as f:
        f.write(text)

    # -- compose comp/temp filenames ----------------------------
    age_sp = int(row["age_SP"])
    age_op = int(row["age_OP"])
    dip    = int(row["dip_deg"])
    base   = (f"crust-dip{dip}_sp-age{age_sp}_op-age{age_op}_"
              f"plate-thick{PLATE_THICK_M:.0f}")

    for kind in ("comp", "temp"):
        (run_dir/f"{kind}_{base}.txt").touch()

print(f"Finished:  {len(rows)}  run directories under {OUT_DIR}")

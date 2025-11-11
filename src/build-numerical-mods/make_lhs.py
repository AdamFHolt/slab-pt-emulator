#!/usr/bin/env python3

import numpy as np
from pyDOE2 import lhs
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "params"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1.  Parameter ranges 
RANGES = {
    # name         (min,    max,   'lin' | 'log')
    "v_conv"    : ( 1.0 ,   8.0 ,  'lin'),  # cm/yr  (convergence rate)
    "age_SP"    : ( 40.0 ,  110.0, 'lin'),  # Myr
    "age_OP"    : ( 10.0 ,  110.0, 'lin'),  # Myr
    "dip_int"   : ( 25.0 ,  75.0,  'lin'),  # degrees (slab dip angle)
    "eta_UM"    : ( 5e19 ,  1e21,  'log'),  # Pa·s  (upper-mantle reference)
}

N_SAMPLES = 400
SEED      = 42

# 2. rounding rules

def round_sig(x, sig=5):
    x = np.asarray(x, dtype=float)
    def _round_one(v):
        if not np.isfinite(v) or v == 0.0:
            return 0.0
        dec = int(sig - 1 - np.floor(np.log10(abs(v))))
        return np.round(v, dec)
    return np.vectorize(_round_one, otypes=[float])(x)

ROUND_RULE = {
    "v_conv"  :  lambda x: np.round(x, 4), 
    "age_SP"  :  lambda x: np.round(x, 2),  
    "age_OP"  :  lambda x: np.round(x, 2),  
    "dip_int" :  lambda x: np.round(x, 2),  
    "eta_UM"  :  lambda x: round_sig(x, 5), 
}

# 3.  Generate unit-cube Latin-hypercube 
unit = lhs(len(RANGES), samples=N_SAMPLES, criterion='maximin', random_state=SEED) # N×5 array of [0,1] values

# 4.  Scale columns to physical space
cols, data = [], []
for i, (name, (lo, hi, scale)) in enumerate(RANGES.items()):
    u = unit[:, i]
    if scale == 'lin':
        vals = lo + u * (hi - lo)
    else:                               # logarithmic
        vals = 10 ** (np.log10(lo) + u * (np.log10(hi) - np.log10(lo)))

    if name in ROUND_RULE:
        vals = ROUND_RULE[name](vals)

    cols.append(name)
    data.append(vals)

# 5.  Save to .csv and .npy 
X = np.column_stack(data) 
df = pd.DataFrame(X, columns=cols)
df.to_csv(DATA_DIR / "params-list.csv", index=False, float_format="%.6e")
np.save(DATA_DIR / "params-list.npy", X)
print(f"[OK] Saved {df.shape} → {DATA_DIR}/params-list.[csv|npy]")

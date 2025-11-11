#!/usr/bin/env python3

import numpy as np
from pyDOE2 import lhs
import pandas as pd

# 1.  Parameter ranges 
RANGES = {
    # name         (min,    max,   'lin' | 'log')
    "v_conv"    : ( 1.0 ,   8.0 ,  'lin'),  # cm/yr  (convergence rate)
    "age_SP"    : ( 50.0 ,  110.0, 'lin'),  # Myr
    "age_OP"    : ( 10.0 ,  110.0, 'lin'),  # Myr
    "dip_int"   : ( 25.0 ,  75.0,  'lin'),  # degrees (slab dip angle)
    "eta_UM"    : ( 5e19 ,  1e21,  'log'),  # Pa·s  (upper-mantle reference)
}

N_SAMPLES = 500
SEED      = 42

# rounding rules
def round_sig(x, sig=5):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide='ignore'):
        mags = np.floor(np.log10(np.abs(x)))
    mags[~np.isfinite(mags)] = 0
    return np.round(x, decimals=(sig - 1 - mags).astype(int))

ROUND_RULE = {
    "v_conv"  :  lambda x: np.round(x, 4), 
    "age_SP"  :  lambda x: np.round(x, 2),  
    "age_OP"  :  lambda x: np.round(x, 2),  
    "dip_int" :  lambda x: np.round(x, 2),  
    "eta_UM"  :  lambda x: round_sig(x, 5), 
}

# 2.  Generate unit-cube Latin-hypercube 
unit = lhs(len(RANGES), samples=N_SAMPLES, criterion='maximin', random_state=SEED) # N×5 array of [0,1] values

# 3.  Scale columns to physical space
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

X = np.column_stack(data) # arrange into N×8 array

# 4.  Save to .csv and .npy (faster)
df = pd.DataFrame(X, columns=cols)
df.to_csv("../data/params-list.suite2.csv", index=False, float_format="%.6e")
np.save("../data/params-list.suite2.npy", X)
print("Saved:", df.shape, "→ ../data/params_list.suite2.csv / .npy")

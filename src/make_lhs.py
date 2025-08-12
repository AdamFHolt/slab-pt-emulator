#!/usr/bin/env python3

import numpy as np
from pyDOE2 import lhs
import pandas as pd

# 1.  Parameter ranges 
RANGES = {
    # name         (min,   max,  'lin' | 'log')
    "v_conv"    : (  1.0 ,  10.0 , 'lin'),  # cm/yr  (convergence rate)
    "age_SP"    : ( 50.0 ,  110.0, 'lin'),  # Myr
    "age_OP"    : ( 5.0 ,  50.0 , 'lin'),  # Myr
    "dip_int"   : ( 20.0 ,  70.0 , 'lin'),  # degrees (slab dip angle)
    "eta_int"   : ( 1e19 ,  1e21 , 'log'),  # Pa·s  (interface viscosity)
    "eta_UM"    : ( 5e19 ,  1e21 , 'log'),  # Pa·s  (upper-mantle reference)
    "eps_trans" : ( 1e-15,  1e-13, 'log')   # s⁻¹   (transition strain-rate)
}

N_SAMPLES = 400
SEED      = 42

# rounding rules
ROUND_RULE = {
    "age_SP"  :  lambda x: np.round(x, 1),  
    "age_OP"  :  lambda x: np.round(x, 1),  
    "dip_int" :  lambda x: np.round(x, 1),  
    "v_conv"  :  lambda x: np.round(x, 4), 
    "eta_int" :  lambda x: np.vectorize(lambda y: '{:0.5e}'.format(y))(x),
    "eta_UM"  :  lambda x: np.vectorize(lambda y: '{:0.5e}'.format(y))(x),
    "eps_trans": lambda x: np.vectorize(lambda y: '{:0.5e}'.format(y))(x),
}

# 2.  Generate unit-cube Latin-hypercube 
unit = lhs(len(RANGES), samples=N_SAMPLES, criterion='maximin', random_state=SEED) # N×8 array of [0,1] values

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
df.to_csv("../data/params-list.csv", index=False, float_format="%.6e")
np.save("../data/params-list.npy", X)
print("Saved:", df.shape, "→ ../data/params_list.csv / .npy")

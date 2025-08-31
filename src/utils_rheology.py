#!/usr/bin/env python3
"""
Utility to compute upper- and lower-mantle creep prefactors
given the *reference* upper-mantle viscosity (eta_UM) and the
transition strain-rate (eps_trans).

Returns a dict:
{
    "Adisl": ...,
    "Adiff": ...,
    "Adiff_lm": ...
}
"""

import numpy as np

# ---- Physical constants & fixed params -----------------------
Edisl,  Vdisl  = 530e3, 18e-6      # J/mol, m³/mol
Ediff,  Vdiff  = 375e3, 4e-6
Vdiff_lm       = 4e-6
ndis, ndiff    = 3.5, 1.0
R              = 8.314
adiabat        = 0.3               # K/km
mid_jump       = 20                # viscosity jump at 660 km

def prefactors(eta_um: float, eps_trans: float) -> dict:
    """Return dislocation, diffusion, diffusion-LM prefactors."""

    # ---- Reference point in upper mantle ---------------------
    depth_ref  = 330e3                  # m
    T_ref      = 1673 + depth_ref*1e-3*adiabat
    P_ref      = 3300.0*9.81*depth_ref

    eta_individual = 2.0 * eta_um   # so that the effective viscosity 
                                    # is the same as the input eta_um   

    # Dislocation prefactor
    Adisl = ((eta_individual)**(-ndis)) / (
        0.5**(-ndis) * eps_trans**(ndis-1) *
        np.exp(-(Edisl + P_ref*Vdisl) / (R*T_ref))
    )

    # Diffusion prefactor (upper mantle)
    Adiff = ((eta_individual)**(-ndiff)) / (
        0.5**(-ndiff) * eps_trans**(ndiff-1) *
        np.exp(-(Ediff + P_ref*Vdiff) / (R*T_ref))
    )

    # ---- Lower mantle diffusion prefactor --------------------
    depth_ref = 660e3
    T_ref     = 1673 + depth_ref*1e-3*adiabat
    P_ref     = 3300.0*9.81*depth_ref
    eta_lm    = mid_jump * (0.5 * Adiff**(-1/ndiff) *
                            eps_trans**((1-ndiff)/ndiff) *
                            np.exp((Ediff + P_ref*Vdiff) / (ndiff*R*T_ref)))
    Adiff_lm  = ((eta_lm)**(-ndiff)) / (
        0.5**(-ndiff) * eps_trans**(ndiff-1) *
        np.exp(-(Ediff + P_ref*Vdiff_lm) / (R*T_ref))
    )

    return {"Adisl": Adisl, "Adiff": Adiff, "Adiff_lm": Adiff_lm}

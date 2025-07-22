#!/usr/bin/python3
import numpy as np

### conditions for ref. visc ###
# checking prefactors in run_400.prm
visc_ref   = 2.0*1.64217e+21 #5.e20
strain_ref = 3.1552e-15  #5e-15
################################
depth_ref  = 330e3 
temp_ref_0 = 1673
midmantle_viscosity_jump = 20

### rheology params
Edisl = 530.e3;  Vdisl = 18e-6;  
Ediff = 375.e3;  Vdiff = 4e-6; 
Vdiff_lowermant = 4e-6;
ndis = 3.5; ndiff = 1.       
R = 8.314

### additional physical params
adiabat = 0.3; # K/km 1694.5 + 330*0.3
temp_ref   = temp_ref_0 + (depth_ref * 1e-3 * adiabat)
press_ref  = 3300. * 9.81 * depth_ref

# compute upper mantle dislocation creep prefactor
# eta = 0.5 * A^(-1/n) * strain_rate^((1-n)/n) * exp((E+PV)/nRT)
# A   = ((eta)^(-n)) / (0.5^-n * strain_rate^(n-1) * exp(-(E+PV)/RT))
Adisl = ((visc_ref)**(-1. * ndis)) / (  0.5**(-1.*ndis) * strain_ref**(ndis-1) * np.exp(-(Edisl+press_ref*Vdisl)/(R*temp_ref))  )
visc_check_disl = 0.5 * Adisl**(-1/ndis) * strain_ref**((1-ndis)/ndis) * np.exp((Edisl+press_ref*Vdisl)/(ndis*R*temp_ref))
print("Dislocation prefactor = %e. (A check: %e = %e)" % (Adisl,visc_ref,visc_check_disl))

# compute upper mantle diffusion creep prefactor
Adiff = ((visc_ref)**(-1. * ndiff)) / (  0.5**(-1.*ndiff) * strain_ref**(ndiff-1) * np.exp(-(Ediff+press_ref*Vdiff)/(R*temp_ref))  )
visc_check_diff = 0.5 * Adiff**(-1/ndiff) * strain_ref**((1-ndiff)/ndiff) * np.exp((Ediff+press_ref*Vdiff)/(ndiff*R*temp_ref))
print("Diffusion prefactor   = %e. (A check: %e = %e)" % (Adiff,visc_ref,visc_check_diff))
print("Effective viscosity  = %e" % ((visc_check_disl * visc_check_diff)/(visc_check_disl + visc_check_diff)))

# compute lower mantle diffusion creep prefactor
depth_ref = 660.e3
temp_ref   = temp_ref_0 + (depth_ref * 1e-3 * adiabat)
press_ref  = 3300. * 9.81 * depth_ref
visc_660_diff = 0.5 * Adiff**(-1/ndiff) * strain_ref**((1-ndiff)/ndiff) * np.exp((Ediff+press_ref*Vdiff)/(ndiff*R*temp_ref))
visc_660_diff_lm = midmantle_viscosity_jump * visc_660_diff; # viscosity below discontinuity
Adiff_lm = ((visc_660_diff_lm)**(-1. * ndiff)) / (  0.5**(-1.*ndiff) * strain_ref**(ndiff-1) * np.exp(-(Ediff+press_ref*Vdiff_lowermant)/(R*temp_ref))  )
visc_check_diff_lm = 0.5 * Adiff_lm**(-1/ndiff) * strain_ref**((1-ndiff)/ndiff) * np.exp((Ediff+press_ref*Vdiff)/(ndiff*R*temp_ref))
print("Diffusion (lower mantle) prefactor   = %e." % (Adiff_lm))
print("(A check: Mid-mantle viscosity jump: %.0f)" % (visc_check_diff_lm/visc_660_diff))


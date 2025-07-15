#!/usr/bin/env python3 

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.special

dip_crust = float(sys.argv[1]) # [degrees]
# depth_full_crust = float(sys.argv[2]) # [m]
sp_age = float(sys.argv[2]) # [Ma]
op_age = float(sys.argv[3]) # [Ma]

ofile = f"outputs/temp_crust-dip{dip_crust}_sp-age{sp_age}_op-age{op_age}.txt"
oplot = f"outputs/temp_crust-dip{dip_crust}_sp-age{sp_age}_op-age{op_age}.png"

# box dimensions (i.e. "extent" in ASPECT input)
xmin=0;xmax=4000.e3 # [m]
ymin=0;ymax=1000.e3

# number of cells in input geometry
xnum= 2000
ynum= 500

# basic parameters
x_SP  = 2000.e3
y_crust = 10e3
radius_outer = 50e3 
depth_curved_notch = 50e3
Ma_to_sec = 3.15576e13 # [s/Ma]
k = 1e-6 # thermal diffusivity [m^2/s]
Tmax = 1600.  # [K]
Tmin = 273.15 # [K]

# empty array to store geometry
No_nodes= (xnum + 1) * (ynum + 1)
T=np.zeros([No_nodes,3],float)

# vector for y values (refined at top)
yvecta_length = ynum // 2
yvectb_length = ynum // 2
yvecta = np.linspace(ymin, ymax - 125e3, yvecta_length + 1)
yvectb = np.linspace(ymax - 125e3, ymax, yvectb_length + 1)
yvect = np.concatenate((yvecta, yvectb[1:]))

# vector for x values (refined around trench)
xvecta_length = xnum // 3
xvectb_length = 1 + (xnum // 3)
xvectc_length = 1 + (xnum // 3)
xvecta = np.linspace(xmin, x_SP - 100e3, xvecta_length + 1)
xvectb = np.linspace(x_SP - 100e3, x_SP + 400e3, xvectb_length + 1)
xvectc = np.linspace(x_SP + 400e3, xmax, xvectc_length + 1)
xvect = np.concatenate((xvecta, xvectb[1:], xvectc[1:]))

ind = 0
for j in range(ynum + 1): 
	for i in range(xnum + 1):

		x = xvect[i]
		y = yvect[j]
		
		T[ind,0] = x
		T[ind,1] = y
		T[ind,2] = Tmax

		erf_term_sp = (ymax - y) / (2 * np.sqrt(k * sp_age * Ma_to_sec))
		erf_term_op = (ymax - y) / (2 * np.sqrt(k * op_age * Ma_to_sec))

		# flat portion of SP
		if x <= (x_SP-radius_outer) and y > (ymax - 250e3):
			T[ind,2]='%.5f'  %   (Tmax - (Tmax - Tmin)*scipy.special.erfc(erf_term_sp))

		# angled portion of crust ("notch")
		elif x > (x_SP - radius_outer) and x < (x_SP + (300e3)/np.tan(np.radians(dip_crust))) and y > (ymax - 250e3):
			x1 = x_SP - radius_outer; 
			y1 = ymax - radius_outer; 

			# curved part
			if ((x-x1)**2 + (y-y1)**2) < radius_outer**2 and (y-y1) > 0:
				T[ind,2]='%.5f'  %   (Tmax - (Tmax - Tmin)*scipy.special.erfc(erf_term_sp))
			elif ((x-x1)**2 + (y-y1)**2) >= radius_outer**2 and (y-y1) > 0:
				T[ind,2]='%.5f'  %   (Tmax - (Tmax - Tmin)*scipy.special.erfc(erf_term_op))
			elif (x-x1) <= radius_outer and (y-y1) <= 0:
				T[ind,2]='%.5f'  %   (Tmax - (Tmax - Tmin)*scipy.special.erfc(erf_term_sp))

			# dipping portion
			top_y = y1 + radius_outer * np.cos(np.radians(dip_crust))
			top_x = x1 + radius_outer * np.sin(np.radians(dip_crust))
			if x >= top_x and y < (top_y - (x - top_x) * np.tan(np.radians(dip_crust))):
				T[ind,2]='%.5f'  %   (Tmax - (Tmax - Tmin)*scipy.special.erfc(erf_term_sp))
			elif x >= top_x and y >= (top_y - (x - top_x) * np.tan(np.radians(dip_crust))):
				T[ind,2]='%.5f'  %   (Tmax - (Tmax - Tmin)*scipy.special.erfc(erf_term_op))
		
		elif x >= (x_SP + (300e3)/np.tan(np.radians(dip_crust))) and y > (ymax - 250e3):
			T[ind,2]='%.5f'  %   (Tmax - (Tmax - Tmin)*scipy.special.erfc(erf_term_op))


		ind=ind+1; 

 
# write to file in ASPECT format
f= open(ofile,"w+")
f.write("# POINTS: %s %s\n" % (str(xnum+1),str(ynum+1)))
f.write("# Columns: x y temperature\n")
for k in range(0,ind):
	f.write("%.6f %.6f %.2f\n" % (T[k,0],T[k,1],T[k,2]))
f.close() 


# quick and dirty plot to visualize 
plt.figure()
levels = np.linspace(Tmin, 1700, 41)  
sz = plt.tricontourf(T[:,0]/1e3, T[:,1]/1e3, T[:,2], levels=levels,cmap='coolwarm') 
plt.colorbar(sz, label='T [K]')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(((xmin/1e3)+1700, (xmax/1e3)-1700))
plt.ylim((ymax/1e3)-300, ymax/1e3)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(oplot, dpi=300)

#plt.show()

#!/usr/bin/env python3 

import sys
import numpy as np
import matplotlib.pyplot as plt

dip_crust = float(sys.argv[1]) # [degrees]
depth_full_crust = float(sys.argv[2]) # [m]

ofile = f"outputs/comp_crust-dip{dip_crust}-depth{depth_full_crust}.txt"
oplot = f"outputs/comp_crust-dip{dip_crust}-depth{depth_full_crust}.png"

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

# empty array to store geometry
No_nodes= (xnum + 1) * (ynum + 1)
C=np.zeros([No_nodes,3],float)

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
		
		C[ind,0] = x
		C[ind,1] = y

		# crust along top of flat portion of SP
		if x <= (x_SP-radius_outer) and y > (ymax - y_crust):
			C[ind,2]=1

		# curved portion of crust ("notch")
		elif x > (x_SP - radius_outer) and x < (x_SP + 5*radius_outer):
			x1 = x_SP - radius_outer; 
			y1 = ymax - radius_outer; 
			if ((x-x1)**2 + (y-y1)**2) < radius_outer**2 and y > (ymax - depth_curved_notch):
				angle=np.arctan((y-y1)/(x-x1))
				if ((x-x1)**2 + (y-y1)**2) > (radius_outer-y_crust)**2:
					if angle > np.radians(90. - dip_crust):
						C[ind,2]=1

			# dipping portion
			bott_x = x1 + (radius_outer-y_crust) * np.sin(np.radians(dip_crust))
			bott_y = y1 + (radius_outer-y_crust) * np.cos(np.radians(dip_crust))
			top_y = y1 + radius_outer * np.cos(np.radians(dip_crust))
			if x >= bott_x and y > (bott_y - (x - bott_x) * np.tan(np.radians(dip_crust))) \
				and y < (bott_y - (x - bott_x) * np.tan(np.radians(dip_crust)) + (y_crust/np.cos(np.radians(dip_crust)))):
				y_to_tip = depth_full_crust - (ymax - bott_y)
				x_to_tip = y_to_tip / np.tan(np.radians(dip_crust))
				y_max_depth = ymax - depth_full_crust + (x - (bott_x + x_to_tip)) * np.tan(np.radians(90-dip_crust))
				if y > y_max_depth:
					C[ind,2]=1
			
			# trim off top of linear-curved transition
			if y >= top_y and ((x-x1)**2 + (y-y1)**2) >= radius_outer**2:
				C[ind,2]=0

		ind=ind+1; 

 
# write to file in ASPECT format
f= open(ofile,"w+")
f.write("# POINTS: %s %s\n" % (str(xnum+1),str(ynum+1)))
f.write("# Columns: x y composition1\n")
for k in range(0,ind):
	f.write("%.6f %.6f %.2f\n" % (C[k,0],C[k,1],C[k,2]))
f.close() 
 
# quick and dirty plot to visualize 
plt.figure()
sz = plt.tricontourf(C[:,0]/1e3, C[:,1]/1e3, C[:,2], levels=20)
plt.colorbar(sz, label='Comp')
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(((xmin/1e3)+1700, (xmax/1e3)-1700))
plt.ylim((ymax/1e3)-300, ymax/1e3)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(oplot, dpi=300)

#plt.show()

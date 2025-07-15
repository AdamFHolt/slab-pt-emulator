#weak zones extend under plates and have a tapering wedge toward the center of the domain
#Wedge taper has a flat top at base of plates.
#!/usr/bin/env python3 

import sys
import numpy

ofile="outputs/comp_ocean-crust-8km_younging_v.txt"

# box dimensions (i.e. "extent" in ASPECT input)
# units are meters
xmin=0;xmax=9000.e3;
ymin=0;ymax=1750.e3;

# number of cells in input geometry
xnum= 9000
ynum= 1750

# geometrical parameters, meters
x_gap = 1000.e3; #distance: model boundary to subduction plate
x_gap_2 = 1000.e3;# distance: overriding plate to model boundary

x_SP  = 5000.e3; #length of subducting plate
depth_notch  = 65e3; #initial depth of slabtop
radius_outer = 245e3; #radius of curvature for slab 
slab_dip = 70.; #dip of slab at . . . ?
y_crust = 8e3; #downgoing plate thickness
lith_core = 10e3; #thickness of non-brittle slab core
lith_core_depth = 20e3; #depth to top of non-brittle slab core
edge_ridge_extent = 500.e3; #half distance over which plate ages away from ridge at edge
int_ridge_extent = 0.e3; #half distance for plate aging within plate; sets gap of lith_core
ridge_loc = 2500.e3; #location of ridge in SP 
op_y_crust = 35e3; #thickness of non-brittle overriding plate
weak_zone = 300e3; #X and Y dimension of weak zones at outer edges of plates
# empty array to store geometry
No_nodes= (xnum + 1) * (ynum + 1)
C=numpy.zeros([No_nodes,6],float)
 
ind=0

for j in range(ynum + 1): 
	for i in range(xnum + 1):

		x = xmin + i * ((xmax - xmin)/xnum)
		y = ymin + j * ((ymax - ymin)/ynum) 
  
		C[ind,0] = x
		C[ind,1] = y

		# weak zone
		if x > (x_gap - weak_zone + 200e3) and x <= (x_gap) and y > (ymax - weak_zone):
			C[ind,4]=1
		# crust along top of flat portion of SP
		if x > (x_gap) and x <= (x_gap + x_SP - radius_outer) and y > (ymax - y_crust):
			C[ind,2]=1
		elif x > (x_gap) and x <= (x_gap + 200e3) and y < (ymax - y_crust - lith_core - lith_core_depth) and y > (ymax - weak_zone):
			C[ind,4]=1
		elif x > (x_gap + 200e3) and x <= (x_gap + 450e3) and y > (.928*x+3364e2) and y < (ymax - y_crust - lith_core - lith_core_depth):
			C[ind,4]=1
		#non-brittle core before internal ridge
		if x > (x_gap) and x <= (x_gap + ridge_loc - int_ridge_extent)\
		and y > (ymax - (lith_core + lith_core_depth)) and y < (ymax - lith_core_depth):
			C[ind,5]=1
		
		#non-brittle core after internal ridge
		elif x > (x_gap + ridge_loc + int_ridge_extent) and x <= (x_gap + x_SP - radius_outer)\
		and y > (ymax - (lith_core + lith_core_depth)) and y < (ymax - lith_core_depth):
			C[ind,5]=1
		
		# curved portion of crust ("notch")
		elif x > (x_gap + x_SP - radius_outer) and x < (x_gap + x_SP):
			x1 = x_gap + x_SP - radius_outer; 
			y1 = ymax - radius_outer;
			if ((x-x1)**2 + (y-y1)**2) < radius_outer**2 and y > (ymax - depth_notch): 
				angle=numpy.arctan((y-y1)/(x-x1));
				if ((x-x1)**2 + (y-y1)**2) > (radius_outer-y_crust)**2:
					if angle > numpy.radians(90. - slab_dip):
						C[ind,2]=1
				#non-brittle core along curved section
				elif ((x-x1)**2 + (y-y1)**2) > (radius_outer-(lith_core + lith_core_depth))**2\
				and ((x-x1)**2 + (y-y1)**2) < (radius_outer-lith_core_depth)**2:
					if angle > numpy.radians(90. - slab_dip):
						C[ind,5]=1

		# overriding plate above notch
		if x > (x_gap + x_SP - radius_outer) and x < (x_gap + x_SP):
			x1 = x_gap + x_SP - radius_outer; 
			y1 = ymax - radius_outer;
			if ((x-x1)**2 + (y-y1)**2) >= radius_outer**2 and y > (ymax - op_y_crust): 
				C[ind,3]= 1

		# flat portion of overriding plate
		if  x >= (x_gap + x_SP) and x < (xmax - x_gap_2) and y > (ymax - op_y_crust): 
			C[ind,3]= 1
		# weak zone
		elif x >= (xmax - x_gap_2 - 450e3) and x < (xmax - x_gap_2 - 200e3) and y > (-1*x + 9250e3) and y < (ymax - op_y_crust):
			C[ind, 4]= 1
		elif x >= (xmax - x_gap_2 - 200e3) and x < (xmax - x_gap_2) and y < (ymax - op_y_crust) and y > (ymax - weak_zone):
			C[ind, 4]= 1
		elif x >= (xmax - x_gap_2) and x < (xmax - x_gap_2 + weak_zone-200e3) and y > (ymax - weak_zone):
			C[ind, 4]= 1

		ind=ind+1;
 

# write to file in ASPECT format
f= open(ofile,"w+")
f.write("# POINTS: %s %s\n" % (str(xnum+1),str(ynum+1)))
f.write("# Columns: x y composition1 composition2 composition3 composition4\n")
#composition3 composition4
for k in range(0,ind):
	f.write("%.6f %.6f %.2f %.2f %.2f %.2f\n" % (C[k,0],C[k,1],C[k,2],C[k,3],C[k,4],C[k,5]))
	# %.2f %.2f
	#,C[k,4], C[k,5]
f.close() 


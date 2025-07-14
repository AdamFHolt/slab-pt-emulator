#!/usr/bin/env python3
"""
Generate temperature & composition input grids for ASPECT.
Returns the two file names so callers can embed them in the .prm.

Usage:
    from input_geometry import make_plate_inputs
    tfile, cfile = make_plate_inputs(dip=45, age_sp=100,
                                     age_op=50, plate_thick=125e3,
                                     out_dir=Path("model_runs/run_000"))
"""
import numpy as np, matplotlib
matplotlib.use("Agg")    # headless
import matplotlib.pyplot as plt
import scipy.special as sp
from pathlib import Path

def make_plate_inputs(*, dip, age_sp, age_op, plate_thick, out_dir):

    """Write temp & comp .txt (and a .png quick-look) into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    base = (f"crust-dip{dip}_sp-age{age_sp}_"
            f"op-age{age_op}_plate-thick{int(plate_thick):d}")

    tfile = out_dir / f"temp_{base}.txt"
    cfile = out_dir / f"comp_{base}.txt"
    pfile = out_dir / f"{base}.png"

    # box dimensions (i.e. "extent" in ASPECT input)
    xmin=0;xmax=4000.e3 # [m]
    ymin=0;ymax=1000.e3

    # number of cells in input geometry
    xnum= 2000
    ynum= 500

    # basic parameters
    depth_full_crust = plate_thick + 15e3
    x_SP  = 2000.e3				# [m]
    y_crust = 7.5e3				# [m]
    radius_outer = 50e3 		# [m]
    Ma_to_sec = 3.15576e13 		# [s/Ma]
    k = 1e-6 					# [m^2/s]
    Tmax = 1400.  				# [deg C]
    Tmin = 0 					# [deg C]
    stiff_thick = 100e3 			# [m] thickness of stiff ends of plates
    stiff_length = 500e3 		# [m] length of stiff ends of plates

    # empty array to store geometry
    No_nodes= (xnum + 1) * (ynum + 1)
    T=np.zeros([No_nodes,3],float)
    C=np.zeros([No_nodes,4],float)

    # vector for y values (refined at top)
    yvecta_length = ynum // 2
    yvectb_length = ynum // 2
    yvecta = np.linspace(ymin, ymax - (1.2*plate_thick), yvecta_length + 1)
    yvectb = np.linspace(ymax - (1.2*plate_thick), ymax, yvectb_length + 1)
    yvect = np.concatenate((yvecta, yvectb[1:]))

    # vector for x values (refined around trench)
    xvecta_length = xnum // 3
    xvectb_length = 1 + (xnum // 3)
    xvectc_length = 1 + (xnum // 3)
    xvecta = np.linspace(xmin, x_SP - 100e3, xvecta_length + 1)
    xvectb = np.linspace(x_SP - 100e3, x_SP + 500e3, xvectb_length + 1)
    xvectc = np.linspace(x_SP + 500e3, xmax, xvectc_length + 1)
    xvect = np.concatenate((xvecta, xvectb[1:], xvectc[1:]))

    ind = 0
    for j in range(ynum + 1): 
        for i in range(xnum + 1):

            x = xvect[i]
            y = yvect[j]
            
            T[ind,0] = x
            T[ind,1] = y
            T[ind,2] = Tmax

            C[ind,0] = x
            C[ind,1] = y
            C[ind,2] = 0
            C[ind,3] = 0

            ########## TEMPERATURE ########################################

            T_term1    = Tmax * ((ymax - y)/plate_thick)

            # flat portion of SP
            if x <= (x_SP-radius_outer) and y > (ymax - plate_thick):
                    T_term2 = 0
                    for n in range(1,5):
                        T_term2 += ((2*Tmax)/(n*np.pi))*np.sin(n*np.pi*(ymax-y)/plate_thick)*np.exp((-1.* n**2 * (np.pi)**2 * k * age_sp * Ma_to_sec)/(plate_thick**2))
                    T[ind,2]='%.5f'  %   (T_term1 + T_term2)


            # angled portion of crust ("notch")
            elif x > (x_SP - radius_outer) and x < (x_SP + (300e3)/np.tan(np.radians(dip))) and y > (ymax - plate_thick):
                x1 = x_SP - radius_outer; 
                y1 = ymax - radius_outer; 
                # curved part
                if ((x-x1)**2 + (y-y1)**2) < radius_outer**2 and (y-y1) > 0:
                    T_term2 = 0
                    for n in range(1,5):
                        T_term2 += ((2*Tmax)/(n*np.pi))*np.sin(n*np.pi*(ymax-y)/plate_thick)*np.exp((-1.* n**2 * (np.pi)**2 * k * age_sp * Ma_to_sec)/(plate_thick**2))
                    T[ind,2]='%.5f'  %   (T_term1 + T_term2)

                elif ((x-x1)**2 + (y-y1)**2) >= radius_outer**2 and (y-y1) > 0:

                    T_term2 = 0
                    for n in range(1,5):
                        T_term2 += ((2*Tmax)/(n*np.pi))*np.sin(n*np.pi*(ymax-y)/plate_thick)*np.exp((-1.* n**2 * (np.pi)**2 * k * age_op * Ma_to_sec)/(plate_thick**2))
                    T[ind,2]='%.5f'  %   (T_term1 + T_term2)

                elif (x-x1) <= radius_outer and (y-y1) <= 0:

                    T_term2 = 0
                    for n in range(1,5):
                        T_term2 += ((2*Tmax)/(n*np.pi))*np.sin(n*np.pi*(ymax-y)/plate_thick)*np.exp((-1.* n**2 * (np.pi)**2 * k * age_sp * Ma_to_sec)/(plate_thick**2))
                    T[ind,2]='%.5f'  %   (T_term1 + T_term2)


                # dipping portion
                top_y = y1 + radius_outer * np.cos(np.radians(dip))
                top_x = x1 + radius_outer * np.sin(np.radians(dip))
                if x >= top_x and y < (top_y - (x - top_x) * np.tan(np.radians(dip))):

                    T_term2 = 0
                    for n in range(1,5):
                        T_term2 += ((2*Tmax)/(n*np.pi))*np.sin(n*np.pi*(ymax-y)/plate_thick)*np.exp((-1.* n**2 * (np.pi)**2 * k * age_sp * Ma_to_sec)/(plate_thick**2))
                    T[ind,2]='%.5f'  %   (T_term1 + T_term2)

                elif x >= top_x and y >= (top_y - (x - top_x) * np.tan(np.radians(dip))):

                    T_term2 = 0
                    for n in range(1,5):
                        T_term2 += ((2*Tmax)/(n*np.pi))*np.sin(n*np.pi*(ymax-y)/plate_thick)*np.exp((-1.* n**2 * (np.pi)**2 * k * age_op * Ma_to_sec)/(plate_thick**2))
                    T[ind,2]='%.5f'  %   (T_term1 + T_term2)


            elif x >= (x_SP + (300e3)/np.tan(np.radians(dip))) and y > (ymax - plate_thick):

                T_term2 = 0
                for n in range(1,5):
                    T_term2 += ((2*Tmax)/(n*np.pi))*np.sin(n*np.pi*(ymax-y)/plate_thick)*np.exp((-1.* n**2 * (np.pi)**2 * k * age_op * Ma_to_sec)/(plate_thick**2))
                T[ind,2]='%.5f'  %   (T_term1 + T_term2)
            

            ########## COMPOSITION ########################################

            # crust along top of flat portion of SP
            if x <= (x_SP-radius_outer) and y > (ymax - y_crust):
                C[ind,2]=1

            # curved portion of crust ("notch")
            elif x > (x_SP - radius_outer) and x < (x_SP + (300e3)/np.tan(np.radians(dip))):
                x1 = x_SP - radius_outer; 
                y1 = ymax - radius_outer; 
                if ((x-x1)**2 + (y-y1)**2) < radius_outer**2 and y > (ymax - radius_outer):
                    angle=np.arctan((y-y1)/(x-x1))
                    if ((x-x1)**2 + (y-y1)**2) > (radius_outer-y_crust)**2:
                        if angle > np.radians(90. - dip):
                            C[ind,2]=1

                # dipping portion
                bott_x = x1 + (radius_outer-y_crust) * np.sin(np.radians(dip))
                bott_y = y1 + (radius_outer-y_crust) * np.cos(np.radians(dip))
                top_y = y1 + radius_outer * np.cos(np.radians(dip))
                if x >= bott_x and y > (bott_y - (x - bott_x) * np.tan(np.radians(dip))) \
                    and y < (bott_y - (x - bott_x) * np.tan(np.radians(dip)) + (y_crust/np.cos(np.radians(dip)))):
                    y_to_tip = depth_full_crust - (ymax - bott_y)
                    x_to_tip = y_to_tip / np.tan(np.radians(dip))
                    y_max_depth = ymax - depth_full_crust + (x - (bott_x + x_to_tip)) * np.tan(np.radians(90-dip))
                    if y > y_max_depth:
                        C[ind,2]=1
                
                # trim off top of linear-curved transition
                if y >= top_y and ((x-x1)**2 + (y-y1)**2) >= radius_outer**2:
                    C[ind,2]=0

            # stiff ends of plates
            if y >= (ymax - stiff_thick):
                if x < stiff_length or x > (xmax - stiff_length):
                    C[ind,3]=1

            #######################################################################

            ind=ind+1; 

    # write T and C to separate file in ASPECT format
    f= open(tfile,"w+")
    f.write("# POINTS: %s %s\n" % (str(xnum+1),str(ynum+1)))
    f.write("# Columns: x y temperature\n")
    for k in range(0,ind):
        f.write("%.6f %.6f %.3f\n" % (T[k,0],T[k,1],T[k,2]+273.))
    f.close() 
    
    f= open(cfile,"w+")
    f.write("# POINTS: %s %s\n" % (str(xnum+1),str(ynum+1)))
    f.write("# Columns: x y composition1 composition2\n")
    for k in range(0,ind):
        f.write("%.6f %.6f %.2f %.2f\n" % (C[k,0],C[k,1],C[k,2],C[k,3]))
    f.close()

    # PLOT ############################################################################### 
    plt.figure(figsize=(10, 8))

    # First subplot for temperature
    plt.subplot(3, 1, 1)
    levels = np.linspace(Tmin+273, 1700, 41)  
    sz = plt.tricontourf(T[:,0]/1e3, T[:,1]/1e3, T[:,2]+273., levels=levels, cmap='coolwarm') 
    plt.contour(C[:,0].reshape((ynum + 1, xnum + 1))/1e3, C[:,1].reshape((ynum + 1, xnum + 1))/1e3, \
        C[:,2].reshape((ynum + 1, xnum + 1)), levels=[0.5], colors='black', linewidths=1.5)
    plt.colorbar(sz, label='T [K]')
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    plt.xlim(((xmin/1e3)+1600, (xmax/1e3)-1500))
    plt.ylim((ymax/1e3)-250, ymax/1e3)
    plt.gca().set_aspect('equal', adjustable='box')

    # Second subplot for composition
    plt.subplot(3, 1, 2)
    sz2 = plt.tricontourf(C[:,0]/1e3, C[:,1]/1e3, C[:,2], levels=20)
    plt.colorbar(sz2, label='C')
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    plt.xlim(((xmin/1e3)+1600, (xmax/1e3)-1500))
    plt.ylim((ymax/1e3)-250, ymax/1e3)
    plt.gca().set_aspect('equal', adjustable='box')

    # Zoomed out plot
    plt.subplot(3, 1, 3)
    levels = np.linspace(Tmin+273, 1700, 41)  
    sz3 = plt.tricontourf(T[:,0]/1e3, T[:,1]/1e3, T[:,2]+273., levels=levels, cmap='coolwarm')
    plt.colorbar(sz3, label='T [K]')
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    plt.xlim(xmin,xmax/1e3)
    plt.ylim(ymin,ymax/1e3)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(pfile, dpi=100)

    print(f"[geom] wrote {tfile.name}  {cfile.name}")
    return tfile.name, cfile.name      # return *just file names* for template


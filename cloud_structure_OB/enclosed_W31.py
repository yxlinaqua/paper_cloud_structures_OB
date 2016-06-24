#enclosed mass as a function of radius
#accumulate as r
#phototulis 
#the center?? 


from astropy import units as u
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

import math

import pyparsing
import pyregion
import pyfits
import matplotlib.pyplot as plt
import matplotlib as mpl
import sep
import os
from astropy.table import Table
import numpy as np
import aplpy
import os
import powerlaw
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import wcs
from astropy import units as u
import astropy.constants as cons
import matplotlib.text as text
params = {'mathtext.default': 'regular' }   
def radial_all(data,annulus_width=1,center_x=None,center_y=None,working_mask=None,x=None,y=None,rmax=None):
    """
    r = radial_data(data,annulus_width,working_mask,x,y)
    
    A function to reduce an image to a radial cross-section.
    
    INPUT:
    ------
    data   - whatever data you are radially averaging.  Data is
            binned into a series of annuli of width 'annulus_width'
            pixels.
    annulus_width - width of each annulus.  Default is 1.
    working_mask - array of same size as 'data', with zeros at
                      whichever 'data' points you don't want included
                      in the radial data computations.
      x,y - coordinate system in which the data exists (used to set
             the center of the data).  By default, these are set to
             integer meshgrids
      rmax -- maximum radial value over which to compute statistics
    
     OUTPUT:
     -------
      r - a data structure containing the following
                   statistics, computed across each annulus:
          .r      - the radial coordinate used (outer edge of annulus)
          .mean   - mean of the data in the annulus
          .std    - standard deviation of the data in the annulus
          .median - median value in the annulus
          .max    - maximum value in the annulus
          .min    - minimum value in the annulus
          .numel  - number of elements in the annulus
    """
    
# 2010-03-10 19:22 IJC: Ported to python from Matlab
# 2005/12/19 Added 'working_region' option (IJC)
# 2005/12/15 Switched order of outputs (IJC)
# 2005/12/12 IJC: Removed decifact, changed name, wrote comments.
# 2005/11/04 by Ian Crossfield at the Jet Propulsion Laboratory
 
    import numpy as ny

    class radialDat:
        """Empty object container.
        """
        def __init__(self): 
            self.mean = None
            self.std = None
            self.median = None
            self.numel = None
            self.max = None
            self.min = None
            self.r = None
            #self.cumsum
            self.sum = None
    #---------------------
    # Set up input parameters
    #---------------------
    data = ny.array(data)
    
    if working_mask==None:
        working_mask = ny.ones(data.shape,bool)
    
    npix, npiy = data.shape
    #if center_x == None and center_y == None:
    #  if x==None or y==None:
    #    x1 = ny.arange(-npix/2.,npix/2.)
    #    y1 = ny.arange(-npiy/2.,npiy/2.)
    #    x,y = ny.meshgrid(y1,x1)
    #else:
    #    x1 = ny.arange(-center_x,center_x)
    #    y1 = ny.arange(-center_y,center_y)
    #    x,y = ny.meshgrid(y1,x1)
    #r = abs(x+1j*y)
    y, x = np.indices(data.shape)

    if not center_x:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center_x, y - center_y)

    if rmax==None:
        rmax = r[working_mask].max()

    #---------------------
    # Prepare the data container
    #---------------------
    dr = ny.abs([x[0,0] - x[0,1]]) * annulus_width
    radial = ny.arange(rmax/dr)*dr + dr/2.
    nrad = len(radial)
    radialdata = radialDat()
    radialdata.mean = ny.zeros(nrad)
    radialdata.std = ny.zeros(nrad)
    radialdata.median = ny.zeros(nrad)
    radialdata.numel = ny.zeros(nrad)
    radialdata.max = ny.zeros(nrad)
    radialdata.min = ny.zeros(nrad)
    radialdata.r = radial
    radialdata.sum = ny.zeros(nrad)
    
    
    #---------------------
    # Loop through the bins
    #---------------------
    import numpy as numpy
    for irad in range(nrad): #= 1:numel(radial)
      minrad = 0.#*irad
      maxrad = irad*dr + dr
      thisindex = (r>=minrad) * (r<maxrad) * working_mask
      if not thisindex.ravel().any():
        radialdata.mean[irad] = ny.nan
        radialdata.std[irad]  = ny.nan
        radialdata.median[irad] = ny.nan
        radialdata.numel[irad] = ny.nan
        radialdata.max[irad] = ny.nan
        radialdata.min[irad] = ny.nan
        radialdata.sum[irad] = ny.nan
      else:
        radialdata.mean[irad] = ny.nanmean(data[thisindex])
        radialdata.std[irad]  = ny.nanstd(data[thisindex])
        radialdata.median[irad] = ny.nanmedian(data[thisindex])
        radialdata.numel[irad] = data[thisindex].size
        radialdata.max[irad] = np.nanmax(data[thisindex])
        radialdata.min[irad] = np.nanmin(data[thisindex])
        radialdata.sum[irad] = np.nansum(data[thisindex])
    
    #---------------------
    # Return with data
    #---------------------
    
    return radialdata
    
import numpy as np

def enclosed_stastic(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof

####################################################################  
dirpath = r'/Users/yuxinlin/final_results_s/'

outpath = r'/Users/yuxinlin/texdoc/'
############################################################################

#filepath1= dirpath+r'W49N_merge_NH2.fits'
#filepath1 = r'/Users/yuxinlin/W49N_1/sed_cut2/merge_1arc/merge_1arc_2.fits'
#filepath2= dirpath+r'W43_main_merge_NH2_power_de21.fits'
#filepath3= dirpath+r'W43_south_merge_NH2_power_de21.fits'
filepath1= dirpath+r'G102_N_trim_cutoff.fits'
filepath2= dirpath+r'G103_N_trim_cutoff.fits'
filepath3= dirpath+r'W43_main_N_trim_cutoff.fits'
filepath4= dirpath+r'W43_south_N_trim_cutoff.fits'



##hdulist1 = fits.open(filepath1)
#hdulist1 = fits.open(filepath1)
#hdulist1 = fits.open(filepath1)
#hdulist1 = fits.open(filepath1)
#hdulist5 = fits.open(filepath5)
#hdulist6 = fits.open(filepath6)

#dirpath = r'/Users/yuxinlin/W43_2/sed_cut/W43_main/derive_250_350/fill_250_350/final_fitting/sm22/final_fitting_best_res/'

#filepath1= dirpath+r'NH2_7_35_45.sm10.ini.hc.asr.inter.fits'
#hdulist1_cor = fits.open(r'/Users/yuxinlin/W49N_1/sed_cut2/derive_450/derive_250_350/fill_250_350/W49N_merge_350/faint_final.regTAN.db.fits')
#hdulist1_cor = fits.open(r'/Users/yuxinlin/W43_2/sed_cut/W43_main/derive_250_350/fill_250_350/W43_main_merge_350/faint_final_2_reg_TAN.db.fits')
hdulist1 = fits.open(filepath1)
hdulist2 = fits.open(filepath2)
hdulist3 = fits.open(filepath3)
hdulist4 = fits.open(filepath4)
#hdulist1_cor = fits.open(r'/Users/yuxinlin/W43_2/sed_cut/W43_south/W43_south_merge_350/faint_final_reg_TAN.db.fits')
#hdulist1_cor = fits.open(r'/Users/yuxinlin/W31_3/sed_cut/W31_merge_350/G102.faint.reg.db.fits')
#hdulist1_cor = fits.open(r'/Users/yuxinlin/W31_3/sed_cut/W31_merge_350/G103.faint.modi.reg.db.fits')
############################################################################# 
#datarg=np.power(10,hdulist1[0].data)

#data1 = (hdulist1[0].data)*1e21
data1 = np.power(10,(hdulist1[0].data))

#data1[data1<6.58*10**20/7*2] = np.nan
#data2 = np.power(10,(hdulist2[0].data))
data2 = np.power(10,(hdulist2[0].data))
#data2[data2<6.58*10**20/7*2] = np.nan
data3 = np.power(10,hdulist3[0].data)
#data3
data4 = np.power(10,hdulist4[0].data)
#data3[data3<6.58*10**20/7*2] = np.nan

#y = data1[data1>0].flatten()

#data_m = ((y*2.8*cons.m_p.cgs/(1.9891e+33*u.g))*(u.cm**-2)).to(u.pc**-2)
##############################################################
distances =4.95*1000*u.pc
import astropy.units as u
import astropy.constants as cons
#distance = *1000.*u.parsec
center_x,center_y = 276,270

#for r in np.range(200):
#  for j in range(data1.shape[0]):
#      for i in range(data1.shape[1]):
#          sum_r = data1[(i-center_x)**2+(j-center_y)**2]

radial_sum_all = radial_all(data1,annulus_width=6,working_mask=None,center_x=center_x,center_y=center_y)
#pixel_size = 1.5 * u.arcsec


mean_NH2= radial_sum_all.mean*u.cm**(-2)
NH2_std = radial_sum_all.std*u.cm**(-2)
aperture =  np.pi*(radial_sum_all.r*1.5*distances/206265.)**2
aperture = aperture.cgs
mass = aperture*mean_NH2*2.8*cons.m_p.cgs/(1.9891e+33*u.g)
mass=radial_sum_all.sum*2.8*cons.m_p.cgs/(1.9891e+33*u.g)*(((1.5*distances/206265.).to(u.cm))**2)
mass_std = aperture*NH2_std*2.8*cons.m_p.cgs.value/(1.9891e+33*u.g)
radius = radial_sum_all.r*1.5*distances/206265#

center_x2, center_y2=272,245
radial_sum_all2 = radial_all(data2,annulus_width=6,working_mask=None,center_x=center_x2,center_y=center_y2)
#pixel_size = 1.5 * u.arcsec
distances2 = 3.22*1000*u.pc

mean_NH22= radial_sum_all2.mean*u.cm**(-2)
NH2_std2 = radial_sum_all2.std*u.cm**(-2)
aperture2 =  np.pi*(radial_sum_all2.r*1.5*distances2/206265.)**2
aperture2 = aperture2.cgs
mass2 = aperture2*mean_NH22*2.8*cons.m_p.cgs/(1.9891e+33*u.g)
mass2=radial_sum_all2.sum*2.8*cons.m_p.cgs/(1.9891e+33*u.g)*(((1.5*distances2/206265.).to(u.cm))**2)
mass_std2 = aperture2*NH2_std2*2.8*cons.m_p.cgs.value/(1.9891e+33*u.g)
radius2 = radial_sum_all2.r*1.5*distances2/206265#

center_x3, center_y3=235,273
center_x3, center_y3=214,211

radial_sum_all3 = radial_all(data3,annulus_width=6,working_mask=None,center_x=center_x3,center_y=center_y3)
#pixel_size = 1.5 * u.arcsec
distances3 = 5.5*1000*u.pc

mean_NH23= radial_sum_all3.mean*u.cm**(-2)
NH2_std3 = radial_sum_all3.std*u.cm**(-2)
aperture3 =  np.pi*(radial_sum_all3.r*1.5*distances3/206265.)**2
aperture3 = aperture3.cgs
mass3 = aperture3*mean_NH23*2.8*cons.m_p.cgs/(1.9891e+33*u.g)
mass3=radial_sum_all3.sum*2.8*cons.m_p.cgs/(1.9891e+33*u.g)*(((1.5*distances3/206265.).to(u.cm))**2)
mass_std3 = aperture3*NH2_std3*2.8*cons.m_p.cgs.value/(1.9891e+33*u.g)
radius3 = radial_sum_all3.r*1.5*distances3/206265#



center_x4, center_y4=222.9999,217.9999
radial_sum_all4 = radial_all(data4,annulus_width=6,working_mask=None,center_x=center_x4,center_y=center_y4)
#pixel_size = 1.5 * u.arcsec
distances4 = 5.5*1000*u.pc

mean_NH24= radial_sum_all4.mean*u.cm**(-2)
NH2_std4 = radial_sum_all4.std*u.cm**(-2)
aperture4 =  np.pi*(radial_sum_all4.r*1.5*distances4/206265.)**2
aperture4 = aperture4.cgs
mass4 = aperture4*mean_NH24*2.8*cons.m_p.cgs/(1.9891e+33*u.g)
mass4=radial_sum_all4.sum*2.8*cons.m_p.cgs/(1.9891e+33*u.g)*(((1.5*distances4/206265.).to(u.cm))**2)
mass_std4 = aperture4*NH2_std4*2.8*cons.m_p.cgs.value/(1.9891e+33*u.g)
radius4 = radial_sum_all4.r*1.5*distances4/206265#

#plt.loglog()
############draw virial r #######################
M_plot = np.linspace(100,5.e6,1000)

r = ((cons.G.cgs*M_plot*(1.9891e+33*u.g)*((1*u.Myr.cgs)**2))**(1./3)).to(u.pc)



############draw omega r#########################
r_omega = (2.*cons.G.cgs*M_plot*(1.9891e+33*u.g)*((10*u.km/u.s)**(-2))).to(u.pc)
fig = plt.figure()

ax = fig.add_subplot(111)
plt.axhline(y=3*1e4, linewidth=2, linestyle='--',color = 'k')
plt.text(60, 3*1e4+7000, '$m_{crit}$',size='large')
plt.text(10, 4*1e5, '$r_{vir}$',size='large')
plt.text(22, 1.8*1e5, '$r_{\Omega}$',size='large')
plt.title(r'Mass-Radius')
plt.xlabel('Radius [pc]')
plt.ylabel('Mass [$M_{\odot}$]')

plt.loglog()
plt.xlim((0.1,40))
plt.legend(loc='upper right', shadow=True,fontsize=10.)
import matplotlib.ticker as mtick
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e')) 
#plt.errorbar(radius.value,mass.value,yerr=mass_std.value,xerr=None,capsize=0.2)
plt.plot(r,M_plot,'k--')#,label='$r_{vir}$')
plt.plot(r_omega,M_plot,'k--')#,label='$r_{\Omega}$')
plt.plot(radius[1:-9],mass[1:-9],'ro-',markeredgecolor='none',markersize=3.6,label='G10.2-0.3')
#4.9
plt.axvline(x=4.9*60*distances.value/206265.,color='b',linestyle='--')
plt.plot(radius2[1:-8],mass2[1:-8],'co-',markeredgecolor='none',markersize=3.6,label='G10.3-0.1')
#5.0
plt.axvline(x=5.0*60*distances2.value/206265.,color='g',linestyle='--')

plt.plot(radius3[1:-3],mass3[1:-3],'bo-',markeredgecolor='none',markersize=3.6,label='W43-main')
#4.88
plt.axvline(x=4.88*60*distances3.value/206265.,color='r',linestyle='--')
plt.plot(radius4[1:-3],mass4[1:-3],'go-',markeredgecolor='none',markersize=3.6,label='W43-south')
#4.85
plt.axvline(x=4.85*60*distances4.value/206265.,color='c',linestyle='--')
plt.legend(loc='upper left',fontsize=15)
plt.show()
#plt.savefig('/Users/yuxinlin/texdoc/enmass_radius_dif.pdf')
#plt.savefig(outpath+'W31_mass_radius.eps')
#major_a=(major*3600*6.*(1000)/206265.)*u.pc.cgs
##*cons.parsec*100# in cgs unit
#
from scipy import interpolate
plt.clf()
radius = radius.value
mass = mass.value
radius2 = radius2.value
mass2 = mass2.value
radius3 = radius3.value
mass3 = mass3.value
radius4 = radius4.value
mass4 = mass4.value


tck = interpolate.splrep(np.log10(radius[1:-9]), np.log10(mass[1:-9]), k=2, s=0)


tck1 = interpolate.splrep(np.log10(radius2[1:-8]), np.log10(mass2[1:-8]), k=2, s=0)
tck2 = interpolate.splrep(np.log10(radius3[1:-3]), np.log10(mass3[1:-3]), k=2, s=0)
tck3 = interpolate.splrep(np.log10(radius4[1:-3]), np.log10(mass4[1:-3]), k=2, s=0)
tck4 = interpolate.splrep(np.log10(r.value), np.log10(M_plot), k=2, s=0)
tck5 = interpolate.splrep(np.log10(r_omega.value), np.log10(M_plot), k=2, s=0)


xnew = np.log10(np.linspace(0.14835285,11,1000))
xnew1 = np.log10(np.linspace(np.nanmin(radius2[1:-8]),np.nanmax(radius2[1:-8]),1000))
xnew2 = np.log10(np.linspace(np.nanmin(radius3[1:-3]),np.nanmax(radius3[:-3]),1000))
xnew3 = np.log10(np.linspace(np.nanmin(radius4[1:-3]),np.nanmax(radius4[1:-3]),1000))
figsize = plt.figure(1.2)
fig, axes = plt.subplots(2,figsize=(8,12))

axes[0].loglog()
axes[0].set_xlim((0.1,40))
axes[0].plot(r,M_plot,'k--')#,label='$r_{vir}$')
axes[0].plot(r_omega,M_plot,'k--')#,label='$r_{\Omega}$')
axes[0].axhline(y=3*1e4, linewidth=2, linestyle='--',color = 'k')
axes[0].text(20, 3*1e4+7000, '$m_{crit}$',size='large')
axes[0].text(10, 4*1e5, '$r_{vir}$',size='large')
axes[0].text(22, 1.8*1e5, '$r_{\Omega}$',size='large') 
axes[0].plot((radius[1:-9]), (mass[1:-9]),  'x-', label = 'G10.2-0.3')

axes[0].plot((radius2[1:-8]), (mass2[1:-8]),  'x-', label = 'G10.3-0.1')

axes[0].plot((radius3[1:-3]), (mass3[1:-3]),  'x-', label = 'W43-main')

axes[0].plot((radius4[1:-3]), (mass4[1:-3]),  'x-', label = 'W43-south')


axes[0].axvline(x=4.9*60*distances.value/206265.,color='b',linestyle='--')

#5.0
axes[0].axvline(x=5.0*60*distances2.value/206265.,color='g',linestyle='--')

#axes[0].plot(radius3[:-3],mass3[:-3],'bo-',markeredgecolor='none',markersize=3.6,label='W43-main')
#4.88
axes[0].axvline(x=4.88*60*distances3.value/206265.,color='r',linestyle='--')

#4.85
axes[0].axvline(x=4.85*60*distances4.value/206265.,color='c',linestyle='--')

axes[0].set_title(r'Mass-Radius')

axes[0].set_ylabel('Mass [$M_{\odot}$]')
axes[0].legend(loc = 'best',fontsize=10)



#axes[0].plot(xnew, interpolate.splev(xnew, tck, der=0), label = 'Fit')
plt.ylim((-0.5,3.5))
plt.axhline(y=3.,color='black',linestyle='--')
plt.axhline(y=1.,color='black',linestyle='--')
axes[1].set_xlim((0.1,40))
axes[1].text(22, 3.2, '$r_{vir}$',size='large')
axes[1].text(22, 1.2, '$r_{\Omega}$',size='large') 
axes[1].set_ylabel(r'$\mathrm{\frac{\partial(\log M(r))}{\partial(\log r)}}$',rotation=0.)
axes[1].set_xlabel('Radius [pc]')
axes[1].set_xscale('log')
axes[1].plot((radius[1:-9]), interpolate.splev(np.log10(radius[1:-9]), tck, der=1), label = '1st der G10.2')
axes[1].plot((radius2[1:-8]), interpolate.splev(np.log10(radius2[1:-8]), tck1, der=1), label = '1st der G10.3')
axes[1].plot((radius3[1:-3]), interpolate.splev(np.log10(radius3[1:-3]), tck2, der=1), label = '1st der W43-main')
axes[1].plot((radius4[1:-3]), interpolate.splev(np.log10(radius4[1:-3]), tck3, der=1), label = '1st der W43-south')
#for ax in axes:
   # ax.legend(loc = 'best',fontsize=10)
axes[1].axvline(x=4.9*60*distances.value/206265.,color='b',linestyle='--')

#5.0
axes[1].axvline(x=5.0*60*distances2.value/206265.,color='g',linestyle='--')

#axes[0].plot(radius3[:-3],mass3[:-3],'bo-',markeredgecolor='none',markersize=3.6,label='W43-main')
#4.88
axes[1].axvline(x=4.88*60*distances3.value/206265.,color='r',linestyle='--')

#4.85
axes[1].axvline(x=4.85*60*distances4.value/206265.,color='c',linestyle='--')
plt.savefig(outpath+'enmass_radius_dif_modi.pdf')

plt.show()
#minor_b=(minor*3600*6.*(1000)/206265.)*u.pc.cgs#*cons.parsec*100# in cgs unit
#
##calculate the mass inside the each clump, A pixel area of the ellipse: A = pi*a*b, total_NH2 column density inside each clump 
## u mean molecular weight 2.8, mH hydrogen weight mH = cons.m_p 
## since total_NH2 is in cm-2, so A should be in cm^2, (14.**4.95*(10***3)/206265.)# in pc unit*cons.parsec*100
##A = math.pow((14.**4.95*(10**3)/206265.)*cons.parsec*100.,2)*volume
#miu = 2.8
#A = major_a*minor_b*np.pi
#mH = cons.m_p.cgs
#mean_NH2 = mean_NH2*(u.cm**(-2))
#NH2=mean_NH2
#
#M = (mean_NH2*A*miu*mH/((1.9891e+33)*u.g)).decompose()
#
#center_x,center_y = 290,305
#calculate the segmented slopes for each source############
#calculate the total mass above 7*10e21 threshold

e

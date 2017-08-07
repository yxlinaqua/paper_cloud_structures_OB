# enclosed mass as a function of radius
# accumulate as r
# phototulis 
# the center
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from astropy import units as u
import astropy.constants as cons
from astropy.io import fits

params = {'mathtext.default': 'regular'}


def radial_data(data, annulus_width=1, working_mask=None, x=None, y=None, rmax=None):
    """
    A function to reduce an image to a radial cross-section.

    Parameters
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

    Returns
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

    # 2015/12/05 to nan forms by yxl
    # 2010-03-10 19:22 IJC: Ported to python from Matlab
    # 2005/12/19 Added 'working_region' option (IJC)
    # 2005/12/15 Switched order of outputs (IJC)
    # 2005/12/12 IJC: Removed decifact, changed name, wrote comments.
    # 2005/11/04 by Ian Crossfield at the Jet Propulsion Laboratory
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

    # ---------------------
    # Set up input parameters
    # ---------------------
    data = np.array(data)

    if working_mask == None:
        working_mask = np.ones(data.shape, bool)

    npix, npiy = data.shape
    if x == None or y == None:
        x1 = np.arange(-npix / 2., npix / 2.)
        y1 = np.arange(-npiy / 2., npiy / 2.)
        x, y = np.meshgrid(y1, x1)

    r = abs(x + 1j * y)

    if rmax == None:
        rmax = r[working_mask].max()

    # ---------------------
    # Prepare the data container
    # ---------------------
    dr = np.abs([x[0, 0] - x[0, 1]]) * annulus_width
    radial = np.arange(rmax / dr) * dr + dr / 2.
    nrad = len(radial)
    radialdata = radialDat()
    radialdata.mean = np.zeros(nrad)
    radialdata.std = np.zeros(nrad)
    radialdata.median = np.zeros(nrad)
    radialdata.numel = np.zeros(nrad)
    radialdata.max = np.zeros(nrad)
    radialdata.min = np.zeros(nrad)
    radialdata.sum = np.zeros(nrad)

    radialdata.r = radial

    # ---------------------
    # Loop through the bins
    # ---------------------
    for irad in range(nrad):  # = 1:numel(radial)
        minrad = 0. #irad * dr
        maxrad = minrad + dr
        thisindex = (r >= minrad) * (r < maxrad) * working_mask
        if not thisindex.ravel().any():
            radialdata.mean[irad] = np.nan
            radialdata.std[irad] = np.nan
            radialdata.median[irad] = np.nan
            radialdata.numel[irad] = np.nan
            radialdata.max[irad] = np.nan
            radialdata.min[irad] = np.nan
            radialdata.sum[irad] = np.nan
        else:
            radialdata.mean[irad] = np.nanmean(data[thisindex])
            radialdata.std[irad] = np.nanstd(data[thisindex])
            radialdata.median[irad] = np.nanmedian(data[thisindex])
            radialdata.numel[irad] = np.count_nonzero(~np.isnan(data[thisindex]))
            radialdata.max[irad] = np.nanmax(data[thisindex])
            radialdata.min[irad] = np.nanmin(data[thisindex])
            radialdata.sum[irad] = np.nansum(data[thisindex])

    return radialdata


dirpath = r'./'
outpath = r'./'
filepath1 = dirpath + r'G102_N_trim_cutoff.fits'
filepath2 = dirpath + r'G103_N_trim_cutoff.fits'
filepath3 = dirpath + r'W43_main_N_trim_cutoff.fits'
filepath4 = dirpath + r'W43_south_N_trim_cutoff.fits'

hdulist1 = fits.open(filepath1)
hdulist2 = fits.open(filepath2)
hdulist3 = fits.open(filepath3)
hdulist4 = fits.open(filepath4)
data1 = np.power(10, (hdulist1[0].data))
data2 = np.power(10, (hdulist2[0].data))
data3 = np.power(10, hdulist3[0].data)
data4 = np.power(10, hdulist4[0].data)
distances = 4.95 * 1000 * u.pc
center_x, center_y = 276, 270

radial_sum_all = radial_data(data1, annulus_width=6, working_mask=None, center_x=center_x, center_y=center_y)
mean_NH2 = radial_sum_all.mean * u.cm ** (-2)
NH2_std = radial_sum_all.std * u.cm ** (-2)
aperture = np.pi * (radial_sum_all.r * 1.5 * distances / 206265.) ** 2
aperture = aperture.cgs
mass = radial_sum_all.sum * 2.8 * cons.m_p.cgs / (1.9891e+33 * u.g) * (((1.5 * distances / 206265.).to(u.cm)) ** 2)
mass_std = aperture * NH2_std * 2.8 * cons.m_p.cgs.value / (1.9891e+33 * u.g)
radius = radial_sum_all.r * 1.5 * distances / 206265  #

center_x2, center_y2 = 272, 245
radial_sum_all2 = radial_data(data2, annulus_width=6, working_mask=None, center_x=center_x2, center_y=center_y2)
distances2 = 3.22 * 1000 * u.pc

mean_NH22 = radial_sum_all2.mean * u.cm ** (-2)
NH2_std2 = radial_sum_all2.std * u.cm ** (-2)
aperture2 = np.pi * (radial_sum_all2.r * 1.5 * distances2 / 206265.) ** 2
aperture2 = aperture2.cgs
mass2 = radial_sum_all2.sum * 2.8 * cons.m_p.cgs / (1.9891e+33 * u.g) * (((1.5 * distances2 / 206265.).to(u.cm)) ** 2)
mass_std2 = aperture2 * NH2_std2 * 2.8 * cons.m_p.cgs.value / (1.9891e+33 * u.g)
radius2 = radial_sum_all2.r * 1.5 * distances2 / 206265  #

center_x3, center_y3 = 235, 273
center_x3, center_y3 = 214, 211

radial_sum_all3 = radial_data(data3, annulus_width=6, working_mask=None, center_x=center_x3, center_y=center_y3)
distances3 = 5.5 * 1000 * u.pc

mean_NH23 = radial_sum_all3.mean * u.cm ** (-2)
NH2_std3 = radial_sum_all3.std * u.cm ** (-2)
aperture3 = np.pi * (radial_sum_all3.r * 1.5 * distances3 / 206265.) ** 2
aperture3 = aperture3.cgs
mass3 = radial_sum_all3.sum * 2.8 * cons.m_p.cgs / (1.9891e+33 * u.g) * (((1.5 * distances3 / 206265.).to(u.cm)) ** 2)
mass_std3 = aperture3 * NH2_std3 * 2.8 * cons.m_p.cgs.value / (1.9891e+33 * u.g)
radius3 = radial_sum_all3.r * 1.5 * distances3 / 206265  #

center_x4, center_y4 = 222.9999, 217.9999
radial_sum_all4 = radial_data(data4, annulus_width=6, working_mask=None, center_x=center_x4, center_y=center_y4)
distances4 = 5.5 * 1000 * u.pc

mean_NH24 = radial_sum_all4.mean * u.cm ** (-2)
NH2_std4 = radial_sum_all4.std * u.cm ** (-2)
aperture4 = np.pi * (radial_sum_all4.r * 1.5 * distances4 / 206265.) ** 2
aperture4 = aperture4.cgs
mass4 = radial_sum_all4.sum * 2.8 * cons.m_p.cgs / (1.9891e+33 * u.g) * (((1.5 * distances4 / 206265.).to(u.cm)) ** 2)
mass_std4 = aperture4 * NH2_std4 * 2.8 * cons.m_p.cgs.value / (1.9891e+33 * u.g)
radius4 = radial_sum_all4.r * 1.5 * distances4 / 206265  #

# plt.loglog()

#draw virial r
M_plot = np.linspace(100, 5.e6, 1000)
r = ((cons.G.cgs * M_plot * (1.9891e+33 * u.g) * ((1 * u.Myr.cgs) ** 2)) ** (1. / 3)).to(u.pc)

#draw Omega
r_omega = (2. * cons.G.cgs * M_plot * (1.9891e+33 * u.g) * ((10 * u.km / u.s) ** (-2))).to(u.pc)

fig = plt.figure()

ax = fig.add_subplot(111)
plt.axhline(y=3 * 1e4, linewidth=2, linestyle='--', color='k')
plt.text(60, 3 * 1e4 + 7000, '$m_{crit}$', size='large')
plt.text(10, 4 * 1e5, '$r_{vir}$', size='large')
plt.text(22, 1.8 * 1e5, '$r_{\Omega}$', size='large')
plt.title(r'Mass-Radius')
plt.xlabel('Radius [pc]')
plt.ylabel('Mass [$M_{\odot}$]')

plt.loglog()
plt.xlim((0.1, 40))
plt.legend(loc='upper right', shadow=True, fontsize=10.)

ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
# plt.errorbar(radius.value,mass.value,yerr=mass_std.value,xerr=None,capsize=0.2)
plt.plot(r, M_plot, 'k--')  # ,label='$r_{vir}$')
plt.plot(r_omega, M_plot, 'k--')  # ,label='$r_{\Omega}$')
plt.plot(radius[1:-9], mass[1:-9], 'ro-', markeredgecolor='none', markersize=3.6, label='G10.2-0.3')
# 4.9
plt.axvline(x=4.9 * 60 * distances.value / 206265., color='b', linestyle='--')
plt.plot(radius2[1:-8], mass2[1:-8], 'co-', markeredgecolor='none', markersize=3.6, label='G10.3-0.1')
# 5.0
plt.axvline(x=5.0 * 60 * distances2.value / 206265., color='g', linestyle='--')
plt.plot(radius3[1:-3], mass3[1:-3], 'bo-', markeredgecolor='none', markersize=3.6, label='W43-main')
# 4.88
plt.axvline(x=4.88 * 60 * distances3.value / 206265., color='r', linestyle='--')
plt.plot(radius4[1:-3], mass4[1:-3], 'go-', markeredgecolor='none', markersize=3.6, label='W43-south')
# 4.85
plt.axvline(x=4.85 * 60 * distances4.value / 206265., color='c', linestyle='--')
plt.legend(loc='upper left', fontsize=15)
plt.show()


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

xnew = np.log10(np.linspace(0.14835285, 11, 1000))
xnew1 = np.log10(np.linspace(np.nanmin(radius2[1:-8]), np.nanmax(radius2[1:-8]), 1000))
xnew2 = np.log10(np.linspace(np.nanmin(radius3[1:-3]), np.nanmax(radius3[:-3]), 1000))
xnew3 = np.log10(np.linspace(np.nanmin(radius4[1:-3]), np.nanmax(radius4[1:-3]), 1000))
figsize = plt.figure(1.2)
fig, axes = plt.subplots(2, figsize=(8, 12))

axes[0].loglog()
axes[0].set_xlim((0.1, 40))
axes[0].plot(r, M_plot, 'k--')  # ,label='$r_{vir}$')
axes[0].plot(r_omega, M_plot, 'k--')  # ,label='$r_{\Omega}$')
axes[0].axhline(y=3 * 1e4, linewidth=2, linestyle='--', color='k')
axes[0].text(20, 3 * 1e4 + 7000, '$m_{crit}$', size='large')
axes[0].text(10, 4 * 1e5, '$r_{vir}$', size='large')
axes[0].text(22, 1.8 * 1e5, '$r_{\Omega}$', size='large')
axes[0].plot((radius[1:-9]), (mass[1:-9]), 'x-', label='G10.2-0.3')

axes[0].plot((radius2[1:-8]), (mass2[1:-8]), 'x-', label='G10.3-0.1')

axes[0].plot((radius3[1:-3]), (mass3[1:-3]), 'x-', label='W43-main')

axes[0].plot((radius4[1:-3]), (mass4[1:-3]), 'x-', label='W43-south')

axes[0].axvline(x=4.9 * 60 * distances.value / 206265., color='b', linestyle='--')

# 5.0
axes[0].axvline(x=5.0 * 60 * distances2.value / 206265., color='g', linestyle='--')

# 4.88
axes[0].axvline(x=4.88 * 60 * distances3.value / 206265., color='r', linestyle='--')

# 4.85
axes[0].axvline(x=4.85 * 60 * distances4.value / 206265., color='c', linestyle='--')
axes[0].set_title(r'Mass-Radius')
axes[0].set_ylabel('Mass [$M_{\odot}$]')
axes[0].legend(loc='best', fontsize=10)

# axes[0].plot(xnew, interpolate.splev(xnew, tck, der=0), label = 'Fit')
plt.ylim((-0.5, 3.5))
plt.axhline(y=3., color='black', linestyle='--')
plt.axhline(y=1., color='black', linestyle='--')
axes[1].set_xlim((0.1, 40))
axes[1].text(22, 3.2, '$r_{vir}$', size='large')
axes[1].text(22, 1.2, '$r_{\Omega}$', size='large')
axes[1].set_ylabel(r'$\mathrm{\frac{\partial(\log M(r))}{\partial(\log r)}}$', rotation=0.)
axes[1].set_xlabel('Radius [pc]')
axes[1].set_xscale('log')
axes[1].plot((radius[1:-9]), interpolate.splev(np.log10(radius[1:-9]), tck, der=1), label='1st der G10.2')
axes[1].plot((radius2[1:-8]), interpolate.splev(np.log10(radius2[1:-8]), tck1, der=1), label='1st der G10.3')
axes[1].plot((radius3[1:-3]), interpolate.splev(np.log10(radius3[1:-3]), tck2, der=1), label='1st der W43-main')
axes[1].plot((radius4[1:-3]), interpolate.splev(np.log10(radius4[1:-3]), tck3, der=1), label='1st der W43-south')
# for ax in axes:
# ax.legend(loc = 'best',fontsize=10)
axes[1].axvline(x=4.9 * 60 * distances.value / 206265., color='b', linestyle='--')

# 5.0
axes[1].axvline(x=5.0 * 60 * distances2.value / 206265., color='g', linestyle='--')

# axes[0].plot(radius3[:-3],mass3[:-3],'bo-',markeredgecolor='none',markersize=3.6,label='W43-main')
# 4.88
axes[1].axvline(x=4.88 * 60 * distances3.value / 206265., color='r', linestyle='--')

# 4.85
axes[1].axvline(x=4.85 * 60 * distances4.value / 206265., color='c', linestyle='--')
plt.savefig(outpath + 'enmass_radius_dif_modi.pdf')

plt.show()

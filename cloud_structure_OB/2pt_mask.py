from astropy.io import fits
import math as math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


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
        minrad = irad * dr
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


####################################################################
dirpath = r'./'
outpath = r'./'
############################################################################
filepath1 = dirpath + r'G106_N_slp_cutoff.fits'
hdulist1 = fits.open(filepath1)
############################################################################# 
datarg = np.power(10, hdulist1[0].data)
data_fluc = datarg  # *1.0e21
mean = np.nanmean(datarg)  # *1.0e21

mask_array = np.ones_like(data_fluc)
for i in range(0, data_fluc.shape[0]):
    for j in range(0, data_fluc.shape[1]):
        if np.isnan(datarg)[i][j] or datarg[i][j] == 0:
            mask_array[i][j] = 0.

mean_ma = np.nanmean(mask_array)
mask_alias = mask_array

space_ac_test2 = signal.fftconvolve(data_fluc, data_fluc[::-1, ::-1], mode='full')  #
space_ac_test2 /= np.nanmax(space_ac_test2)  # normalization 1
space_ac_test1 = signal.fftconvolve(mask_array, mask_array[::-1, ::-1], mode='full')
space_ac_test1 = np.ceil(space_ac_test1)
#########three estimators see more in Kleiner\& Dickman 1985
# unbiased
norm_ac = space_ac_test2 / space_ac_test1 * np.count_nonzero(data_fluc)  # /space_ac_test3
norm_ac[np.where(space_ac_test1 == 0)] = np.nan

# biased
norm_ac_biased = space_ac_test2
norm_ac_biased[np.isinf(norm_ac_biased)] = np.nan

# median
norm_ac_me = space_ac_test2 / space_ac_test1 * np.count_nonzero(data_fluc) * np.sqrt(
    space_ac_test1 * np.count_nonzero(data_fluc))
norm_ac_me[space_ac_test1 == 0] = np.nan
norm_ac_me /= np.nanmax(norm_ac_me)
norm_ac_me[np.isinf(norm_ac_me)] = np.nan

distance = 4.95 * 1000.

limit = math.sqrt(data_fluc.shape[0] ** 2 + data_fluc.shape[1] ** 2)
radial_profile1 = radial_data(norm_ac, annulus_width=2., rmax=limit)
radial_profile2 = radial_data(norm_ac_biased, annulus_width=2., rmax=limit)
radial_profile3 = radial_data(norm_ac_me, annulus_width=2., rmax=limit)
pc_r1 = radial_profile1.r * 3.0 * distance / 206265  # *cons.parsec*100
pc_r2 = radial_profile2.r * 3.0 * distance / 206265  # *cons.parsec*100
pc_r3 = radial_profile3.r * 3.0 * distance / 206265  # *cons.parsec*100

radial_profile_number = radial_data(space_ac_test1, annulus_width=2., rmax=limit)

plt.clf()
plt.plot(np.hstack([pc_r1[radial_profile1.r < limit]]), np.hstack([radial_profile1.mean[radial_profile1.r < limit]]),
         label='unbiased')
plt.plot(np.hstack([pc_r2[radial_profile2.r < limit]]), np.hstack([radial_profile2.mean[radial_profile2.r < limit]]),
         label='biased')
plt.plot(np.hstack([pc_r3[radial_profile3.r < limit]]), np.hstack([radial_profile3.mean[radial_profile3.r < limit]]),
         label='median')
plt.title(r'G10.6-0.4 2pt')
plt.xlabel('r pc')
plt.legend(loc='upper right', shadow=True, fontsize=10.)
x0 = np.hstack([pc_r1[radial_profile1.r < limit]])
y0 = np.hstack([radial_profile1.mean[radial_profile1.r < limit]])
yerr = radial_profile1.std[radial_profile1.r < limit]
radial_profile_number.sum[np.logical_not(radial_profile_number.sum > 0)] = 1.
n_sum = np.sum(radial_profile_number.sum[:-1])
plt.errorbar(x0, y0, yerr=yerr, xerr=None, capsize=0.2)

plt.savefig(outpath + '3_estim_G106_tr.eps')
np.savetxt('./2pt/G106_2pt_dev_radial.txt',
           np.c_[pc_r1, radial_profile1.r, radial_profile1.mean, radial_profile1.std, radial_profile_number.sum],
           fmt='%.4f  %.4f %.4f %.4f %.4f ')

plt.show()

import math
import decimal
import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
from scipy.spatial.distance import pdist
from astropy.io import fits
from astropy import wcs


def random_dots(immask, n_dots=10, mindist=None, n_loop_max=1E9):
    """
    To generate random dots in a 2D masked image (with minimum separation)

    Parameters
    ----------
    #imdata: numpy.ndarray (float)
    #    image data
    imdata: numpy.ndarray (bool)
        image mask data.
    n_dots: int
        number of requested random dots
    mindist: None|tuple of float
        if None, no criteria on distance between dots
        if specified, the min distance between dots is set
    n_loop_max: int
        max loop can be used to generate random dots
    Returns
    -------
    x, y: X and Y index
    """
    assert np.ndim(immask) == 2
    assert n_dots > 0

    if mindist is None:
        # no criteria on min distance
        sz = immask.shape
        n_ele = sz[0] * sz[1]
        n_val = np.float(np.sum(immask))
        n_est = np.int64(10 * n_dots / (n_val / n_ele))

        x = np.random.randint(0, sz[1], n_est)
        y = np.random.randint(0, sz[0], n_est)
        ind_val = immask[y, x]
        sum_val = np.sum(ind_val)
        if sum_val > n_dots:
            # successful
            return x[ind_val][:n_dots], y[ind_val][:n_dots], True
        else:
            # unsucessful
            return random_dots(immask, n_dots=n_dots, mindist=mindist)
    else:
        # set min distance
        assert len(mindist) == 2
        mindist_y, mindist_x = mindist
        sz = immask.shape

        x, y = np.zeros((n_dots,), np.int64), np.zeros((n_dots,), np.int64)
        x[0], y[0] = _random_dot_valid(immask, sz)
        c = 1
        n_loop = 0
        print('@Cham: got random dots [%s/%s] (y, x) = (%d,%d),  ...' % (c, n_dots, y[0], x[0]))
        while c < n_dots and n_loop < n_loop_max:
            n_loop += 1
            x_, y_ = _random_dot_valid(immask, sz)
            dist = ((x_ - x[:c]) / mindist_x) ** 2. + ((y_ - y[:c]) / mindist_y) ** 2.
            if np.all(dist.flatten() > 1.):
                x[c], y[c] = x_, y_
                c += 1
                print('generate all random dots [%s/%s] (y, x) = (%d,%d),  ...' % (c, n_dots, y_, x_))
        if c < n_dots:
            print('generate all requested random dots successfully!')
            return x, y, False
        else:
            print('fail to generate all requested random dots!')
            return x, y, True


def _random_dot_valid(immask, sz=None):
    assert np.sum(immask) > 0
    if sz is None:
        sz = immask.shape
    x_, y_ = np.random.randint(0, sz[1]), np.random.randint(0, sz[0])
    while not immask[y_, x_]:
        x_, y_ = np.random.randint(0, sz[1]), np.random.randint(0, sz[0])
    return x_, y_


dirpath = r'./'
outpath = r'./'

filepath1 = dirpath + r'W43_main_N_erosion.fits'
filepath2 = dirpath + r'W43_main_Td_trim.fits'
filename = r'./nn_W43-main_erosion.txt'
distance = 5490. # source distance

hdulist1 = fits.open(filepath1)
hdulist2 = fits.open(filepath2)

hdulist1[0].data = np.power(10, hdulist1[0].data)
data1 = hdulist1[0].data
data2 = hdulist2[0].data

data2[np.where(np.logical(np.isnan(data1)))] = np.nan

nn_true = np.loadtxt(filename, unpack=True, usecols=[0])#true mean nn distance

# random dots in put
a_frac_min = 0.3  #approximately the fration of the valid_region/total_region)
n_repeat = 1000
n_clump = len(nn_true) #true number of elements

immask = np.logical_not(np.isnan(data1))

w = wcs.WCS(hdulist1[0].header, hdulist1)
ra_mc = []
dec_mc = []
count = 0
i = 0

mean_8_all = []
dist_all = []
while count < 1000:
    #1000 times of random dots generation realization
    ra_mc = []
    dec_mc = []
    dist = []
    x_use, y_use, flag = random_dots(immask, n_dots=n_clump, mindist=(0., 0.), n_loop_max=1e9)

    for i in np.arange(len(nn_true)):
        index = i
        ra_mc = np.append(ra_mc, w.all_pix2world(x_use[index], y_use[index], 1)[0])
        dec_mc = np.append(dec_mc, w.all_pix2world(x_use[index], y_use[index], 1)[1])
    dist = pdist(np.column_stack(np.array([ra_mc, dec_mc])))
    tree = scipy.spatial.KDTree(zip(ra_mc, dec_mc))
    dist_ = dist * 3600 * distance / 206265
    dist_all = np.append(dist_all, dist_)
    nja = []
    for i in range(0, len(tree.data)):
        pts = np.array([tree.data[i][0], tree.data[i][1]])
        nja.append(pts)
    nearest = []
    sep = []
    nn_id = []
    for i in range(0, len(nja)):
        nearest = tree.query(nja[i], k=len(nn_true))
        sep = np.append(sep, nearest[0])
        nn_id = np.append(nn_id, nearest[1])
    sep = sep.reshape(len(nn_true), len(nn_true)).T
    distance_sep = sep * 3600 * distance / 206265
    N = np.array(len(nn_true))
    mean_8 = np.mean(distance_sep, axis=1)
    mean_8_all = np.append(mean_8_all, mean_8)
    count += 1
t = mean_8_all.reshape(count, len(nn_true))
f = r'./mc_W43_south_result.txt'
np.savetxt(f, np.transpose(t), fmt='%.8f', newline='\n')
d = dist_all.reshape(count, (len(nn_true) ** 2 - len(nn_true)) / 2)
f1 = r'./mc_W43_south_sepall_result.txt'
np.savetxt(f1, np.transpose(d), fmt='%.8f', newline='\n')

plt.clf()
for i in np.arange((count)):
    plt.plot(nn_true / (t[i, :]))
plt.clf()
plt.plot(nn_true / (np.nanmean(t, axis=0)), label='MC simulation')
# np.nanstd(t,axis=0)

NH2 = data1
Td = data2
A = (np.array(Td.flatten().shape) - np.array(Td[np.isnan(Td)].shape)) * (1.5 * distance / 206265) ** 2

random_k = []
for i in range(len(nn_true)):
    random_k = np.append(random_k, float(
        i * math.factorial(2 * i) / decimal.Decimal(2.0 ** i * math.factorial(i)) / decimal.Decimal(
            2.0 ** i * math.factorial(i))) / np.sqrt(len(nn_true) / A))
nn_index = nn_true / random_k
plt.plot(nn_index, label='empirical')
plt.legend(shadow=False)
plt.show()

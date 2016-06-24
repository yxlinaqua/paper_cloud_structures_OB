from astropy.io import fits
from astropy import units as u

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy import wcs  
from astropy.coordinates import SkyCoord,Angle
import scipy as scipy
from scipy.spatial.distance import pdist
####################################################################  
dirpath = r'./'

outpath = r'./'
############################################################################

filepath1= dirpath+r'W43_main_N_erosion.fits'
filepath2 = dirpath+r'W43_main_Td_trim.fits'
filename = r'./nn_W43-main_erosion.txt'

hdulist1 =fits.open(filepath1)

hdulist2 = fits.open(filepath2)

hdulist1[0].data=np.power(10,hdulist1[0].data)
data = hdulist1[0].data
for i in range(0,hdulist1[0].data.shape[0]):
  for j in range(0,hdulist1[0].data.shape[1]):
       if np.isnan((hdulist1[0].data)[i][j]):
          (hdulist2[0].data)[i][j]=np.nan


nn_true = np.loadtxt(filename,unpack=True,usecols=[0])

#random distribution

a_frac_min = 0.3
n_repeat = 1000
n_clump = len(nn_true)

len_rand = n_clump / a_frac_min * n_repeat

# random int
randx = np.random.randint(0,data.shape[0], size=(len_rand))
randy = np.random.randint(0,data.shape[1], size=(len_rand))


ind_rand_valid = np.logical_not(np.isnan(data[randx, randy]))


x_use = randx[ind_rand_valid]
y_use = randy[ind_rand_valid]
x_use, y_use

w = wcs.WCS(hdulist1[0].header, hdulist1)
ra_mc = []
dec_mc = []
count = 0
i = 0
distance = 5490

mean_8_all = []
dist_all = []
while count < 1000:
 ra_mc=[]
 dec_mc=[]
 dist = []
 
 for i in np.arange(len(nn_true)):
    index = count*len(nn_true)+i
    ra_mc = np.append(ra_mc,w.all_pix2world(x_use[index],y_use[index] , 1)[0])
    dec_mc = np.append(dec_mc,w.all_pix2world(x_use[index],y_use[index] , 1)[1])
    dist=pdist(np.column_stack(np.array([ra_mc, dec_mc])))
 tree =scipy.spatial.KDTree(zip(ra_mc, dec_mc)) 
 dist = dist[dist < 50]

 dist_ = dist*3600*distance/206265
 dist_all = np.append(dist_all,dist_)
#s, n,f =plt.hist(dist_,bins=20,normed=False)
#p = 2*s/(len(nn_true)**2-1)/(n[1]-n[0])
 nja = []
 for i in range(0,len(tree.data)):
      pts = np.array([tree.data[i][0],tree.data[i][1]])
      nja.append(pts)
 nearest = []
 sep = []
 nn_id = []
 for i in range(0, len(nja)):
   # if not (np.array_equal(nja,tree.data)):
     nearest = tree.query(nja[i],k=len(nn_true))
     sep =np.append(sep,nearest[0])
     nn_id = np.append(nn_id,nearest[1])
     #print nearest
 sep = sep.reshape(len(nn_true),len(nn_true)).T
 distance_sep = sep*3600*distance/206265
 N = np.array(len(nn_true))
 mean_8 = np.mean(distance_sep,axis=1)
 mean_8_all = np.append(mean_8_all,mean_8)
# np.savetxt(f,np.c_[mean_8],fmt='%.8f',newline='\n')
 count +=1
t=mean_8_all.reshape(count,len(nn_true))
f = r'./mc_W43_south_result.txt'
np.savetxt(f,np.transpose(t),fmt='%.8f',newline='\n')
d =dist_all.reshape(count,(len(nn_true)**2-len(nn_true))/2)
f1 = r'./mc_W43_south_sepall_result.txt'
np.savetxt(f1,np.transpose(d),fmt='%.8f',newline='\n')
nn_true
plt.clf()
for i in np.arange((count)):
    plt.plot(nn_true/ (t[i,:]))
plt.clf()
plt.plot(nn_true/(np.nanmean(t,axis=0)),label='MC simulation')
#np.nanstd(t,axis=0)

NH2 = data
Td = hdulist2[0].data
A = (np.array(Td.flatten().shape)-np.array(Td[np.isnan(Td)].shape))*(1.5*distance/206265)**2
#mean_e = 0.5/np.sqrt(N/A) 
import math
random_k = []
import decimal
for i in range(len(nn_true)):
    random_k = np.append(random_k,float(i*math.factorial(2*i)/decimal.Decimal(2.0**i*math.factorial(i))/decimal.Decimal(2.0**i*math.factorial(i)))/np.sqrt(len(nn_true)/A))
nn_index = nn_true/random_k
plt.plot(nn_index,label='empirical')
plt.legend(shadow=False)
plt.show()

################################################################

fig = plt.figure()
plt.imshow(data,)
plt.plot(randy[ind_rand_valid][0:50],randx[ind_rand_valid][0:50],'r.')
plt.plot(randy[ind_rand_valid][50:100],randx[ind_rand_valid][50:100],'g.')
plt.plot(randy[ind_rand_valid][100:150],randx[ind_rand_valid][100:150],'y.')
plt.plot(randy[ind_rand_valid][150:200],randx[ind_rand_valid][150:200],'mx')
plt.show()



#######calculate the perimeter of the region#################
NH2 = data
peri = np.zeros_like(NH2)
for i in range(NH2.shape[0]):
    for j in range(NH2.shape[1]):
        if not np.isnan(NH2[i][j]):
            peri[i][j] = 1.
 
perimeter = np.sum(peri[:,1:] != peri[:,:-1]) + np.sum(peri[1:,:] != peri[:-1,:]) 
perimeter = perimeter*1.5*distance/206265
#outpath = r'./sep_distance_W43-main_erosion.txt'
#np.savetxt(outpath,dist_.flatten(), newline='\n')  
#n_interior = abs(np.diff(peri, axis=0)).sum() + abs(np.diff(peri, axis=1)).sum()
#n_boundary = peri[0,:].sum() + peri[:,0].sum() + peri[-1,:].sum() + peri[:,-1].sum()
#perimeter = n_interior + n_boundary
################################################################


#outpath = r'./knn_W43-main_erosion.txt'
#np.savetxt(outpath,nn_index.flatten(), newline='\n')  
#outpath1 = r'./nn_W43-main_erosion.txt'
#np.savetxt(outpath1,mean_8.flatten(), newline='\n')  
#SE = 0.26135/np.sqrt(N**2/A)



#zANN = (mean_8[1]-mean_e)/SE

######################modified NNA###############################

#mean_e_modi = 0.5/np.sqrt(N/A)+(0.0514+0.041/np.sqrt(N))*perimeter/N
#
#Var = 0.070*A/N**2+0.037*perimeter*np.sqrt(A/N**5)
#
#Z_stat = (mean_8[1]-mean_e_modi)/np.sqrt(Var)
#
#print Z_stat

#plt.clf()

















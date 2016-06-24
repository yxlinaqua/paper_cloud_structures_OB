from skimage.morphology import remove_small_objects,closing
from skimage.morphology import disk,erosion,opening
import numpy as np
import os
from astropy.io import fits
from astrodendro import Dendrogram, pp_catalog,ppv_catalog
from astrodendro.analysis import PPStatistic
#from astrodendro.scatter import Scatter
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import astrodendro.pruning as pr
from scipy import linspace, polyval, polyfit, sqrt, stats, randn
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def dendroplot(axis=ax3, axname1='area_exact', axname2='flux',
                   axscale1=1.,
                   axscale2=1.,
                   leaves_list=[d.leaves],
                   # r, b, g
                   color_list=['#CC4444', '#4444CC', '#44CC44'],
                   highlight_monotonic=True,
                   marker='s',
                   marker2=None,
                   linestyle='-', **kwargs):
        for leaves, color in zip(leaves_list,color_list):
            for i,leaf in enumerate(leaves):
              xax,yax = ([cat[leaf.idx][axname1]*axscale1],
                           [cat[leaf.idx][axname2]*axscale2])
                #if axname1 in ('v_rms','reff'):
                #    xax *= gcorfactor[leaf.idx]
                #if axname2 in ('v_rms','reff'):
                #    yax *= gcorfactor[leaf.idx]
              if not (NH2_mean[i] < 10*8.0e20/2.35*5. ):

               axis.plot(xax, yax, marker, color=color, markeredgecolor='none', alpha=0.5)
               obj = leaf.parent
               if obj is not None:
                if obj.parent is not None: 
                 while obj.parent:
                    xax.append(cat[obj.idx][axname1]*axscale1)
                    yax.append(cat[obj.idx][axname2]*axscale2)
                    obj = obj.parent
                 if np.any(np.isnan(yax)):
                    ok = ~np.isnan(yax)
                    axis.plot(np.array(xax)[ok], np.array(yax)[ok], alpha=0.5,
                #label=leaf.idx, 
                              color='b', zorder=5,
                              linestyle=linestyle, marker=marker2, **kwargs)
                 else:
                    axis.plot(xax, yax, alpha=0.1, 
                    #label=leaf.idx, 
                    color=color,
                              zorder=5, linestyle=linestyle, marker=marker2,
                              **kwargs)
                 if highlight_monotonic:
                    signs = np.sign(np.diff(yax))
                    if np.all(signs==1) or np.all(signs==-1):
                        axis.plot(xax, yax, alpha=0.1, linewidth=5, zorder=0, color='g')
                #else: 
                #  while obj.ancestor:
                #    xax.append(cat[obj.idx][axname1]*axscale1)
                #    yax.append(cat[obj.idx][axname2]*axscale2)
                #    obj = obj.parent
                #  if np.any(np.isnan(yax)):
                #    ok = ~np.isnan(yax)
                #    axis.plot(np.array(xax)[ok], np.array(yax)[ok], alpha=0.5,
                ##label=leaf.idx, 
                #              color='b', zorder=5,
                #              linestyle=linestyle, marker=marker2, **kwargs)
                #  else:
                #    axis.plot(xax, yax, alpha=0.1, 
                #    #label=leaf.idx, 
                #    color=color,
                #              zorder=5, linestyle=linestyle, marker=marker2,
                #              **kwargs)
                #  if highlight_monotonic:
                #    signs = np.sign(np.diff(yax))
                #    if np.all(signs==1) or np.all(signs==-1):
                #        axis.plot(xax, yax, alpha=0.1, linewidth=5, zorder=0, color='g')
####################################################################  
dirpath = r'./'

outpath = r'./'
############################################################################

filepath1= dirpath+r'W43_main_N_trim.fits'

hdulist1 =fits.open(filepath1)

hdulist1[0].data=np.power(10,hdulist1[0].data)
data = hdulist1[0].data
##############write fits file##################
data=fits.getdata('./W43_main_N_trim.fits')

mask=np.isfinite(data)


mask2=erosion(mask, disk(20))
plt.ion()
plt.imshow(data/mask2, origin='lower',interpolation='nearest')
f = data/mask2
f = np.power(10,f)
NH2 = f
#plt.show()
outpath1=r'./W43_main_N_erosion.fits'
if os.path.exists(outpath1):
    os.remove(outpath1)
if os.path.exists(outpath1):
    os.remove(outpath1)  
print 'Writing',outpath1
fits.writeto(outpath1,np.log10(f),header=hdulist1[0].header)

params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)
d = Dendrogram.compute(f,min_value=7.0e21,min_delta=8e20/2.35*5.,min_npix=7.0)

########settings#################################
metadata = {}
metadata['data_unit'] = u.Jy
metadata['spatial_scale'] =  1.5 * u.arcsec
metadata['beam_major'] =  10.0 * u.arcsec
metadata['beam_minor'] =  10.0 * u.arcsec
##############calculate dendrogram##################################

d.save_to('W43_main_my_dendrogram_erosion.fits')
NH2_mean =[]
for i,leaf in enumerate(d.leaves):
    NH2_mean= np.append(NH2_mean,np.nanmean(NH2[leaf.indices()[0],leaf.indices()[1]]))
    
fig = plt.figure(figsize=(17,7))
p3 = d.plotter()
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)

ax1.imshow(np.log10(f), origin='lower', interpolation='nearest',
          cmap=plt.cm.Blues, vmin=21.7,vmax=23)

count = 0
p3.plot_tree(ax2, color='black',lw=1.0)
for i,leaf in enumerate(d.leaves):
        if not (NH2_mean[i] < (10.*8e20/2.35*5.) ):#and eff_radius[i]**2 > 0.15):# and not(NH2_mean[i] < 2.1e22 and eff_radius[i]**2 <0.08):
           p3.plot_contour(ax1, structure=leaf, lw=3, colors='red')
           p3.plot_tree(ax2, color='red',structure=leaf)
           count = count+1
        else:
           p3.plot_contour(ax1, structure=leaf, lw=3, colors='orange')
           p3.plot_tree(ax2, color='orange',structure=leaf)
ax2.hlines(10*8e20/2.35*5., *ax2.get_xlim(), color='b', linestyle=':')

mask_ = fits.PrimaryHDU(~mask.astype('short'), hdulist1[0].header)


axins = zoomed_inset_axes(ax2, 3.5, loc=2) # zoom = 6
p3.plot_tree(axins, color='black')
for i,leaf in enumerate(d.leaves):
        if not (NH2_mean[i] < 10*8e20/2.35*5. ):
           p3.plot_tree(axins, color='red',structure=leaf)
        else:
           p3.plot_tree(axins, color='orange',structure=leaf)
#axins.imshow(Z2, extent=extent, interpolation="nearest",
             #origin="lower")

# sub region of the original image
x1,x2,y1, y2 = 0,30,1e22, 0.5e23
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.hlines(10*8e20/2.35*5., *ax2.get_xlim(), color='b', linestyle=':')
#ax2.set_xticks([])
axins.set_yticks([])
#ax2.set_xtick_labels([])
plt.xticks(visible=False)
plt.yticks(visible=False)

mark_inset(ax2, axins, loc1=2, loc2=4, fc="none", ec="0.5")
cat = pp_catalog(d, metadata)
###########leaf and brances parameters###########################################
from astropy import constants as con
distance = 5500*u.pc
eff_radius = np.sqrt(cat['area_exact']/np.pi)*distance/206265.*u.pc
mass = cat['flux']*2.8*con.m_p.cgs/(1.9891e+33*u.g)*(((1.5*distance/206265.).to(u.cm))**2)

mass_corrected_leaf = []
mass_leaf = []
eff_radius_leaf = []
#########calculate leaf parameters###################################################
for i,leaf in enumerate(d.leaves):
   if leaf.parent is not None:
      merging_level = np.sum(leaf.parent.values(subtree=False))/leaf.parent.get_npix(subtree=False)
   else:
      merging_level = np.sum(leaf.ancestor.values(subtree=False))/leaf.ancestor.get_npix(subtree=False)
   fluxes_corrected = cat[leaf.idx]['flux']-merging_level*leaf.get_npix()
   mass_corrected_leaf = np.append(mass_corrected_leaf,fluxes_corrected*2.8*con.m_p.cgs/(1.9891e+33*u.g)*(((1.5*distance/206265.).to(u.cm))**2))
   mass_leaf= np.append(mass_leaf,cat[leaf.idx]['flux']*2.8*con.m_p.cgs/(1.9891e+33*u.g)*(((1.5*distance/206265.).to(u.cm))**2))
   eff_radius_leaf = np.append(eff_radius_leaf,np.sqrt(cat[leaf.idx]['area_exact']/np.pi)*distance/206265.*u.pc)


cat['area_exact']=eff_radius
cat['flux'] = mass
ax3 = fig.add_subplot(1, 3, 3)
ax3.loglog()
ax3.set_ylabel('Mass [$M_{\odot}$]')
ax3.set_xlabel('Effective Radius [pc]')
ax3.set_xlim([0.01*3,15])
ax3.set_ylim([0.1,10**6])
ax3.set_title('W43-main')


################draw m-r for all structures, modules from Adam#################################
r = np.linspace(0.001,15,1000)

m_thres = 870*r**1.33
ax3.set_title('W43-main')
ax3.fill_between(r,0.001, m_thres, color='0.8', alpha=0.5)


ax3.plot(eff_radius_leaf[NH2_mean > 10*8e20/2.35*5. ], mass_leaf[NH2_mean > 10*8e20/2.35*5. ],'s',color='red',label='mass',markeredgecolor='none',alpha=0.5) 

ax3.plot(eff_radius_leaf[NH2_mean > 10*8e20/2.35*5. ], mass_corrected_leaf[NH2_mean > 10*8e20/2.35*5. ],'s',color='purple',label='corrected mass',markeredgecolor='none',alpha=0.5) 
ax3.legend(loc='lower right', fontsize=12.)                       
dendroplot()

##############reference lines for m-r relations######################
r = np.linspace(0.001,15,1000)*u.pc
m_loci = 4.0/3.0*np.pi*r**3.0*2.8*con.m_p.cgs*1e3/(1.9891e+33*u.g)*((u.pc).to(u.cm))**3
m_loci2 = 4.0/3.0*np.pi*r**3.0*2.8*con.m_p.cgs*1e5/(1.9891e+33*u.g)*((u.pc).to(u.cm))**3
ax3.plot(r,m_loci,'b:')
ax3.plot(r,m_loci2,'b:')

m_larson = 1e22*np.pi*(r.value*u.pc.to(u.cm))**2*2.8*con.m_p.cgs/(1.9891e+33*u.g)

ax3.plot(r,m_larson,'g--')

###################fit m-r relations###########################
##linear regression###

(ar,br)=polyfit(np.log10(eff_radius_leaf[NH2_mean > 10*8.0e20/2.35*5. ]),np.log10(mass_leaf[NH2_mean > 10*8.0e20/2.35*5. ]),1)
xr=polyval([ar,br],np.log10(r.value))

ax3.plot(r,np.power(10,xr),'r--')

##linear regression###
(ar1,br1)=polyfit(np.log10(eff_radius_leaf[NH2_mean > 10*8.0e20/2.35*5. ]),np.log10(mass_corrected_leaf[NH2_mean > 10*8.0e20/2.35*5. ]),1)
xr1=polyval([ar1,br1],np.log10(r.value))

ax3.plot(r,np.power(10,xr1),'--',color='purple')
print ar1,ar,br1,br
##################save the plot########################################
fig.savefig(outpath+'W43_main_dendro_mass_radius1_erosion_prune.pdf'
            ,bbox_inches='tight')






#density distribution and log-normal fitting
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

from lmfit import Parameters, minimize, fit_report

import powerlaw_modi
import plfit_modi_2

params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)
##seg linear fits###################
def seg_model_strengths_2(params, lags):
 (m0,m1,b0,p1) = params
 b1 = m0*p1+b0-m1*p1
 #b0 = 1.5*m0
 num_data = len(lags)
 model_strengths = np.empty(num_data)
 idx = 0
 while idx < num_data:
         lag = lags[idx]
         if lag < p1:
            model_strength = m0*lag+b0
         elif p1<= lag:
             model_strength = m1*lag+b1
         model_strengths[idx] = model_strength
         idx += 1
 return model_strengths
 
def residual2(pars, x, data=None):
    vals = pars.valuesdict()
    k1 =  vals['k1']
    k2 =  vals['k2']
    b0 = vals['b0']
    p1 = vals['p1']
    model =  seg_model_strengths_2(params =np.array([k1,k2,b0,p1]), lags=x)
    if data is None:
        return model
    return (model - data)
####################################################################  
dirpath = r'./'
outpath = r'./'
############################################################################
filepath1= dirpath+r'W43_main_N_trim.fits'
hdulist1 = fits.open(filepath1)
############################################################################# 
data1 = np.power(10,(hdulist1[0].data))
cutoff = np.power(10.,22.18 )

#subtract the contamination
y2 = (data1[np.where(np.logical_not(np.isnan(data1)))].flatten()-10**21.16)
y2 = y2[y2>(cutoff-10**21.16)]

meanvalue = np.mean(y2)
y2 = y2/np.mean(y2)

fit_x = (np.linspace(1e21, 1e24, 2000))

#pl fit from minimum
ro_fit = powerlaw_modi.Fit((y2),xmin=(y2.min()),linear_bins=True)

#optimal pl fit
ro1_fit = powerlaw_modi.Fit((y2))
bin_edges1,probability2 = ro1_fit.pdf()
x_set = np.log(ro1_fit.power_law.xmin)

########################draw the plfit results with xmin derived from powerlaw-package###################
MyPL0 = plfit_modi_2.plfit(y2,xmin=y2.min())
MyPL = plfit_modi_2.plfit(y2,xmin=np.exp(x_set))

test0 = MyPL0.plotpdf(normed=False,histcolor='none',plcolor='none')
s0 = []
for i in range(len(test0[1])):
  if i < len(test0[1])-1:
     s0 = np.append(s0,[test0[1][i+1]-test0[1][i]])

test = MyPL.plotpdf(normed=False,histcolor='none',plcolor='none')
s = []
for i in range(len(test[1])):
  if i < len(test[1])-1:
     s = np.append(s,[test[1][i+1]-test[1][i]])
################fit the broken powerlaw###############
#(test[2]),test[3]/np.sum(test[0][1:]*s)
#np.log(test[1]),(test[0]/np.sum(test[0][1:]*s))

#g_init = BrokenPowerLaw1D(amplitude=0.5,x_break=np.exp(2.),alpha_1=2.7,alpha_2=2.1)
#fitter = LevMarLSQFitter()
#bins_n=test[2]
#t1,t2,t3=plt.hist(y[y>400], bins_n, facecolor='none',edgecolor = "grey", alpha=0.5,normed=1,histtype=u'stepfilled') 
#print g_bpl.amplitude.value,g_bpl.x_break.value,g_bpl.alpha_1.value,g_bpl.alpha_2.value
#x_set1 = np.log(g_bpl.x_break.value)
#ro2_fit = powerlaw_modi.Fit((y2),xmin=g_bpl.x_break.value)
#MyPL2 = plfit_modi_2.plfit(y2,xmin=g_bpl.x_break.value)
#test1 = MyPL2.plotpdf(normed=False,histcolor='none',plcolor='none')
#s1 = []     
#for i in range(len(test1[1])):
  #if i < len(test1[1])-1:
   #  s1 = np.append(s1,[test1[1][i+1]-test1[1][i]])
     
#x_lower=np.linspace(np.min(y),g_bpl.x_break.value,200)
#x_higher=np.linspace(g_bpl.x_break.value,np.max(bins[1:]),200)
#plt.plot((x_lower),(g_bpl(x_lower)),'r-',lw=2,label='broken power-law')
#plt.plot((x_higher),(g_bpl(x_higher)),'r-',lw=2)


data  = np.log(test[0]/np.sum(test[0][1:]*s))[test[1]>np.exp(x_set)]#np.log(3.4/4.95)
test[1][test[1]>np.exp(x_set)], (test[0]/np.sum(test[0][1:]*s))[test[1]>np.exp(x_set)]
fit_params1 = Parameters()
fit_params1.add('k1', value=-0.3)
fit_params1.add('k2', value=-0.1)

fit_params1.add('b0', value=-0.3)
fit_params1.add('p1', value=0.8)
x=np.log(test[1][test[1]>np.exp(x_set)])
#f = data.min()
#data = data[data>f]
#f = data.min()
#data = data[data>f]
#x = x[data>f]
out1 = minimize(residual2, fit_params1, args=(x,),method='leastsq', kws={'data':data})

print(fit_report(out1))

ro2_fit = powerlaw_modi.Fit((y2),xmin=np.exp(out1.params['p1'].value))
MyPL2 = plfit_modi_2.plfit(y2,xmin=np.exp(out1.params['p1'].value))

test1 = MyPL2.plotpdf(normed=False,histcolor='none',plcolor='none')

s1 = []     
for i in range(len(test1[1])):
  if i < len(test1[1])-1:
     s1 = np.append(s1,[test1[1][i+1]-test1[1][i]])


#plt.axvline((s[4]))
plt.clf()
fig = plt.figure()
ax1 = fig.add_subplot(111)

ax2 = ax1.twiny()
ax2.set_xlabel('$log_{10}N(H_{2})$')
ticks1 =np.linspace(-1.5,2.5,6)
ax1.set_xticks(ticks=ticks1,minor=True)
#ax1.plot(np.log(s[1][s[1]<(cutoff/meanvalue)]),(s[0])[s[1]<(cutoff/meanvalue)],linestyle='--', drawstyle='steps',color='k')
#ax1.plot(np.log(s[1][s[1]>(cutoff/meanvalue)]),(s[0])[s[1]>(cutoff/meanvalue)],linestyle='-', drawstyle='steps',color='k')
##############draw the first PL#####################
cutoff = cutoff-10**21.16

norm_n = np.log(test0[1][test0[1]>(cutoff/meanvalue)])
norm_p = (test0[0]/np.sum(test0[0][1:]*s))[[test0[1]>(cutoff/meanvalue)]]
ax1.plot(norm_n,norm_p,linestyle='-', drawstyle='steps',color='k',linewidth = 1.0)

ax1.errorbar(0.5*(norm_n[:-1]+norm_n[1:]),norm_p[1:],yerr=((np.sqrt(test0[0]))/np.sum(test0[0][1:]*s))[[test0[1]>(cutoff/meanvalue)]][1:],capsize=1.0,capthick=1,linewidth=1.0, ls='none', color='black')
ax1.plot(np.log(test[2]),test[3]/np.sum(test[0][1:]*s),'r--',linewidth = 1.5,label='power-law fit from optimal column density')
ax1.plot(np.log(test0[2]),test0[3]/np.sum(test0[0][1:]*s0),'g--',linewidth = 1.5,label='overall power-law fit')


ax1.set_yscale('log')
#ax1.plot(np.log(s[2]),10.0*(s[3]),'r--',linewidth = 1.5)

###############draw the second PL#################
#ax1.plot(np.log(test1[2]),test1[3]/np.sum(test1[0][1:]*s1),'b--',linewidth = 1.5)
ax1.plot(np.log(s1[2]),10.0*(s1[3]),'b--',linewidth = 1.5,label='power-law fit at high density')

##############calculate the mass percentage of LN and PL######################################
x_set1 = out1.params['p1'].value
y_set = np.exp(x_set)*meanvalue
y_set1 = np.exp(x_set1)*meanvalue
y3 = y2*meanvalue
mass_excess_per = np.sum(y3[y3>y_set])/np.sum(y3)#*2.8*con.m_p.cgs/(1.9891e+33*u.g)*(((1.5*distance/206265.).to(u.cm))**2)
mass_excess_per1 = np.sum(y3[y3>y_set1])/np.sum(y3)
###############################################################################################
fitx = np.linspace(-1.2,2.1,1000)
y_ln = 1.0/(ro_fit.lognormal.sigma*np.sqrt(2*np.pi))*np.exp(-1*(fitx-ro_fit.lognormal.mu)**2/(2*ro_fit.lognormal.sigma**2))

ax1.set_ylim((1e-4,3.0))
ax1.set_xlim((-1.5,4))
ax1.set_xscale('linear')
ax1.set_ylabel('p($\eta$)')
ax1.text(1.5,0.2,'$s_{0}$ = %.2f (%.2f)\n$s_{1}$ = %.2f (%.2f)\n$s_{2}$ = %.2f (%.2f)'%(-1*ro_fit.power_law.alpha,ro_fit.power_law.sigma,-1*ro1_fit.power_law.alpha,ro1_fit.power_law.sigma,-1*ro2_fit.power_law.alpha,ro2_fit.power_law.sigma),size='large',color='black')

ax1.set_xlabel('$\eta$')
ax1.axvline(x = np.log(cutoff/meanvalue),color='grey',linestyle=':')

ax1.text(-1.3,0.001,'W43-main',color='black',size='x-large')

ax1.text(0.5,0.01,'$%.0f $%%$\ of\ the\ mass$'%(mass_excess_per*100),color='red')
ax1.text(0.5,0.001*3,'$%.0f $%%$\ of\ the\ mass$'%(mass_excess_per1*100),color='blue')

ticks2 = np.log10(np.exp(np.linspace(-1.5,2.5,6))*meanvalue)
ax2.set_xticklabels(['%.2f' % x for x in ticks2], 
fontsize=8) 

ax2.set_xticks(ticks=ticks2,minor=True)
ax1.legend(loc='lower right', shadow=False,fontsize=12.)
plt.show()

print ro_fit.lognormal.mu, ro_fit.lognormal.sigma,ro1_fit.power_law.alpha,ro2_fit.power_law.alpha,meanvalue,x_set,x_set1

plt.savefig(outpath+'W43-main_pdf_2PL_xmin_err.pdf')

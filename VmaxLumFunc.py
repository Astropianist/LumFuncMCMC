import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
import matplotlib as mp
mp.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lmfit import Model
from astropy.table import Table
from HaHbStacking_even import get_bins
from itertools import cycle
from scipy.optimize import fsolve
from mpmath import gammainc
import sys
import os
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
import seaborn as sns
sns.set_context('paper')
mp.rcParams['font.family']='serif'
import os.path as op
sns.set_style("ticks")
sns.set_style({"xtick.direction": "in","ytick.direction": "in",
               "xtick.top":True, "ytick.right":True,
               "xtick.major.size":12, "xtick.minor.size":4,
               "ytick.major.size":12, "ytick.minor.size":4,
               })
### color palettes
colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
colors += ["cloudy blue", "browny orange", "dark sea green"]
sns.set_palette(sns.xkcd_palette(colors))
orig_palette_arr = sns.color_palette()
orig_palette = cycle(tuple(orig_palette_arr))
markers = cycle(tuple(['o','^','*','s','+','v','<','>']))

h, OmegaM, OmegaL, a0 = 0.6778,  0.30821, 0.69179, 1.0
H0 = 0.000333562*h #H0=100h km/s/Mpc in units Mpc^-1
n = 100 #For bootstrap analysis
sqarcsec = 4.0*np.pi * (180./np.pi * 3600.0)**2

def schechter(L, al, phistar, Lstar):
    """ Schechter function """
    return phistar * (L/Lstar)**al * np.exp(-L/Lstar)

def schechter_log(L,al,phistar,Lstar):
    """ Schechter function for logL space; also log params """
    return np.log(10.0) * 10**phistar * 10**((L-Lstar)*(al+1))*np.exp(-10**(L-Lstar))

def schechter_integ(L,al,integ,Lstar):
    """ Schechter function with the integral over the function as a parameter instead of phi* """
    # integ_nophi, err = quad(schechter,1.0e36/Lstar,1.0e50/Lstar,args=(al,1.0,1.0))
    integ_nophi = float(gammainc(al+1,1.0e36/Lstar))
    phistar = integ/integ_nophi
    return schechter(L,al,phistar,Lstar)

def schechter_integ_log(L,al,integ,Lstar):
    """ Schechter function with the integral over the function as a parameter instead of phi* using logL, log(integral), and log(Lstar) values """
    # integ_nophi, err = quad(schechter_log,36.0,50.0,args=(al,0.0,Lstar))
    # integ_nophi = np.log10(integ_nophi)
    integ_nophi = float(gammainc(al+1,10**(36.0-Lstar)))
    # if integ_nophi<0.0: print "Negative integ_nophi for schechter_integ_log given params %f, %f, %f"%(al,integ,Lstar)
    integ_nophi = np.log10(integ_nophi)
    phistar = integ - integ_nophi
    sch = schechter_log(L,al,phistar,Lstar)
    # if np.isnan(sch).any() or np.isinf(sch).any():
    #     print "NaN value gotten for schechter_integ_log given params %.3f, %.2f, %.2f"%(al,integ,Lstar)
    #     for s in sch: print s
    return sch

def p(F,Flim=3.0e-17,alpha=-3.5,Fmin=0.0):
    """ Completeness (Fleming) curve as function of Flim and alpha """
    return 0.5*(1.0 - (2.5*alpha*np.log10(F/Flim))/np.sqrt(1.0+ (2.5*alpha*np.log10(F/Flim))**2))

def Ha(a,w=-1.0,omega_m=OmegaM,omega_l=OmegaL,omega_r=0.,omega_k=0.): 
    """ Hubble parameter as a function of a; this ignores the complicated transition of neutrinos from relativistic to non-relativistic in the early universe """
    return H0*np.sqrt(omega_m*a**(-3) + omega_l*a**(-3*(w+1)) + omega_r*a**(-4) + omega_k*a**(-2))

def Hz(z,w=-1.0):
    """ Hubble parameter as a function of z """
    return Ha(1/(1+z),w)

def Hzinv(z,w):
    """ 1/H(z) for integration purposes """
    return 1.0/Hz(z,w)
    
def chiz(z,w=-1.0): 
    """ Integral to get basic comoving radial coordinate chi(z) """
    ans, err = quad(Hzinv,0.,z,args=(w))
    return ans

def dAz(z,w=-1.0): 
    """ Angular diameter distance as a function of z in Mpc """
    return chiz(z,w)/(1.0+z)

def dLz(z,w=-1.0):
    """ Luminosity distance as a function of z in Mpc """
    return dAz(z,w)*(1.+z)**2
    
def dVdz(z,w=-1.0):
    """ Volume differential--does not include the area multiplication--just the lengthwise (along z-change) component; unit is Mpc^3 """
    return 4.0*np.pi*dAz(z,w)**2/(a0*Hz(z,w))

def lumfuncint(z,F,Omega_0,Flim,alpha,Fmin): 
    """ Integrand of luminosity function MLE
    
    Input
    -----
    F: Float
        Flux in erg/cm^2/s
    Omega_0: Float 
        Effective survey area in square arcseconds
    Flim: Float
        Flux at which there's 50% completeness (according to Fleming curve) in erg/cm^2/s
    alpha: Float
        Fleming curve alpha (slope) parameter
    """
    return Omega_0/sqarcsec * p(F,Flim,alpha,Fmin)*dVdz(z)

def lumfuncintv2(z,F,Omega_0,func,Flim,alpha,Fmin):
    """ Integrand of luminosity function MLE for faster computation
    
    Input
    -----
    F: Float
        Flux in erg/cm^2/s
    Omega_0: Float 
        Effective survey area in square arcseconds
    func: Interp1d function object
        Interpolation function for dV/dz(z) for much quicker computation
    Flim: Float
        Flux at which there's 50% completeness (according to Fleming curve) in erg/cm^2/s
    alpha: Float
        Fleming curve alpha (slope) parameter
    """
    return Omega_0/sqarcsec * p(F,Flim,alpha,Fmin)*func(z)

#phi(L)--1/Veff estimator
def lumfunc(F,func,Omega_0=100.0,minz=1.16,maxz=1.9,Flim=3.0e-17,alpha=-3.5,Fmin=0.0):
    """ Luminosity function volume^-1 weights for a given flux

    Input
    -----
    F: Float
        Flux in erg/cm^2/s
    func: Interp1d function object
        Interpolation function for dV/dz(z) for much quicker computation
    Omega_0: Float 
        Effective survey area in square arcseconds
    minz: Float
        Minimum redshift in sample
    maxz: Float
        Maximum redshift in sample
    Flim: Float
        Flux at which there's 50% completeness (according to Fleming curve) in erg/cm^2/s
    alpha: Float
        Fleming curve alpha (slope) parameter
    """
    ans, err = quad(lumfuncintv2, minz, maxz, args=(F,Omega_0,func,Flim,alpha,Fmin))
    return 1.0/ans

def getlumfunc(F,z,Omega_0=100.0,Flim=3.0e-17,alpha=-3.5,Fmin=0.0):
    """ Computation of luminosities and effective volume weights given fluxes and redshifts in sample 

    Input
    -----
    F: 1-D Numpy Array
        Sample fluxes in erg/cm^2/s
    z: 1-D Numpy Array (same size as F)
        Sample redshifts
    Omega_0: Float 
        Effective survey area in square arcseconds
    Flim: Float
        Flux at which there's 50% completeness (according to Fleming curve) in erg/cm^2/s
    alpha: Float
        Fleming curve alpha (slope) parameter
    Fmin: Float
        Minimum flux considered in the Veff method (erg/cm^2/s)

    Return
    ------
    Lfunc: 1-D Numpy Array
        Array of luminosities pertaining to fluxes in erg/s
    phifunc: 1-D Numpy Array
        Array of volume^-1 weights for each flux
    minz: Float
        Minimum z value in data set
    maxz: Float
        Maximum z value in data set
    """
    minz, maxz = min(z), max(z)
    zint = np.linspace(0.95*minz,1.05*maxz,1001)
    ######### Create interpolation function for dV/dz
    dVdzint = np.zeros_like(zint)
    for i,zi in enumerate(zint):
        dVdzint[i] = dVdz(zi)
    dVdzf = interp1d(zint,dVdzint)
    ######## Get luminosity and effective volume^-1 weights for each flux #####
    Lfunc, phifunc = np.zeros(len(F)), np.zeros(len(F))
    for i in range(len(F)):
        Lfunc[i] = 4.0*np.pi*(dLz(z[i])*3.086e24)**2*F[i]
        phifunc[i] = lumfunc(F[i],dVdzf,Omega_0,minz,maxz,Flim,alpha,Fmin)
    return Lfunc, phifunc, minz, maxz

def getBootErrLog(L,phi,minz,maxz,nboot=100,nbin=25,Fmin=1.0e-20):
    """ Estimate true luminosity function and errors on the "measurements" using bootstrap method
    This function is for log luminosities and is used with a Schechter function with log quantities (not in this code)

    Input
    -----
    L: 1-D Numpy Array
        Array of log luminosities in log(erg/s)
    phi: 1-D Numpy Array
        Array of volume^-1 weights using V_eff method
    minz: Float
        Minimum z value in data set
    maxz: Float
        Maximum z value in data set
    nboot: Int
        Number of iterations to use for bootstrap method
    nbin: Int
        Number of bins for creating luminosity function
    Fmin: Float
        Minimum flux considered in the Veff method (erg/cm^2/s)

    Return
    ------
    Lavg: 1-D Numpy Array
        Array of average log luminosities in each bin
    lfbinorig: 1-D Numpy Array
        Array of true luminosity function values dn/dlogL in each bin
    var: 1-D Numpy Array
        Array of variances derived from bootstrap method
    """
    ##### Bin the data by luminosity to create a true luminosity function #####
    Larr = np.linspace(min(L),max(L),nbin+1) #To establish bin boundaries
    Lavg = np.linspace((Larr[0]+Larr[1])/2.0,(Larr[-1]+Larr[-2])/2.0,len(Larr)-1) #Centers of bins
    dL = Lavg[1]-Lavg[0]
    lfbin = np.zeros((nboot,len(Lavg)))
    lfbinorig = np.zeros(len(Lavg))
    for j in range(len(lfbinorig)):
        cond1 = L>=Larr[j]
        cond2 = L<Larr[j+1]
        cond = np.logical_and(cond1,cond2)
        if len(phi[cond]): 
            lfbinorig[j] = sum(phi[cond])/dL
    ###### Bootstrap calculation for errors ######
    for k in range(nboot):
        boot = np.random.randint(len(phi),size=len(phi))
        for j in range(len(Lavg)):
            cond1 = L[boot]>=Larr[j]
            cond2 = L[boot]<Larr[j+1]
            cond = np.logical_and(cond1,cond2)
            if len(phi[boot][cond]):
                lfbin[k,j] = sum(phi[boot][cond])/dL
    binavg = np.average(lfbin,axis=0)
    var = 1./(nboot-1) * np.sum((lfbin-binavg)**2,axis=0)
    var[var<=0.0] = min(var[var>0.0]) #Don't want values of 0 in variance
    ########### Correct for luminosity bins that are only partially included #########
    zarr = np.linspace(0.9*minz,1.1*maxz,201)
    Lminarr = np.zeros_like(zarr)
    for i,zi in enumerate(zarr):
        Lminarr[i] = get_L_constF(Fmin,zi)
    Lminarr = np.log10(Lminarr)
    Lminzf = interp1d(zarr,Lminarr,kind='cubic')
    for j in range(len(lfbinorig)):
        mult = get_mult_factor(Larr[j],Larr[j+1],Lminzf,minz,maxz)
        # print "mult[%d]=%.2f"%(j,mult)
        if abs(mult-1.0)<1.0e-8:
            break
        lfbinorig[j]*=mult
        var[j]*=mult**2
    return Lavg, lfbinorig, var

def getBootErr(L,phi,minz,maxz,nboot=100,nbin=25,Fmin=0.0):
    """ Estimate true luminosity function and errors on the "measurements" using bootstrap method
    This function is for linear luminosities 

    Input
    -----
    L: 1-D Numpy Array
        Array of luminosities in erg/s
    phi: 1-D Numpy Array
        Array of volume^-1 weights using V_eff method
    minz: Float
        Minimum z value in data set
    maxz: Float
        Maximum z value in data set
    nboot: Int
        Number of iterations to use for bootstrap method
    nbin: Int
        Number of bins for creating luminosity function
    Fmin: Float
        Minimum flux considered in the Veff method (erg/cm^2/s)
    
    Return
    ------
    Lavg: 1-D Numpy Array
        Array of average luminosities in each bin
    lfbinorig: 1-D Numpy Array
        Array of true luminosity function values dn (not divided by luminosity interval yet) in each bin
    var: 1-D Numpy Array
        Array of variances derived from bootstrap method
    """
    ##### Bin the data by luminosity to create a true luminosity function #####
    Larr = np.linspace(min(L),max(L),nbin+1) #To establish bin boundaries
    Lavg = np.linspace((Larr[0]+Larr[1])/2.0,(Larr[-1]+Larr[-2])/2.0,len(Larr)-1) #Centers of bins

    lfbin = np.zeros((nboot,len(Lavg)))
    lfbinorig = np.zeros(len(Lavg))
    for j in range(len(lfbinorig)):
        cond1 = L>=Larr[j]
        cond2 = L<Larr[j+1]
        cond = np.logical_and(cond1,cond2)
        if len(phi[cond]): 
            lfbinorig[j] = sum(phi[cond])
    ###### Bootstrap calculation for errors ######
    for k in range(nboot):
        boot = np.random.randint(len(phi),size=len(phi))
        for j in range(len(Lavg)):
            cond1 = L[boot]>=Larr[j]
            cond2 = L[boot]<Larr[j+1]
            cond = np.logical_and(cond1,cond2)
            if len(phi[boot][cond]):
                lfbin[k,j] = sum(phi[boot][cond])
    binavg = np.average(lfbin,axis=0)
    var = 1./(nboot-1) * np.sum((lfbin-binavg)**2,axis=0)
    var[var<=0.0] = min(var[var>0.0]) #Don't want values of 0 in variance
    ########### Correct for luminosity bins that are only partially included #########
    zarr = np.linspace(0.9*minz,1.1*maxz,201)
    Lminarr = np.zeros_like(zarr)
    for i,zi in enumerate(zarr):
        Lminarr[i] = get_L_constF(Fmin,zi)
    Lminzf = interp1d(zarr,Lminarr,kind='cubic')
    for j in range(len(lfbinorig)):
        mult = get_mult_factor(Larr[j],Larr[j+1],Lminzf,minz,maxz)
        # print "mult[%d]=%.2f"%(j,mult)
        if abs(mult-1.0)<1.0e-8:
            break
        lfbinorig[j]*=mult
        var[j]*=mult**2
    return Lavg, lfbinorig, var

def fit_Schechter(Lavg,lfbinorig,var,name='OIII',alpha_value=None,log=False,integ=False):
    """ Using lmfit to fit Schechter function to the measured (true) luminosity function 
    
    Input
    -----
    Lavg: 1-D Numpy Array
        Array of average luminosities in each bin
    lfbinorig: 1-D Numpy Array
        Array of true luminosity function values dn (not divided by luminosity interval yet) in each bin
    var: 1-D Numpy Array
        Array of variances derived from bootstrap method
    name: String
        Name of line or monochromatic luminosity quantity
    alpha_value: Float
        Value for alpha parameter if one wants it fixed
    log: Bool
        Whether the log form of the Schechter Equation should be fit or not
    integ: Bool
        Whether the integrated parameter should be used instead of phistar

    Return
    ------
    schfit: LMFIT fit result object
    """
    if log and integ:
        schmod = Model(schechter_integ_log)
    elif log and (not integ):
        schmod = Model(schechter_log)
    elif integ and (not log):
        schmod = Model(schechter_integ)
    else:
        schmod = Model(schechter)
    pars = schmod.make_params()
    # print "List of parameters:", pars
    if alpha_value is not None:
        al_val = alpha_value
        pars['al'].set(value=al_val, vary=False)
    else:
        if name=='OIII': 
            al_val = -2.461
        else: 
            al_val = -1.714
        pars['al'].set(value=al_val,max=0.0,min=-5.0)

    if not log: 
        if not integ:
            pars['phistar'].set(value=1.0e-3,min=0.0)
        else:
            pars['integ'].set(value=0.1,min=0.0)
        pars['Lstar'].set(value=1.0e42,min=1.0e40,max=1.0e45)
    else:
        if not integ:
            pars['phistar'].set(value=-3.0,min=-8.0)
        else:
            pars['integ'].set(value=-1.0,min=-7.0,max=5.0)
        pars['Lstar'].set(value=42.5,min=40.0,max=45.0)
    
    cond = lfbinorig>0.0
    schfit = schmod.fit(lfbinorig,pars,L=Lavg,weights=1.0/np.sqrt(var))
    print schfit.fit_report()
    return schfit

def plotSchechter(Lavg,lfbinorig,var,schfit,name,img_dir="ImageFiles",log=False,integ=False):
    """ Plotting best-fit Schechter function over true luminosity function measurements 
    Note: Here, we divide the dn values of luminosity function by the interval Delta(L)/L* to get a real LF
    
    Input
    -----
    Lavg: 1-D Numpy Array
        Array of average luminosities in each bin
    lfbinorig: 1-D Numpy Array
        Array of true luminosity function values dn (not divided by luminosity interval yet) in each bin
    var: 1-D Numpy Array
        Array of variances derived from bootstrap method
    schfit: LMFIT fit result object
    name: String
        Name of line or monochromatic luminosity quantity
    img_dir: String
        Directory for placing produced figure
    log: Bool
        Whether things are log or not
    integ: Bool
        Whether the integrated parameter should be used instead of phistar
    """
    pars = schfit.params
    fig, ax = plt.subplots()
    if not log:
        ax.set_xscale("log")
    ax.set_yscale("log")
    if not log:
        dL = Lavg[1]-Lavg[0]
        ratio = pars['Lstar']/dL
        if not integ:
            ax.plot(Lavg, schfit.best_fit*ratio, 'r-', label=r'Fit: $\alpha=%.3f$, $\phi_*=%.1e$, $L_*=%.1e$' % (pars['al'],pars['phistar']*ratio,pars['Lstar']))
        else:
            ax.plot(Lavg, schfit.best_fit*ratio, 'r-', label=r'Fit: $\alpha=%.3f$, ${\rm{I}}=%.1e$, $L_*=%.1e$' % (pars['al'],pars['integ']*ratio,pars['Lstar']))
    else:
        ratio = 1.0
        if not integ:
            ax.plot(Lavg, schfit.best_fit*ratio, 'r-', label=r'Fit: $\alpha=%.3f$, $\log \phi_*=%.2f$, $\log {\rm{L}}_*=%.2f$' % (pars['al'],pars['phistar']*ratio,pars['Lstar']))
        else:
            ax.plot(Lavg, schfit.best_fit*ratio, 'r-', label=r'Fit: $\alpha=%.3f$, $\log {\rm{I}}=%.2f$, $\log {\rm{L}}_*=%.2f$' % (pars['al'],pars['integ']*ratio,pars['Lstar']))
    
    ax.errorbar(Lavg,lfbinorig*ratio,yerr=np.sqrt(var)*ratio,fmt='b^',label='Measured LF')
    try:
        dely = schfit.eval_uncertainty(sigma=3)
        ax.fill_between(Lavg,ratio*(schfit.best_fit-dely),ratio*(schfit.best_fit+dely),color='r',alpha=0.2,label=r'$3 \sigma$ Uncertainty Band')
    except: pass
    if not log:
        plt.xlabel(r"L (erg s$^{-1}$)")
        plt.ylabel(r"$dn/d({\rm{L/L_*}})$ (Number Mpc$^{-3}$)")
    else:
        plt.xlabel(r"$\log$ L (erg s$^{-1}$)")
        plt.ylabel(r"$\phi_{\rm{true}}$ (Number Mpc$^{-3}$ dex$^{-1}$)")
    
    plt.legend(loc='best')
    plt.minorticks_on()
    if not log:
        if not integ:
            fn = op.join(img_dir,"Lin",name)
        else:
            fn = op.join(img_dir,"Lin","%s_integ.png"%(name.split('.')[0]))
    else:
        if not integ:
            fn = op.join(img_dir,"Log","%s_log.png"%(name.split('.')[0]))
        else:
            fn = op.join(img_dir,"Log","%s_log_integ.png"%(name.split('.')[0]))
    plt.savefig(fn,bbox_inches='tight',dpi=300)
    plt.close()

def combineSteps(F,z,name,Omega_0=100.0,Flim=3.0e-17,alpha=-3.5,nboot=100,nbin=25,img_dir='../LuminosityFunction/Veff',Fmin=0.0,integ=False):
    """ Basically perform multiple functions to simplify necessary commands; see other functions for detailed input and output descriptions """
    print "About to start Veff process for", name
    print "Length of arrays:", len(F), len(z)
    Lfunc, phifunc, minz, maxz = getlumfunc(F,z,Omega_0,Flim,alpha,Fmin)
    print "Finished calculating true luminosity function"
    Lavg, lfbinorig, var = getBootErr(Lfunc,phifunc,minz,maxz,nboot,nbin,Fmin)
    print "Finished getting bootstrap-based errors"
    schfit = fit_Schechter(Lavg,lfbinorig,var,integ=integ)
    print "Fit Schechter function to true luminosity function"
    plotSchechter(Lavg,lfbinorig,var,schfit,name,img_dir,integ=integ)
    print "Finished plotting true luminosity and best-fit Schechter fit"

def combineStepsLog(F,z,name,Omega_0=100.0,Flim=3.0e-17,alpha=-3.5,nboot=100,nbin=25,img_dir='../LuminosityFunction/Veff',Fmin=0.0,integ=False):
    """ Basically perform multiple functions to simplify necessary commands in log case; see other functions for detailed input and output descriptions """
    print "About to start Veff process for", name
    print "Length of arrays:", len(F), len(z)
    Lfunc, phifunc, minz, maxz = getlumfunc(F,z,Omega_0,Flim,alpha,Fmin)
    print "Finished calculating true luminosity function"
    Lavg, lfbinorig, var = getBootErrLog(np.log10(Lfunc),phifunc,minz,maxz,nboot,nbin,Fmin)
    print "Finished getting bootstrap-based errors"
    schfit = fit_Schechter(Lavg,lfbinorig,var,log=True,integ=integ)
    print "Fit Schechter function to true luminosity function"
    plotSchechter(Lavg,lfbinorig,var,schfit,name,img_dir,log=True,integ=integ)
    print "Finished plotting true luminosity and best-fit Schechter fit"

def zEvolSteps(F,z,name,Omega_0=100.0,Flim=3.0e-17,alpha=-3.5,nboot=100,nbins=25,img_dir='../LuminosityFunction/Veff',zbins=5,Fmin=0.0,log=False,integ=False):
    """ Perform multiple functions to simplify necessary commands; in addition, bin overall sample by redshift and compute luminosity function for each bin, keeping alpha constant for additional redshift bins. See other functions for detailed descriptions of inputs and outputs """
    print "About to start Veff process for", name
    print "Length of arrays:", len(F), len(z)
    fig, ax = plt.subplots()
    if not log:
        ax.set_xscale("log")
    ax.set_yscale("log")
    indhist = get_bins(z,zbins)
    bin_edges = min(z)*np.ones(zbins+1)
    alpha_value = None
    for i in range(zbins):
        # print "Starting z-bin Number", i+1
        condhist = indhist == i
        if i==zbins-1: 
            bin_edges[i+1] = max(z)
            zlabel = r"$%.2f < z \leq %.2f$"%(bin_edges[i],bin_edges[i+1])
        else: 
            condhist2 = indhist == i+1
            bin_edges[i+1] = (max(z[condhist])+min(z[condhist2]))/2.0
            if i==0: zlabel = r"$%.2f \leq z<%.2f$"%(bin_edges[i],bin_edges[i+1])
            else: zlabel = r"$%.2f<z<%.2f$"%(bin_edges[i],bin_edges[i+1])
        # print "Length of binned arrays:", len(F[condhist]),len(z[condhist])
        Lfunc, phifunc, minz, maxz = getlumfunc(F[condhist],z[condhist],Omega_0,Flim,alpha,Fmin=Fmin)
        # print "Finished calculating true luminosity function for bin number", i+1
        if not log:
            Lavg, lfbinorig, var = getBootErr(Lfunc,phifunc,minz,maxz,nboot,nbins,Fmin=Fmin)
            # print "Finished getting bootstrap-based errors for bin number", i+1
            schfit = fit_Schechter(Lavg,lfbinorig,var,name=name.split('_')[0],alpha_value=alpha_value,integ=integ)
            pars = schfit.params
            # if i==0: 
                # alpha_value = pars['al']
            dL = Lavg[1]-Lavg[0]
            ratio = pars['Lstar']/dL
            if not integ:
                label = r'%s: $\alpha=%.2f$, $\phi_*=%.1e$, ${\rm{L}}_*=%.1e$' % (zlabel,pars['al'],pars['phistar']*ratio,pars['Lstar'])
            else:
                label = r'%s: $\alpha=%.2f$, ${\rm{I}}=%.1e$, ${\rm{L}}_*=%.1e$' % (zlabel,pars['al'],pars['integ']*ratio,pars['Lstar'])
        else:
            Lavg, lfbinorig, var = getBootErrLog(np.log10(Lfunc),phifunc,minz,maxz,nboot,nbins,Fmin=Fmin)
            # print "Finished getting bootstrap-based errors for bin number", i+1
            schfit = fit_Schechter(Lavg,lfbinorig,var,name=name.split('_')[0],alpha_value=alpha_value,log=True,integ=integ)
            pars = schfit.params
            # if i==0: 
                # alpha_value = pars['al']
            ratio = 1.0
            if not integ:
                label = r'%s: $\alpha=%.2f$, $\log \phi_*=%.2f$, $\log {\rm{L}}_*=%.2f$' % (zlabel,pars['al'],pars['phistar']*ratio,pars['Lstar'])
            else:
                label = r'%s: $\alpha=%.2f$, $\log {\rm{I}}=%.2f$, $\log {\rm{L}}_*=%.2f$' % (zlabel,pars['al'],pars['integ']*ratio,pars['Lstar'])
        
        # print "Fit Schechter function to true luminosity function for bin number", i+1
        ax.errorbar(Lavg,lfbinorig*ratio,yerr=np.sqrt(var)*ratio,color=orig_palette.next(),marker=markers.next(),linestyle='none',label='')
        ax.plot(Lavg, schfit.best_fit*ratio, color=ax.lines[-1].get_color(), label=label)
        try:
            dely = schfit.eval_uncertainty(sigma=3)
            ax.fill_between(Lavg,ratio*(schfit.best_fit-dely),ratio*(schfit.best_fit+dely),color=ax.lines[-1].get_color(),alpha=0.2,label='')
        except: pass
    if not log:
        plt.xlabel(r"L (erg s$^{-1}$)")
        plt.ylabel(r"$dn/d({\rm{L/L_*}})$ (Number Mpc$^{-3}$)")
    else:
        plt.xlabel(r"$\log$ L (erg s$^{-1}$)")
        plt.ylabel(r"$\phi_{\rm{true}}$ (Number Mpc$^{-3}$ dex$^{-1}$)")
    plt.legend(loc='best',fontsize='x-small')
    plt.minorticks_on()
    if not log:
        if not integ:
            fn = op.join(img_dir,"Lin",name)
        else:
            fn = op.join(img_dir,"Lin","%s_integ.png"%(name.split('.')[0]))
    else:
        if not integ:
            fn = op.join(img_dir,"Log","%s_log.png"%(name.split('.')[0]))
        else:
            fn = op.join(img_dir,"Log","%s_log_integ.png"%(name.split('.')[0]))
    plt.savefig(fn,bbox_inches='tight',dpi=300)
    plt.close()
    print "Finished plotting true luminosity and best-fit Schechter fit"

def get_min_flux(min_comp_frac,Flim,alpha):
    """ Get flux corresponding to the minimum completeness fraction allowed
    
    Input
    -----
    min_comp_frac: Float
        Minimum completeness fraction considered
    Flim: Float
        Flim (F50) parameter for the Fleming completeness curve
    alpha: Float
        alpha parameter for the Fleming completeness curve
        
    Return
    ------
    min_flux: Float
        Minimum flux considered in erg/cm^2/s """
    if abs(min_comp_frac-0.0)<1.0e-6:
        return 0.0
    else:
        return fsolve(lambda x: p(x,Flim,alpha)-min_comp_frac,[Flim])[0]

def get_L_constF(F,z):
    """ Get the luminosity at a given flux value
    
    Input
    -----
    F: Float
        Flux in erg/cm^2/s
    z: Float
        Redshift
        
    Return
    ------
    L: Float
        Luminosity (erg/s) """
    return 4.0*np.pi*(dLz(z)*3.086e24)**2 * F

def get_mult_factor(lum0,lum1,Lminzf,zmin,zmax):
    """ Factor to multiply counts by when a luminosity bin has values not considered at some redhshifts
    Input
    -----
    lum0, lum1: Floats
        Bin edges for luminosity
    Lminzf: Interp1d Function
        Function for getting min luminosity considered at a given redshift
    zmin, zmax: Floats
        Min and max redshifts in the sample
        
    Return
    ------
    mult_factor: Float
        Value to multiply counts by for the given bin """
    # print "lum0, Lminzf(zmin), Lminzf(zmax), lum1: %.2e, %.2e, %.2e, %.2e"%(lum0,Lminzf(zmin),Lminzf(zmax),lum1)
    if Lminzf(zmin)>=lum1: 
        return np.inf
    elif Lminzf(zmax)<=lum0:
        return 1.0
    else:
        denom, err = quad(lambda x: min(max(lum1-Lminzf(x),0.0),lum1-lum0), zmin, zmax)
        return (zmax-zmin)*(lum1-lum0) / denom

def main():
    dat = Table.read("../AllTextFiles/combined_all_Swift_AEB_515_NoAGN.dat",format='ascii')
    oiii = dat['OIII5007']; ha = dat['Ha']; z = dat['z']
    oiii *= 3.98/2.98 #All OIII
    Ha *= 0.71 #NII correction
    min_comp_frac = 0.5
    # Flim_OIII, Flim_Ha = 4.0e-17, 3.1e-17
    Flim_OIII, Flim_Ha = 4.0e-17*3.98/2.98, 3.1e-17*0.71
    alpha_OIII, alpha_Ha = -2.12, -2.20
    rootoiii = get_min_flux(min_comp_frac,Flim_OIII,alpha_OIII)
    rootha = get_min_flux(min_comp_frac,Flim_Ha,alpha_Ha)
    condoiii = oiii>1.0e17*rootoiii; condha = ha>1.0e17*rootha
    nbin = 50
    Omega_0 = 1.0e6
    zbin_list = [1,3,5]
    # zbins = 1
    # nbin_list = [10,50,80]
    combineStepsLog(1.0e-17*oiii[condoiii],z[condoiii],"OIII_all_Vmax_LF_mcf_%d_bins_%d.png"%(int(100*min_comp_frac),nbin),Flim=Flim_OIII,alpha=alpha_OIII,nbin=nbin,Omega_0=Omega_0,Fmin=rootoiii,integ=True)
    combineStepsLog(1.0e-17*ha[condha],z[condha],"Ha_corr_Vmax_LF_mcf_%d_bins_%d_new.png"%(int(100*min_comp_frac),nbin),Flim=Flim_Ha,alpha=alpha_Ha,nbin=nbin,Omega_0=Omega_0,Fmin=rootha,integ=True)
    for zbins in zbin_list:
        zEvolSteps(1.0e-17*oiii[condoiii],z[condoiii],"OIII_all_Vmax_LF_zbin_%d_nbin_%d_mcf_%d.png"%(zbins,nbin,int(100*min_comp_frac)),Flim=Flim_OIII,alpha=alpha_OIII,nbins=nbin,zbins=zbins,Fmin=rootoiii,Omega_0=Omega_0,log=True,integ=True)
        zEvolSteps(1.0e-17*ha[condha],z[condha],"Ha_corr_Vmax_LF_zbin_%d_nbin_%d_mcf_%d.png"%(zbins,nbin,int(100*min_comp_frac)),Flim=Flim_Ha,alpha=alpha_Ha,nbins=nbin,zbins=zbins,Fmin=rootha,Omega_0=Omega_0,log=True,integ=True)

if __name__=='__main__': 
    main()
import numpy as np 
import logging
import emcee
import pickle
from uncertainties import unumpy, ufloat
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.integrate import trapz
from scipy.interpolate import RegularGridInterpolator as RGIScipy
from scipy.stats import binned_statistic
from math import lgamma
from astropy.table import Table
from time import time
import matplotlib.pyplot as plt
import corner
import VmaxLumFunc as V
from scipy.optimize import fsolve
from multiprocessing import Pool
import seaborn as sns
sns.set_context("paper",font_scale=1.3) # options include: talk, poster, paper
sns.set_style("ticks")
sns.set_style({"xtick.direction": "in","ytick.direction": "in",
               "xtick.top":True, "ytick.right":True,
               "xtick.major.size":12, "xtick.minor.size":4,
               "ytick.major.size":12, "ytick.minor.size":4,
               })

c = 3.0e18 # Speed of light in Angstroms/s
num_cores = 20 #In Joel's thingy

def poisson_lnpmf(k, mu):
    return k*np.log(mu) - lgamma(k+1) - mu

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def getRealLumRed(file_name='N501_with_atm.txt', interp_type='cubic', wav_rest=1215.67, delznum=51, logL_width=None):
    trans_dat = Table.read(file_name, format='ascii')
    lam, tra = trans_dat['lambda'], trans_dat['transmission']
    zs = (lam-wav_rest) / wav_rest
    if 'perfect' in file_name.lower():
        zmin, zmax = zs.min(), zs.max()
        return lambda x: np.piecewise(x, [x<=zmin, x<zmax, x>=zmax], [np.inf, 0.0, np.inf])
    del_logL = np.log10(tra.max()) - np.log10(tra)
    del_logL_lin = np.linspace(del_logL.min(), del_logL.max(), delznum)
    delz = np.zeros_like(del_logL_lin)
    for i, tv in enumerate(del_logL_lin):
        inds = np.where(del_logL<=tv)[0]
        inds_consec = consecutive(inds)
        zs_consec = [zs[indsi] for indsi in inds_consec]
        delz[i] = sum([zsi[-1] - zsi[0] for zsi in zs_consec])
    delzf = interp1d(del_logL_lin, delz, kind=interp_type, bounds_error=False, fill_value = (0, delz.max()))
    if logL_width is not None: delz_use = delzf(logL_width)
    else: delz_use = None
    return interp1d(zs, del_logL, kind=interp_type, bounds_error=False, fill_value = (del_logL[0], del_logL[-1])), delzf, delz_use, zs[np.argmax(tra)]

def getTransPDF(lam, tra, pdflen=10000, num_discrete=51, interp_type='cubic', wav_rest=1215.67):
    del_logL = np.log10(tra.max()) - np.log10(tra)
    il, ir = 0, len(del_logL)-1
    while del_logL[il+1]-del_logL[il]<0.0 or del_logL[il+1]>(del_logL.max()-del_logL.min())/2.0: il+=1
    while del_logL[ir-1]-del_logL[ir]<0.0 or del_logL[ir-1]>(del_logL.max()-del_logL.min())/2.0: ir-=1

    flat_frac = (lam[ir]-lam[il])/(lam[-1]-lam[0])

    fl = interp1d(del_logL[:il],lam[:il],kind=interp_type)
    fr = interp1d(del_logL[ir:],lam[ir:],kind=interp_type)

    del_logL_arr_orig = np.linspace(max(del_logL[:il].min(),del_logL[ir:].min()), min(del_logL[:il].max(),del_logL[ir:].max()),pdflen)
    del_z_arr = (fr(del_logL_arr_orig) - fl(del_logL_arr_orig))/wav_rest
    del_logL_arr = 1.0 * del_logL_arr_orig

    del_logL_arr_diff = np.diff(del_logL_arr)
    pdf_arr = np.diff(fr(del_logL_arr))/del_logL_arr_diff - np.diff(fl(del_logL_arr))/del_logL_arr_diff
    pdf_arr = np.append(pdf_arr, pdf_arr[-1])
    # del_logL_arr = np.append(del_logL_arr, tra.max())
    # pdf_arr = np.append(pdf_arr, pdf_arr[-1]) # Assume the PDF stays constant for the small sliver constituting the mostly flat top
    if del_logL_arr[0]-del_logL.min()>1.0e-12:
        del_logL_arr = np.insert(del_logL_arr, 0, del_logL.min())
        pdf_arr = np.insert(pdf_arr, 0, pdf_arr[0])
    integ = trapz(pdf_arr[1:], del_logL_arr[1:])
    pdf_arr[1:] *= (1.0-flat_frac) / integ # Normalize
    # pdf_arr[0] = flat_frac/(1.0-flat_frac) * integ / (del_logL_arr[1]-del_logL_arr[0])
    pdf_arr[0] = flat_frac / (del_logL_arr[1]-del_logL_arr[0])
    pdf_arr /= trapz(pdf_arr, del_logL_arr) # Just normalize again since the trapezoid rule is not a perfect integrator by any means
    log_pdf = np.log10(pdf_arr)
    diff_log = np.hstack([abs(np.diff(log_pdf)),0.0])
    diff_cumsum = np.cumsum(diff_log)/diff_log.sum() #Normalized cumulative sum
    logL_discrete = np.zeros(num_discrete)
    logL_discrete[-1] = del_logL_arr.max()
    indi_arr = np.zeros(del_logL_arr.size,dtype=int)
    for i in range(1,num_discrete-1):
        indi = np.argmin(abs(diff_cumsum-i/num_discrete))
        if indi<=indi_arr[i-1]:
            while indi<=indi_arr[i-1]: indi+=1
        indi_arr[i] = indi
        logL_discrete[i] = del_logL_arr[indi]

    # pdf_arr_sort, indsort = np.unique(pdf_arr, return_index=True)
    # f_reverse = interp1d(pdf_arr_sort,del_logL_arr[indsort],kind='cubic',fill_value=0.0,bounds_error=False)
    # pdf_even_space = np.linspace(pdf_arr.min(), pdf_arr.max(), num_discrete)
    # logL_discrete = f_reverse(pdf_even_space)

    return interp1d(del_logL_arr, pdf_arr, kind=interp_type, fill_value=0.0, bounds_error=False), logL_discrete, interp1d(del_logL_arr_orig, del_z_arr, kind=interp_type, fill_value=(del_z_arr[0],del_z_arr[-1]), bounds_error=False)

def getBoundsTransPDF(logL_width=2.0, file_name='N501_with_atm.txt', pdflen=100000, fulllen=10000, wav_rest=1215.67, maglen=101, num_discrete=51):
    trans_dat = Table.read(file_name, format='ascii')
    lam, trans = trans_dat['lambda'], trans_dat['transmission']
    transf = interp1d(lam, trans, kind='cubic', bounds_error=False, fill_value=0.0)
    lam_full = np.linspace(lam[0],lam[-1],fulllen)
    trans_max = trans.max()
    trans_min = 10**(-1.0*logL_width) * trans_max
    trans_full = transf(lam_full)
    tfam = np.argmax(trans_full)
    left_ind = np.argmin(abs(trans_full[:tfam+1]-trans_min))
    right_ind = np.argmin(abs(trans_full[tfam:]-trans_min)) + tfam

    # logLs = np.linspace(0.0,logL_width,maglen)
    # delz = np.zeros_like(logLs)
    # for i, logL in enumerate(logLs):
    #     trans_mini = 10**(-1.0*logL) * trans_max
    #     left_indi = np.argmin(abs(trans_full[:tfam+1]-trans_mini))
    #     right_indi = np.argmin(abs(trans_full[tfam:]-trans_mini)) + tfam
    #     delz[i] = (lam[right_indi]-lam[left_indi])/wav_rest
    interp_type = 'linear'
    transpdf, logL_discrete, delzf = getTransPDF(lam_full[left_ind:right_ind], trans_full[left_ind:right_ind], pdflen=pdflen, num_discrete=num_discrete, interp_type=interp_type, wav_rest=wav_rest)
    
    return transpdf, logL_discrete, delzf # (lam_full[right_ind]-lam_full[left_ind])/wav_rest #, interp1d(logLs, delz, bounds_error=False, fill_value=(delz[0],delz[-1]))

class RGINNExt:
    def __init__( self, points, values, method='cubic' ):
        self.interp = RGIScipy(points, values, method=method,
                                              bounds_error=False, fill_value=np.nan)
        self.nearest = RGIScipy(points, values, method='nearest',
                                           bounds_error=False, fill_value=None)
        
    def __call__( self, xi ):
        vals = self.interp( xi )
        idxs = np.isnan( vals )
        if type(xi)==tuple: vals[idxs] = self.nearest((xi[0][idxs], xi[1][idxs]))
        else: vals[idxs] = self.nearest( xi[idxs] )
        return vals

def makeCompFunc(file_name='cosmos_completeness_grid_extrap.pickle'):
    with open(file_name,'rb') as f:
        dat = pickle.load(f)
    mag, dist, comp = dat['Mags'], dat['Dist'], dat['Comp']
    interp_comp = RGINNExt((dist, mag), comp)
    interp_comp_simp = RectBivariateSpline(dist, mag, comp)
    return interp_comp, interp_comp_simp

def cgs2magAB(cgs, wave, dwave):
    Flam = cgs/dwave
    Fnu = Flam*wave**2/c
    return -2.5*np.log10(Fnu)-48.6

def magAB2cgs(mag, wave, dwave):
    Fnu = 10**(-0.4*(mag+48.6))
    Flam = Fnu*c/wave**2
    return Flam * dwave

def lum2cgs(lum, DL):
    return 10**lum / (4.0*np.pi*(3.086e24*DL)**2)

def cgs2lum(cgs, DL):
    return np.log10(cgs * 4.0*np.pi*(3.086e24*DL)**2)

def TrueLumFunc(logL,alpha,logLstar,logphistar):
    ''' Calculate true luminosity function (Schechter form)

    Input
    -----
    logL : float or numpy 1-D array
        Value or array of log luminosities in erg/s
    alpha: float
        Schechther alpha parameter
    logLstar: float
        Schechther log(Lstar) parameter
    logphistar: float
        Schechter log(phistar) parameter

    Returns
    -------
    Phi(logL,z) : Float or 1-D array (same size as logL and/or z)
        Value or array giving luminosity function in Mpc^-3/dex
    '''
    return np.log(10.0) * 10**logphistar * 10**((logL-logLstar)*(alpha+1))*np.exp(-10**(logL-logLstar))

def TrueLumFuncNoPhi(logL,alpha,logLstar):
    return 10**((logL-logLstar)*(alpha+1))*np.exp(-10**(logL-logLstar))

def MakeTLFInterp(logL_range, alpha_range, logLstar_range, num_dim=201):
    logL = np.linspace(logL_range[0],logL_range[1],num_dim)
    alpha = np.linspace(alpha_range[0],alpha_range[1],num_dim)
    logLstar = np.linspace(logLstar_range[0],logLstar_range[1],num_dim)
    Lg, ag, Lsg = np.meshgrid(logL, alpha, logLstar, indexing='ij', sparse=True)
    tlf = TrueLumFuncNoPhi(Lg, ag, Lsg)
    return RGIScipy((logL, alpha, logLstar), tlf, method='cubic', bounds_error=False, fill_value = 0.0)

def Omega(logL,dLz,compfunc,Omega_0,wave,dwave):
    ''' Calculate fractional area of the sky in which galaxies have fluxes large enough so that they can be detected

    Input
    -----
    logL : float or numpy 1-D array
        Value or array of log luminosities in erg/s
    dLz : float
        Luminosity distance (for a given z=z0) in Mpc
    compfunc: interp1d instance
        1-D interpolation function for average completeness vs magnitude
    Omega_0: float
        Effective survey area in square arcseconds

    Returns
    -------
    Omega(logL,z0) : Float or 1-D array (same size as logL)
    '''
    if callable(compfunc): 
        L = 10**logL
        flux_cgs = L/(4.0*np.pi*(3.086e24*dLz)**2)
        mags = cgs2magAB(flux_cgs, wave, dwave)
        comp = compfunc(mags)
    else: 
        comp = compfunc
    return Omega_0/V.sqarcsec * comp

def normalFunc(x,mu,sig):
    return 1.0/(np.sqrt(2.0*np.pi)*sig) * np.exp(-(x-mu)**2/(2.0*sig**2))

class LumFuncMCMC:
    def __init__(self,z,del_red=None,flux=None,flux_e=None,line_name="OIII",
                 line_plot_name=r'[OIII] $\lambda 5007$',lum=None,
                 lum_e=None,Omega_0=43200.,nbins=50,nboot=100,sch_al=-1.6,
                 sch_al_lims=[-3.0,1.0],Lstar=42.5,Lstar_lims=[40.0,45.0],
                 phistar=-3.0,phistar_lims=[-8.0,5.0],Lc=40.0,Lh=46.0,
                 nwalkers=100,nsteps=1000,fix_sch_al=False,
                 min_comp_frac=0.5,diff_rand=True,field_name='COSMOS',
                 interp_comp=None,interp_comp_simp=None,dist_orig=None,dist=None,
                 maglow=26.0,maghigh=19.0,magnum=25,distnum=100,comps=None,
                 size_ln=1001,wav_filt=5015.0,filt_width=73.0,
                 binned_stat_num=50,err_corr=False,wav_rest=1215.67,
                 size_ln_conv=41,size_lprime=51,logL_width=2.0,
                 trans_only=False,norm_only=False,trans_file='N501_with_atm.txt',
                 maxlum=None, minlum=None, transsim=False,
                 corrf=None, corref=None, flux_lim=15.0):
        ''' Initialize LumFuncMCMC class

        Init
        ----
        z : Float
            Redshift of objects in field
        del_red: Float
            Width of redshift bin
        flux : 1-D Numpy array or None Object
            Array of fluxes in 10^-17 erg/cm^2/s
        flux_e : 1-D Numpy array or None Object
            Array of flux errors in 10^-17 erg/cm^2/s
        line_name: string
            Name of line or monochromatic luminosity element
        line_plot_name: (raw) string
            Fancier name of line or luminosity element for putting in plot labels
        lum: numpy array (1 dim) or None Object
            Array of log luminosities in erg/s
        lum_e: numpy array (1 dim) or None Object
            Array of log luminosity errors in erg/s
        Omega_0: Float
            Effective survey area in square arcseconds
        nbins: int
            Number of bins for plotting luminosity function and conducting V_eff method
        nboot: int
            Number of iterations for bootstrap method for determining errors for V_eff method
        sch_al: float
            Schechther alpha parameter
        sch_al_lims: two-element list
            Minimum and maximum values allowed in Schechter alpha prior
        Lstar: float
            Schechther log(Lstar) parameter
        Lstar_lims: two-element list
            Minimum and maximum values allowed in Schechter log(Lstar) prior
        phistar: float
            Schechther log(phistar) parameter
        phistar_lims: two-element list
            Minimum and maximum values allowed in Schechter log(phistar) prior
        Lc, Lh: floats
            Minimum and maximum log luminosity, respectively, for likelihood integral
        nwalkers : int
            The number of walkers for emcee when fitting a model
        nsteps : int
            The number of steps each walker will make when fitting a model
        fix_sch_al: Bool
            Whether or not to fix the alpha parameter of true luminosity function
        min_comp_frac: Float
            Minimum completeness fraction considered
        field_name: String
            Name of field
        interp_comp: Interpolation function
            Interpolation function for completeness
        comp1d: Interpolation function
            1-D interpolation function for completeness averaged over distance from field center
        dist: 1-D Numpy Array
            Array of distances from center of field
        maglow, maghigh: Floats
            Min and max magnitudes for distance-averaging
        comps: 1-D Numpy Array
            Array of completeness values for all objects
        wav_filt: Float
            Central wavelength of filter
        '''
        self.z, self.del_red = z, del_red
        self.min_comp_frac = min_comp_frac
        self.line_name = line_name
        self.line_plot_name = line_plot_name
        self.Lc, self.Lh = Lc, Lh
        self.Omega_0 = Omega_0
        self.Omega_0_sr = Omega_0/V.sqarcsec
        self.nbins, self.nboot = nbins, nboot
        self.sch_al, self.sch_al_lims = sch_al, sch_al_lims
        self.Lstar, self.Lstar_lims = Lstar, Lstar_lims
        self.phistar, self.phistar_lims = phistar, phistar_lims
        self.nwalkers, self.nsteps = nwalkers, nsteps
        self.fix_sch_al = fix_sch_al
        self.all_param_names = ['Lstar','phistar','sch_al']
        self.diff_rand = diff_rand
        self.field_name = field_name
        self.dist, self.dist_orig, self.distnum = dist, dist_orig, distnum
        self.maglow, self.maghigh, self.magnum = maglow, maghigh, magnum
        self.comps, self.size_ln, self.wav_filt = comps, size_ln, wav_filt
        self.size_ln_conv, self.size_lprime = size_ln_conv, size_lprime
        self.filt_width, self.binned_stat_num = filt_width, binned_stat_num
        self.err_corr, self.wav_rest, self.logL_width = err_corr, wav_rest, logL_width
        self.trans_only, self.norm_only = trans_only, norm_only
        self.transf, self.logL_discrete, self.delzf = getBoundsTransPDF(logL_width=self.logL_width,wav_rest=self.wav_rest,num_discrete=self.size_lprime,file_name=trans_file)
        # self.filt_width_eff = self.del_red_eff * self.wav_rest
        self.maxlum, self.minlum, self.transsim = maxlum, minlum, transsim
        self.corrf, self.corref = corrf, corref
        self.flux_lim = flux_lim
        self.logLfuncz, _, self.delz_use, self.ztmax = getRealLumRed(file_name=trans_file, wav_rest=self.wav_rest, delznum=self.size_lprime, logL_width=self.logL_width)
        
        self.setDLdVdz()
        print("Finished DL, dVdz")
        if flux is not None: 
            self.flux = 1.0e-17*flux
            if flux_e is not None:
                self.flux_e = 1.0e-17*flux_e
        else:
            self.lum, self.lum_e = lum, lum_e
            self.getFluxes()
        if lum is None: 
            self.getLumin()
        self.N = self.lum.size
        print("Finished getting fluxes and luminosities")
        self.mags = cgs2magAB(self.flux, self.wav_filt, self.filt_width) # For the completeness
        if interp_comp is None: self.interp_comp, self.interp_comp_simp = makeCompFunc()
        else: self.interp_comp, self.interp_comp_simp = interp_comp, interp_comp_simp
        if self.comps is None: self.comps = self.interp_comp_simp.ev(self.dist, self.mags)
        print("Got completeness")
        self.getalls()
        
        if not self.transsim:
            self.get1DComp()
            logL_min = self.logL_norm.min()
            logL_max = self.logL_norm.max() + self.logL_discrete.max()
            self.tlf_interp = MakeTLFInterp([logL_min, logL_max], self.sch_al_lims, self.Lstar_lims)
        else:
            self.Omega_arr = Omega(self.lum,self.DL,self.comps,self.Omega_0,self.wav_filt,self.filt_width)
            print("Finished getting Omega array")
        self.setup_logging()

    def getalls(self):
        alls_file_name = f'Likes_alls_field{self.field_name}_z{self.z}_mcf{self.min_comp_frac}_fl{self.flux_lim}.pickle'
        with open(alls_file_name, 'rb') as f:
            alls_output = pickle.load(f)
        als, lss, likes = alls_output['Alphas'], alls_output['Lstars'], alls_output['likelihoods']
        self.likeallsf = RectBivariateSpline(als, lss, likes)

    def get1DComp(self):
        ''' Get LAE-point-averaged estimates of the 1-D completeness function (of magnitude) '''
        print("Setting the computational arrays")
        maggrid = np.linspace(self.maghigh, self.maglow, self.magnum)
        distgrid = np.sort(np.random.choice(self.dist_orig, size=self.distnum))
        distg, magg = np.meshgrid(distgrid, maggrid, indexing='ij')
        comps = self.interp_comp_simp.ev(distg.ravel(), magg.ravel())
        comps = comps.reshape(self.distnum, self.magnum)
        roots = np.zeros(self.distnum)
        for i in range(self.distnum):
            func = interp1d(maggrid, comps[i], bounds_error=False, fill_value=(comps[i][0], comps[i][-1]))
            roots[i] = fsolve(lambda x: func(x)-self.min_comp_frac, [25.0])[0]
        fluxes = magAB2cgs(roots, self.wav_filt, self.filt_width)
        minlums = np.log10(4.0*np.pi*(self.DL*3.086e24)**2 * fluxes)
        self.minflux, self.minlum = np.average(fluxes), np.average(minlums)
        self.minlum_conv = self.minlum - 3.0*self.lum_err_func(self.minlum)
        comp_avg_dist = np.average(comps,axis=0)
        self.comp1df = interp1d(maggrid, comp_avg_dist, bounds_error=False, fill_value=(comp_avg_dist[0], comp_avg_dist[-1]))
        self.comps1d = self.comp1df(self.mags)
        self.Omega_arr = Omega(self.lum,self.DL,self.comps,self.Omega_0,self.wav_filt,self.filt_width)
        self.logL = np.linspace(self.minlum,self.Lh,self.size_ln)
        self.Omega_gen = Omega(self.logL,self.DL,self.comp1df,self.Omega_0,self.wav_filt,self.filt_width)

        ########### Things for new version of transmission convolution ###########
        cgs = lum2cgs(self.logL, self.DL)
        mags = cgs2magAB(cgs, self.wav_filt, self.filt_width)
        self.comps_full = np.average(self.interp_comp_simp.ev(self.dist[:,None], mags), axis=0)
        self.trans_mult = 10**(-self.logLfuncz(self.zarr)) * self.dVdzs
        self.ptransmult = self.Omega_0_sr * self.comps_full[:,None] * self.trans_mult
        # self.ptransmult = self.Omega_0_sr * self.comps_full * np.average(self.dVdzs)

        ########### For convolution part ###########
        # self.logL_conv = np.linspace(self.minlum_conv,self.Lh,self.size_ln_conv)
        # self.logL_conv_all = np.zeros((self.size_ln_conv,self.size_lprime))
        # for i in range(self.size_ln_conv):
        #     self.logL_conv_all[i] = np.linspace(self.logL_conv[i],self.logL_conv[i]+self.logL_width,self.size_lprime)
        # L_all = 10**self.logL_conv_all
        # flux_cgs = L_all/(4.0*np.pi*(3.086e24*self.DL)**2)
        # mags = cgs2magAB(flux_cgs, self.wav_filt, self.filt_width)
        # self.comp_conv = self.comp1df(mags)
        # self.trans_conv = self.transf(self.logL_discrete)
        # self.norm_vals = normalFunc(self.logL_conv[None],self.lum[:,None],self.lum_e[:,None])

        ####### Just transmission convolution #######
        self.trans_conv = self.transf(self.logL_discrete)
        self.logL_trans_lnpart = self.lum[:,None] + self.logL_discrete
        self.logL_trans_integ = self.logL[:,None] + self.logL_discrete
        
        L_all = 10**self.logL_trans_lnpart
        flux_cgs = L_all/(4.0*np.pi*(3.086e24*self.DL)**2)
        mags = cgs2magAB(flux_cgs, self.wav_filt, self.filt_width)
        self.comps_trans_lnpart = self.comp1df(mags)

        L_all = 10**self.logL_trans_integ
        flux_cgs = L_all/(4.0*np.pi*(3.086e24*self.DL)**2)
        mags = cgs2magAB(flux_cgs, self.wav_filt, self.filt_width)
        self.comps_trans_integ = self.comp1df(mags)

        self.delz_trans = self.delzf(self.logL_trans_integ-self.minlum)
        self.not_tlf = self.comps_trans_integ * self.delz_trans * self.trans_conv

        ###### Just normal convolution #####
        self.logL_norm = np.zeros((self.lum.size,self.size_ln_conv))
        for i in range(self.lum.size):
            self.logL_norm[i] = self.lum[i] + np.linspace(-3.0*self.lum_e[i],3.0*self.lum_e[i],self.size_ln_conv)
        self.norm_vals_norm = normalFunc(self.logL_norm,self.lum[:,None],self.lum_e[:,None])
        L_all = 10**self.logL_norm
        flux_cgs = L_all/(4.0*np.pi*(3.086e24*self.DL)**2)
        mags = cgs2magAB(flux_cgs, self.wav_filt, self.filt_width)
        self.comps_norm = self.comp1df(mags)

        ###### New combination of everything that is centered for normal distribution around the mean #####
        self.logL_conv = self.logL_norm[:,:,None] + self.logL_discrete
        L_all = 10**self.logL_conv
        flux_cgs = L_all/(4.0*np.pi*(3.086e24*self.DL)**2)
        mags = cgs2magAB(flux_cgs, self.wav_filt, self.filt_width)
        self.comps_conv = self.comp1df(mags)
        
    def setDLdVdz(self):
        ''' Create 1-D interpolated functions for luminosity distance (cm) and comoving volume differential (Mpc^3); also get function for minimum luminosity considered '''
        print("Setting DL and dVdz")
        self.DL = V.cosmo.luminosity_distance(self.z).value
        self.dVdz = V.cosmo.differential_comoving_volume(self.z).value
        # if self.err_corr or self.trans_only: self.volume = self.dVdz * self.del_red_eff
        self.zarr = np.linspace(self.ztmax-0.55*self.delz_use, self.ztmax+0.55*self.delz_use, self.size_lprime)
        self.dVdzs = V.cosmo.differential_comoving_volume(self.zarr).value
        self.volume = self.dVdz * self.del_red # Actual total volume per steradian of the survey (redshift integral separate from luminosity function integral)

    def getLumin(self):
        ''' Set the sample log luminosities (and error if flux errors available)
            based on given flux values and luminosity distance values
        '''
        if self.flux_e is not None: 
            ulum = unumpy.log10(4.0*np.pi*(self.DL*3.086e24)**2 * unumpy.uarray(self.flux,self.flux_e))
            self.lum, self.lum_e = unumpy.nominal_values(ulum), unumpy.std_devs(ulum)
            self.lum_bin_edges = np.percentile(self.lum,np.linspace(0.,100.,self.binned_stat_num+1))
            self.lum_err_bins, _, _ = binned_statistic(self.lum, self.lum_e, statistic='median',bins=self.lum_bin_edges)
            self.lum_bin_mid = np.array([(self.lum_bin_edges[i]+self.lum_bin_edges[i+1])/2.0 for i in range(self.binned_stat_num)])
            self.lum_err_func = interp1d(self.lum_bin_mid, self.lum_err_bins, bounds_error=False, fill_value=(self.lum_err_bins[0],self.lum_err_bins[-1]))
        else:
            self.lum = np.log10(4.0*np.pi*(self.DL*3.086e24)**2 * self.flux)
            self.lum_e = None
            self.lum_bin_edges, self.lum_err_bins, self.lum_bin_mid, self.lum_err_func = None, None, None, None

    def getFluxes(self):
        ''' Set sample fluxes based on luminosities if not available '''
        if self.lum_e is not None:
            ulum = 10**unumpy.uarray(self.lum,self.lum_e)
            uflux = ulum/(4.0*np.pi*(self.DL*3.086e24)**2)
            self.flux, self.flux_e = unumpy.nominal_values(uflux), unumpy.std_devs(uflux)
        else:
            self.flux = 10**self.lum/(4.0*np.pi*(self.DL*3.086e24)**2)
            self.flux_e = None

    def calclikeLsal(self, alnum=50, lsnum=50):
        als = np.linspace(self.sch_al_lims[0], self.sch_al_lims[1], alnum)
        lss = np.linspace(self.Lstar_lims[0], self.Lstar_lims[1], lsnum)
        compgrid = np.zeros((len(self.dist), *self.logL_trans_integ.shape))
        L_all = 10**self.logL_trans_integ.ravel()
        flux_cgs = L_all/(4.0*np.pi*(3.086e24*self.DL)**2)
        mags = cgs2magAB(flux_cgs, self.wav_filt, self.filt_width)
        for i, dist in enumerate(self.dist):
            if i%400==0: print(f"Got to i={i} for calculating comps grid")
            comps = self.interp_comp_simp.ev(dist, mags)
            compgrid[i] = comps.reshape(*self.logL_trans_integ.shape)
        compG = compgrid * self.trans_conv[None,None]

        ldo = len(self.dist)
        likes = np.zeros((alnum, lsnum))
        for i in range(alnum):
            print(f"Got to i={i} in main al ls loop")
            for j in range(lsnum):
                tlf = TrueLumFuncNoPhi(self.logL_trans_integ, als[i], lss[j])
                integ = tlf[None] * compG
                phiobs = trapz(integ, self.logL_trans_integ[None], axis=2)
                phiobs /= trapz(phiobs, self.logL, axis=1)[:,None]
                likeij = np.zeros(ldo)
                for k in range(ldo):
                    likeij[k] = np.interp(self.lum[k], self.logL, phiobs[k])
                likes[i,j] = np.log(likeij).sum()
        return als, lss, likes

    def setup_logging(self):
        '''Setup Logging for MCSED

        Builds
        -------
        self.log : class
            self.log.info() is for general print and self.log.error() is
            for raise cases
        '''
        self.log = logging.getLogger('lumfuncmcmc')
        if not len(self.log.handlers):
            # Set format for logger
            fmt = '[%(levelname)s - %(asctime)s] %(message)s'
            fmt = logging.Formatter(fmt)
            # Set level of logging
            level = logging.INFO
            # Set handler for logging
            handler = logging.StreamHandler()
            handler.setFormatter(fmt)
            handler.setLevel(level)
            # Build log with name, mcsed
            self.log = logging.getLogger('lumfuncmcmc')
            self.log.setLevel(logging.DEBUG)
            self.log.addHandler(handler)

    def set_parameters_from_list(self,input_list):
        ''' For a given set of model parameters, set the needed class variables.

        Input
        -----
        theta : list
            list of input parameters for Schechter Fit'''
        self.Lstar = input_list[0]
        self.phistar = input_list[1]
        if self.fix_sch_al: pass
        else: self.sch_al = input_list[2]

    def lnprior(self):
        ''' Simple, uniform prior for input variables

        Returns
        -------
        0.0 if all parameters are in bounds, -np.inf if any are out of bounds
        '''
        flag = 1.0
        for param in self.all_param_names:
            flag *= ((getattr(self,param) >= getattr(self,param+'_lims')[0]) *
                     (getattr(self,param) <= getattr(self,param+'_lims')[1]))
        if not flag: 
            return -np.inf
        else: 
            return 0.0

    def lnlike(self):
        ''' Calculate the log likelihood and return the value and stellar mass
        of the model as well as other derived parameters

        Returns
        -------
        log likelihood (float)
            The log likelihood includes a ln term and an integral term (based on Poisson statistics). '''
        lnpart = np.log(TrueLumFunc(self.lum,self.sch_al,self.Lstar,self.phistar)*self.comps).sum()
        integ = TrueLumFunc(self.logL,self.sch_al,self.Lstar,self.phistar) * self.Omega_gen
        fullint = self.volume * trapz(integ,self.logL)
        return lnpart - fullint
    
    def lnlike_conv(self):
        tlf = np.log(10.0) * 10**self.phistar * TrueLumFuncNoPhi(self.logL_conv,self.sch_al,self.Lstar)
        not_norm = tlf*self.comps_conv*self.trans_conv
        trapz_inner = trapz(not_norm,self.logL_conv)
        numer = trapz(trapz_inner*self.norm_vals_norm, self.logL_norm)
        # denom = trapz(trapz_inner, self.logL_conv)
        lnpart = np.log(numer).sum()
        # fullint = self.Omega_0_sr * self.volume * denom
        integ = np.log(10.0) * 10**self.phistar * TrueLumFuncNoPhi(self.logL_trans_integ,self.sch_al,self.Lstar) * self.not_tlf
        fullint = self.Omega_0_sr * self.dVdz * trapz(trapz(integ,self.logL_trans_integ),self.logL)
        return lnpart - fullint

    def lnlike_trans(self):
        tlf = np.log(10.0) * 10**self.phistar * TrueLumFuncNoPhi(self.logL_trans_lnpart,self.sch_al,self.Lstar)
        lnpart = np.log(trapz(tlf*self.comps_trans_lnpart*self.trans_conv,self.logL_trans_lnpart)).sum()
        integ = np.log(10.0) * 10**self.phistar * TrueLumFuncNoPhi(self.logL_trans_integ,self.sch_al,self.Lstar) * self.not_tlf
        fullint = self.Omega_0_sr * self.dVdz * trapz(trapz(integ,self.logL_trans_integ),self.logL)
        return lnpart - fullint

    def lnlike_trans_new(self):
        # time1 = time()
        like_alls = self.likeallsf.ev(self.sch_al, self.Lstar)
        # time2 = time()
        tlf = TrueLumFunc(self.logL, self.sch_al, self.Lstar, self.phistar)
        integ = tlf[:,None] * self.ptransmult
        # integ = tlf * self.ptransmult
        # time3 = time()
        num = trapz(trapz(integ, self.zarr), self.logL)
        # num = self.del_red * trapz(integ, self.logL)
        # time4 = time()
        # like_phi = np.log(self.rv.pmf(np.average(nums).astype(int)))
        like_phi = poisson_lnpmf(int(num), self.N)
        # time5 = time()
        # print("Times:", time2-time1, time3-time2, time4-time3, time5-time4)
        return like_alls + like_phi

    def lnlike_norm(self):
        tlf = np.log(10.0) * 10**self.phistar * TrueLumFuncNoPhi(self.logL_norm,self.sch_al,self.Lstar)
        lnpart = np.log(trapz(tlf*self.comps_norm*self.norm_vals_norm,self.logL_norm)).sum()
        integ = np.log(10.0) * 10**self.phistar * TrueLumFuncNoPhi(self.logL,self.sch_al,self.Lstar) * self.Omega_gen
        fullint = self.volume * trapz(integ,self.logL)
        return lnpart - fullint

    def lnprob(self, theta):
        ''' Calculate the log probability 

        Returns
        -------
        log prior + log likelihood, (float)
            The log probability is just the sum of the logs of the prior and likelihood. '''
        self.set_parameters_from_list(theta)
        lp = self.lnprior()
        if np.isfinite(lp):
            lnl = self.lnlike()
            return lnl+lp
        else:
            return -np.inf
        
    def lnprob_conv(self, theta):
        self.set_parameters_from_list(theta)
        lp = self.lnprior()
        if np.isfinite(lp):
            lnl = self.lnlike_conv()
            return lnl+lp
        else:
            return -np.inf
        
    def lnprob_trans(self, theta):
        self.set_parameters_from_list(theta)
        lp = self.lnprior()
        if np.isfinite(lp):
            lnl = self.lnlike_trans_new()
            return lnl+lp
        else:
            return -np.inf
        
    def lnprob_norm(self, theta):
        self.set_parameters_from_list(theta)
        lp = self.lnprior()
        if np.isfinite(lp):
            lnl = self.lnlike_norm()
            return lnl+lp
        else:
            return -np.inf

    def get_init_walker_values(self, num=None):
        ''' Before running emcee, this function generates starting points
        for each walker in the MCMC process.

        Returns
        -------
        pos : np.array (2 dim)
            Two dimensional array with Nwalker x Ndim values
        '''
        # theta = [self.sch_al, self.Lstar, self.phistar]
        theta_lims = np.vstack((self.Lstar_lims,self.phistar_lims))
        if not self.fix_sch_al: theta_lims = np.vstack((theta_lims,self.sch_al_lims))
        if num is None:
            num = self.nwalkers
        if self.diff_rand: pos_part1 = np.random.rand(num,len(theta_lims))
        else: pos_part1 = np.random.rand(num)[:,np.newaxis]
        pos = (pos_part1 * (theta_lims[:, 1]-theta_lims[:, 0]) + theta_lims[:, 0])
        return pos

    def get_param_names(self):
        ''' Grab the names of the parameters for plotting

        Returns
        -------
        names : list
            list of all parameter names
        '''
        names = [r'$\log L_*$',r'$\log \phi_*$']
        if not self.fix_sch_al: names += [r'$\alpha$']
        return names

    def get_params(self):
        ''' Grab the the parameters in each class

        Returns
        -------
        vals : list
            list of all parameter values
        '''
        vals = [self.Lstar,self.phistar]
        if not self.fix_sch_al: vals += [self.sch_al]
        self.nfreeparams = len(vals)
        return vals
    
    def fit_model(self):
        ''' Using emcee to find parameter estimations for given set of
        data measurements and errors
        '''
        self.log.info('Fitting Schechter model to true luminosity function using emcee')
        pos = self.get_init_walker_values()
        ndim = pos.shape[1]
        if self.err_corr: func = 'lnprob_conv'
        else: func = 'lnprob'
        if self.trans_only: func = 'lnprob_trans'
        if self.norm_only: func = 'lnprob_norm'
        sampler = emcee.EnsembleSampler(self.nwalkers, ndim, getattr(self,func))
        # Do real run
        start = time()
        sampler.run_mcmc(pos, self.nsteps, rstate0=np.random.get_state())
        end = time()
        elapsed = end - start
        self.log.info("Total time taken: %0.2f s" % elapsed)
        self.log.info("Time taken per step per walker: %0.2f ms" %
                        (elapsed / (self.nsteps) * 1000. /
                       self.nwalkers))
        # Calculate how long the run should last
        tau = np.max(sampler.acor)
        burnin_step = int(tau*3)
        if burnin_step>self.nsteps//2: burnin_step = self.nsteps//2
        self.log.info("Mean acceptance fraction: %0.2f" %
                      (np.mean(sampler.acceptance_fraction)))
        self.log.info("AutoCorrelation Steps: %i, Number of Burn-in Steps: %i"
                      % (np.round(tau), burnin_step))
        new_chain = np.zeros((self.nwalkers, self.nsteps, ndim+1))
        new_chain[:, :, :-1] = sampler.chain
        self.chain = sampler.chain
        new_chain[:, :, -1] = sampler.lnprobability
        self.samples = new_chain[:, burnin_step:, :].reshape((-1, ndim+1))
        self.log.info("Shape of self.samples")
        self.log.info(self.samples.shape)
        self.log.info("Median lnprob: %.5f; Max lnprob: %.5f"%(np.median(sampler.lnprobability), np.amax(sampler.lnprobability)))

    def VeffLF(self, varying=False):
        ''' Use V_Eff method to calculate properly weighted measured luminosity function '''
        print("Ready to calculate V effective method")
        if varying: self.phifunc = 1.0/(self.dVdz * self.delzf(self.lum - self.minlum) * self.Omega_arr)
        else: self.phifunc = 1.0/(self.volume * self.Omega_arr)
        self.Lavg, self.lfbinorig, self.var = V.getBootErrLog(self.lum,self.phifunc,self.nboot,self.nbins,Lmin=self.minlum, Lmax=self.maxlum)
        if self.corrf is not None:
            ucorr_orig = unumpy.uarray(self.corrf(self.Lavg), self.corref(self.Lavg))
            ulf = unumpy.uarray(self.lfbinorig, np.sqrt(self.var))
            ulf_new = 10 ** (unumpy.log10(ulf) + ucorr_orig)
            self.lfbinorig_orig, self.var_orig = self.lfbinorig*1.0, self.var*1.0 #Want to show original values
            self.lfbinorig = unumpy.nominal_values(ulf_new)
            self.var = unumpy.std_devs(ulf_new) ** 2

    def set_median_fit(self,rndsamples=200,lnprobcut=7.5):
        '''
        set attributes
        median modeled ("observed") luminosity function for rndsamples random samples
        This function is applied only when a triangle plot is not desired

        Input
        -----
        rndsamples : int
            number of random samples over which to compute medians
        lnprobcut : float
            Some of the emcee chains include outliers.  This value serves as
            a cut in log probability space with respect to the maximum
            probability.  For reference, a Gaussian 1-sigma is 2.5 in log prob
            space.

        Creates
        -------
        self.medianLF : list (1d)
            median fitted ("observed") luminosity function
        '''
        nsamples = []
        while len(nsamples)<len(self.samples)//4: 
            chi2sel = (self.samples[:, -1] >
                    (np.max(self.samples[:, -1], axis=0) - lnprobcut))
            nsamples = self.samples[chi2sel, :]
            lnprobcut *= 2.0
        # nsamples = self.samples
        self.log.info("Shape of nsamples (with a lnprobcut applied)")
        self.log.info(nsamples.shape)
        lf = []
        for i in np.arange(rndsamples):
            ind = np.random.randint(0, nsamples.shape[0])
            self.set_parameters_from_list(nsamples[ind, :])
            modlum = TrueLumFunc(self.lum,self.sch_al,self.Lstar,self.phistar)
            lf.append(modlum)
        self.medianLF = np.median(np.array(lf), axis=0)
        self.VeffLF()

    def plotPracLumFunc(self, tlft, phiobs):
        fig, ax = plt.subplots()
        self.add_LumFunc_plot(ax)
        ax.plot(self.logL, tlft, 'b-', label='True Luminosity Function')
        ax.plot(self.logL, phiobs, 'r-', label='Observed Luminosity Function')
        ax.set_ylim(bottom=1.0e-10)
        plt.show()

    def add_LumFunc_plot(self,ax1):
        """ Set up the plot for the luminosity function """
        ax1.set_yscale('log')
        ax1.set_xlabel(r"$\log$ L (erg s$^{-1}$)")
        ax1.set_ylabel(r"$\phi_{\rm{true}}$ (Mpc$^{-3}$ dex$^{-1}$)")
        ax1.minorticks_on()

    def VeffPlotCommands(self, ax):
        markersize = self.nfreeparams * 1
        cond_veff = self.Lavg >= self.minlum
        ax.errorbar(self.Lavg[cond_veff],self.lfbinorig[cond_veff],yerr=np.sqrt(self.var[cond_veff]),fmt='b^', label='Measured LF', markersize=markersize)
        ax.errorbar(self.Lavg[~cond_veff],self.lfbinorig[~cond_veff],yerr=np.sqrt(self.var[~cond_veff]),fmt='b^',alpha=0.2, label='', markersize=markersize)
        if self.corrf is not None:
            ax.errorbar(self.Lavg[cond_veff],self.lfbinorig_orig[cond_veff],yerr=np.sqrt(self.var_orig[cond_veff]),fmt='rs', label='LF without Transmission Correction', markersize=markersize)
            ax.errorbar(self.Lavg[~cond_veff],self.lfbinorig_orig[~cond_veff],yerr=np.sqrt(self.var_orig[~cond_veff]),fmt='rs',alpha=0.2, label='', markersize=markersize)
            ax.legend(loc='best', frameon=False, fontsize='xx-small')

    def plotVeff(self, outname, imgtype='png', varying=False):
        self.VeffLF(varying=varying)
        fig, ax = plt.subplots()
        self.add_LumFunc_plot(ax)
        self.VeffPlotCommands(ax)
        fig.savefig(outname+'.'+imgtype, bbox_inches='tight', dpi=300)

    def plotVeffEnv(self, Lavgs, lfbinorigs, vars, minlums, labels, outname, imgtype='png', fmt_seq=['b^', 'r*', 'ko', 'mx', 'cs', 'gh', 'y+'], lflums=None, lfs=None, linestyle_seq=['-', '--', '-.', ':', '-', '--', '-.', ':']):
        fig, ax = plt.subplots()
        self.add_LumFunc_plot(ax)
        ilist = np.arange(len(Lavgs))
        for i, Lavg, lfbinorig, var, minlum in zip(ilist, Lavgs, lfbinorigs, vars, minlums):
            col = fmt_seq[i][0]
            cond_veff = Lavg >= minlum
            ax.plot(Lavg[cond_veff],lfbinorig[cond_veff],fmt_seq[i],linestyle='none',label=labels[i])
            ax.errorbar(Lavg[cond_veff],lfbinorig[cond_veff],yerr=np.sqrt(var[cond_veff]),fmt='none',ecolor=col,label='',alpha=0.1)
            ax.errorbar(Lavg[~cond_veff],lfbinorig[~cond_veff],yerr=np.sqrt(var[~cond_veff]),fmt=fmt_seq[i],alpha=0.1,label='')
            if lfs is not None: 
                lfli, lfi = lflums[i], lfs[i]
                indsort = np.argsort(lfli)
                ax.plot(lfli[indsort], lfi[indsort], col+linestyle_seq[i], label='')
        ax.legend(loc='best', frameon=False, fontsize='small')
        fig.savefig(outname+'.'+imgtype, bbox_inches='tight', dpi=300)

    def add_subplots(self,ax1,nsamples,rndsamples=200):
        ''' Add Subplots to Triangle plot below '''
        lf = []
        indsort = np.argsort(self.lum)
        lstars = np.zeros(rndsamples)
        for i in np.arange(rndsamples):
            ind = np.random.randint(0, nsamples.shape[0])
            self.set_parameters_from_list(nsamples[ind, :])
            lstars[i] = self.Lstar
            modlum = TrueLumFunc(self.lum,self.sch_al,self.Lstar,self.phistar)
            lf.append(modlum)
            ax1.plot(self.lum[indsort],modlum[indsort],color='r',linestyle='solid',alpha=0.1, label='')
        self.medianLF = np.median(np.array(lf), axis=0)
        self.VeffLF()
        if self.corrf is not None: label = 'Best-fit'
        else: label = ''
        ax1.plot(self.lum[indsort],self.medianLF[indsort],color='dimgray',linestyle='solid',label=label)
        self.VeffPlotCommands(ax1)
        xmin = self.minlum
        xmax = min(max(self.lum),np.median(lstars)+1.0)
        ax1.set_xlim(left=xmin,right=xmax)
        cond = np.logical_and(self.lum<=xmax,self.lum>=xmin)
        ax1.set_ylim(bottom=np.percentile(self.medianLF[cond],0),top=np.percentile(self.medianLF[cond],100))
        
    def triangle_plot(self, outname, lnprobcut=7.5, imgtype='png'):
        ''' Make a triangle corner plot for samples from fit

        Input
        -----
        outname : string
            The triangle plot will be saved as "triangle_{outname}.png"
        lnprobcut : float
            Some of the emcee chains include outliers.  This value serves as
            a cut in log probability space with respect to the maximum
            probability.  For reference, a Gaussian 1-sigma is 2.5 in log prob
            space.
        imgtype : string
            The file extension of the output plot
        '''
        # Make selection for three sigma sample
        nsamples = []
        while len(nsamples)<len(self.samples)//4: 
            chi2sel = (self.samples[:, -1] >
                    (np.max(self.samples[:, -1], axis=0) - lnprobcut))
            nsamples = self.samples[chi2sel, :]
            lnprobcut *= 2.0
        # nsamples = self.samples
        self.log.info("Shape of nsamples (with a lnprobcut applied)")
        self.log.info(nsamples.shape)
        names = self.get_param_names()
        indarr = np.arange(len(nsamples[0]))
        fsgrad = 11+int(round(0.75*len(indarr)))
        percentilerange = [.95] * len(names)
        fig = corner.corner(nsamples[:, :-1], labels=names,
                            range=percentilerange,
                            label_kwargs={"fontsize": fsgrad}, show_titles=True,
                            title_kwargs={"fontsize": fsgrad-2},
                            quantiles=[0.16, 0.5, 0.84], bins=30)
        w = fig.get_figwidth()
        if len(indarr)>=4: 
            figw = w-(len(indarr)-13)*0.025*w
            poss = [0.50-0.008*(len(indarr)-4), 0.78-0.001*(len(indarr)-4), 0.48+0.008*(len(indarr)-4), 0.19+0.001*(len(indarr)-4)]
        else: 
            figw = w
            poss = [0.67,0.75,0.32,0.23]
        fig.set_figwidth(figw)
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.set_position(poss)
        self.add_LumFunc_plot(ax1)
        self.add_subplots(ax1,nsamples)
        fig.savefig("%s.%s" % (outname,imgtype), dpi=200)
        plt.close(fig)

    def add_fitinfo_to_table(self, percentiles, start_value=1, lnprobcut=7.5):
        ''' Assumes that "Ln Prob" is the last column in self.samples'''
        nsamples = []
        while len(nsamples)<len(self.samples)//4: 
            chi2sel = (self.samples[:, -1] >
                    (np.max(self.samples[:, -1], axis=0) - lnprobcut))
            nsamples = self.samples[chi2sel, :-1]
            lnprobcut *= 2.0
        # nsamples = self.samples[:,:-1]
        self.log.info("Number of table entries: %d"%(len(self.table[0])))
        self.log.info("Len(percentiles): %d; len(other axis): %d"%(len(percentiles), len(np.percentile(nsamples,percentiles[0],axis=0))))
        n = len(percentiles)
        for i, per in enumerate(percentiles):
            for j, v in enumerate(np.percentile(nsamples, per, axis=0)):
                self.table[-1][(i + start_value + j*n)] = v
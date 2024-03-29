import numpy as np 
import logging
import emcee
from uncertainties import unumpy, ufloat
import matplotlib
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.integrate import trapz
import time
import pdb
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner
import VmaxLumFunc as V
from scipy.optimize import fsolve
import pdb
import seaborn as sns
sns.set_context("paper",font_scale=1.3) # options include: talk, poster, paper
sns.set_style("ticks")
sns.set_style({"xtick.direction": "in","ytick.direction": "in",
               "xtick.top":True, "ytick.right":True,
               "xtick.major.size":12, "xtick.minor.size":4,
               "ytick.major.size":12, "ytick.minor.size":4,
               })

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
        Schechther log(phistar) parameter

    Returns
    -------
    Phi(logL,z) : Float or 1-D array (same size as logL and/or z)
        Value or array giving luminosity function in Mpc^-3/dex
    '''
    return np.log(10.0) * 10**logphistar * 10**((logL-logLstar)*(alpha+1))*np.exp(-10**(logL-logLstar))

# 
def Omega(logL,z,dLzfunc,Omega_0,Flim,alpha,fcmin=0.1):
    ''' Calculate fractional area of the sky in which galaxies have fluxes large enough so that they can be detected

    Input
    -----
    logL : float or numpy 1-D array
        Value or array of log luminosities in erg/s
    z : float or numpy 1-D array
        Value or array of redshifts; logL and z cannot be different-sized arrays
    dLzfunc : interp1d function
        1-D interpolation function for luminosity distance in Mpc
    Omega_0: float
        Effective survey area in square arcseconds
    Flim: float
        50% completeness flux value
    alpha: float
        Completeness-related slope

    Returns
    -------
    Omega(logL,z) : Float or 1-D array (same size as logL and/or z)
    '''
    L = 10**logL
    return Omega_0/V.sqarcsec * V.fleming(L/(4.0*np.pi*(3.086e24*dLzfunc(z))**2),Flim,alpha,fcmin)

class LumFuncMCMC:
    def __init__(self,z,flux=None,flux_e=None,Flim=[2.35,3.12,2.20,2.86,2.85],Flim_lims=[1.0,6.0],
                 alpha=3.5, alpha_lims=[1.0,6.0],line_name="OIII",
                 line_plot_name=r'[OIII] $\lambda 5007$',lum=None,lum_e=None,Omega_0=[100.0,100.0,100.0,100.0,100.0],nbins=50,
                 nboot=100,sch_al=-1.6, sch_al_lims=[-3.0,1.0],Lstar=42.5,Lstar_lims=[40.0,45.0],
                 phistar=-3.0,phistar_lims=[-8.0,5.0],Lc=40.0,Lh=46.0,nwalkers=100,nsteps=1000,
                 fix_sch_al=False,fcmin=0.1,fix_comp=False,min_comp_frac=0.5,
                 field_names=None,field_ind=None,diff_rand=True):
        ''' Initialize LumFuncMCMC class

        Init
        ----
        z : List of numpy arrays (1 dim)
            List of arrays of redshifts for sample (one for each field)
        flux : list of numpy arrays (1 dim) or None Object
            List of arrays of fluxes in 10^-17 erg/cm^2/s
        flux_e : list of numpy arrays (1 dim) or None Object
            List of arrays of flux errors in 10^-17 erg/cm^2/s
        Flim: list of floats (multiple of 1e-17 erg/cm^2/s)
            50% completeness flux parameters for each field (AEGIS, COSMOS, GOODSN, GOODSS, UDS)
        Flim_lims: two-element list
            Minimum and maximum values allowed in Flim prior (same for all fields)
        alpha: float
            Completeness-related slope
        line_name: string
            Name of line or monochromatic luminosity element
        line_plot_name: (raw) string
            Fancier name of line or luminosity element for putting in plot labels
        lum: numpy array (1 dim) or None Object
            Array of log luminosities in erg/s
        lum_e: numpy array (1 dim) or None Object
            Array of log luminosity errors in erg/s
        Omega_0: List of floats
            Effective survey area in square arcseconds for each field
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
        root: List of floats
            Minimum flux cutoff for each field based on the completeness curve parameters and desired minimum completeness
        fix_sch_al: Bool
            Whether or not to fix the alpha parameter of true luminosity function
        fcmin: Float
            Completeness fraction below which the modification to the Fleming curve becomes important
        fix_comp: Bool
            Whether or not to fix the completeness parameters
        min_comp_frac: Float
            Minimum completeness fraction considered
        field_names: 1-D Numpy array
            List of fields used in data
        field_ind: 1-D Numpy array
            Indices that convert the unique field names to the full list of fluxes
        '''
        self.z = np.concatenate(z)
        self.zmin, self.zmax = min(self.z), max(self.z)
        self.fcmin, self.min_comp_frac = fcmin, min_comp_frac
        self.Flim, self.Flim_lims = Flim, Flim_lims
        self.fields, self.nfields = field_names, len(self.Flim)
        self.field_ind = field_ind
        self.alpha, self.alpha_lims = alpha, alpha_lims
        self.line_name = line_name
        self.line_plot_name = line_plot_name
        self.Lc, self.Lh = Lc, Lh
        self.Omega_0 = Omega_0
        self.nbins, self.nboot = nbins, nboot
        self.sch_al, self.sch_al_lims = sch_al, sch_al_lims
        self.Lstar, self.Lstar_lims = Lstar, Lstar_lims
        self.phistar, self.phistar_lims = phistar, phistar_lims
        self.nwalkers, self.nsteps = nwalkers, nsteps
        self.fix_sch_al, self.fix_comp = fix_sch_al, fix_comp
        self.all_param_names = ['Lstar','phistar','sch_al','Flim','alpha']
        self.diff_rand = diff_rand
        self.defineFlimOmArr()
        self.getRoot()
        self.setDLdVdz()
        if flux is not None: 
            self.flux = 1.0e-17*np.concatenate(flux)
            if flux_e is not None:
                self.flux_e = 1.0e-17*np.concatenate(flux_e)
        else:
            self.lum, self.lum_e = np.concatenate(lum), np.concatenate(lum_e)
            self.getFluxes()
        if lum is None: 
            self.getLumin()
        self.setOmegaLz()
        self.roots_ln = self.rootsf.ev(self.Flim,self.alpha)
        self.allind = np.arange(len(self.lum))
        self.setlnsimple()
        self.setup_logging()

    def setDLdVdz(self):
        ''' Create 1-D interpolated functions for luminosity distance (cm) and comoving volume differential (Mpc^3); also get function for minimum luminosity considered '''
        # self.DL = np.zeros(len(self.z))
        zint = np.linspace(0.95*self.zmin,1.05*self.zmax,len(self.z))
        # dVdzarr, DLarr = np.zeros(len(zint)), np.zeros(len(zint))
        self.minlumf = []
        self.DL = V.cosmo.luminosity_distance(self.z).value
        DLarr = V.cosmo.luminosity_distance(zint).value
        dVdzarr = V.cosmo.differential_comoving_volume(zint).value
        # for i,zi in enumerate(self.z):
        #     # self.DL[i] = V.dLz(zi) # In Mpc
        #     self.DL[i] = V.cosmo.luminosity_distance(zi)
        #     # DLarr[i] = V.dLz(zint[i])
        #     DLarr[i] = V.cosmo.luminosity_distance(zint[i])
        #     # dVdzarr[i] = V.dVdz(zint[i])
        #     dVdzarr[i] = V.cosmo.differential_comoving_volume(zint[i])
        self.DLf = interp1d(zint,DLarr)
        self.dVdzf = interp1d(zint,dVdzarr)
        roots = self.rootsf.ev(self.Flim,self.alpha)
        for ii in range(self.nfields):
            if self.min_comp_frac<=0.001: minlum = np.zeros_like(DLarr)
            else: minlum = np.log10(4.0*np.pi*(DLarr*3.086e24)**2 * roots[ii])
            self.minlumf.append(interp1d(zint,minlum))
            
    def setOmegaLz(self,size=501):
        ''' Create a 2-D interpolated function for Omega (fraction of sources that can be observed) '''
        logL = np.linspace(self.Lc,self.Lh,size)
        zarr = np.linspace(0.95*self.zmin,1.05*self.zmax,size)
        # xx, yy = np.meshgrid(logL,zarr)
        self.Omegaf = []
        Omegaarr = np.empty((size,size))
        for ii in range(self.nfields):
            for i in range(size):
                # Omegaarr = Omega(xx,yy,self.DLf,self.Omega_0[i],1.0e-17*self.Flim[i],self.alpha,self.fcmin)
                Omegaarr[i] = Omega(logL[i],zarr,self.DLf,self.Omega_0[ii],1.0e-17*self.Flim[ii],self.alpha,self.fcmin)
            self.Omegaf.append(RectBivariateSpline(logL,zarr,Omegaarr))

    def setlnsimple(self):
        '''Makes arrays needed for faster calculation of lnlike'''
        if self.fix_comp: self.size_ln = 201
        else: self.size_ln = 101
        self.zarr = np.linspace(self.zmin,self.zmax,self.size_ln)
        self.DL_zarr = self.DLf(self.zarr)
        self.volume_part = self.dVdzf(self.zarr)
        self.logL, self.integ_part = [], []
        self.logLi = np.empty((self.size_ln,self.size_ln))
        self.zarr_rep = np.repeat(self.zarr[None],self.size_ln,axis=0)
        for ii in range(self.nfields):
            minlumsi = self.minlumf[ii](self.zarr)
            minlumsi[minlumsi<np.min(self.lum)] = np.min(self.lum)
            for i in range(self.size_ln):
                self.logLi[:,i] = np.linspace(minlumsi[i],self.Lh,self.size_ln)
            self.logL.append(self.logLi)
            Om_part = self.Omegaf[ii].ev(self.logLi,self.zarr_rep)
            self.integ_part.append(self.volume_part * Om_part)
        self.Om_arr = Omega(self.lum,self.z,self.DLf,self.Omega_0_arr,1.0e-17*self.Flims_arr,self.alpha,self.fcmin)

    def setlncomp(self):
        '''Sets the necessary arrays for lnlike in not fixed case and calculates integral part'''
        fullint = 0.0
        for ii in range(self.nfields):
            minlumsi = np.log10(4.0*np.pi*(self.DL_zarr*3.086e24)**2 * self.roots_ln[ii])
            # minlumsi[minlumsi<np.min(self.lum)] = np.min(self.lum)
            for i in range(self.size_ln):
                self.logLi[:,i] = np.linspace(minlumsi[i],self.Lh,self.size_ln)
            tlf = TrueLumFunc(self.logLi,self.sch_al,self.Lstar,self.phistar)
            Om_part = Omega(self.logLi,self.zarr_rep,self.DLf,self.Omega_0[ii],1.0e-17*self.Flim[ii],self.alpha,self.fcmin)
            integ = tlf * self.volume_part * Om_part
            fullint += trapz(trapz(integ,self.logLi,axis=0),self.zarr)
        return fullint

    def getLumin(self):
        ''' Set the sample log luminosities (and error if flux errors available)
            based on given flux values and luminosity distance values
        '''
        if self.flux_e is not None: 
            ulum = unumpy.log10(4.0*np.pi*(self.DL*3.086e24)**2 * unumpy.uarray(self.flux,self.flux_e))
            self.lum, self.lum_e = unumpy.nominal_values(ulum), unumpy.std_devs(ulum)
        else:
            self.lum = np.log10(4.0*np.pi*(self.DL*3.086e24)**2 * self.flux)
            self.lum_e = None

    def getFluxes(self):
        ''' Set sample fluxes based on luminosities if not available '''
        if self.lum_e is not None:
            ulum = 10**unumpy.uarray(self.lum,self.lum_e)
            uflux = ulum/(4.0*np.pi*(self.DL*3.086e24)**2)
            self.flux, self.flux_e = unumpy.nominal_values(uflux), unumpy.std_devs(uflux)
        else:
            self.flux = 10**self.lum/(4.0*np.pi*(self.DL*3.086e24)**2)
            self.flux_e = None

    def getRoot(self,size=201):
        ''' Get minimum flux depending on minimum completeness fraction as interpolation'''
        flims = np.linspace(self.Flim_lims[0],self.Flim_lims[1],size)
        alphas = np.linspace(self.alpha_lims[0],self.alpha_lims[1],size)
        roots = np.zeros((size,size))
        if self.min_comp_frac>0.001:
            for i in range(size):
                for j in range(size):
                    roots[i,j] = fsolve(lambda x: V.fleming(x,1.0e-17*flims[i],alphas[j],self.fcmin)-self.min_comp_frac,[3.0e-17])[0]
        self.rootsf = RectBivariateSpline(flims,alphas,roots)

    def defineFlimOmArr(self):
        '''Function to initially define arrays of same length as the entire input to facilitate different Flim calculation''' 
        self.Flims_arr, self.Omega_0_arr = np.zeros(self.field_ind[-1]), np.zeros(self.field_ind[-1],dtype=int)
        for ii in range(self.nfields):
            self.Flims_arr[self.field_ind[ii]:self.field_ind[ii+1]] = self.Flim[ii]
            self.Omega_0_arr[self.field_ind[ii]:self.field_ind[ii+1]] = self.Omega_0[ii]

    def getFlim(self):
        '''Function to re-compute Flim full-length array''' 
        for ii in range(self.nfields):
            self.Flims_arr[self.field_ind[ii]:self.field_ind[ii+1]] = self.Flim[ii]

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
        if self.fix_comp:
            if self.fix_sch_al: pass
            else: self.sch_al = input_list[2]
        else:
            if self.fix_sch_al:
                self.Flim, self.alpha = input_list[2:2+self.nfields], input_list[2+self.nfields]
            else: 
                self.sch_al = input_list[2]
                self.Flim, self.alpha = input_list[3:3+self.nfields], input_list[3+self.nfields]

    def lnprior(self):
        ''' Simple, uniform prior for input variables

        Returns
        -------
        0.0 if all parameters are in bounds, -np.inf if any are out of bounds
        '''
        flag = 1.0
        for param in self.all_param_names:
            if param=='Flim':
                for i in range(self.nfields):
                    flag *= ((getattr(self,param)[i] >= getattr(self,param+'_lims')[0]) *
                     (getattr(self,param)[i] <= getattr(self,param+'_lims')[1]))
            else:
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
        self.roots_ln = self.rootsf.ev(self.Flim,self.alpha)
        self.getFlim()
        lnpart = np.log(TrueLumFunc(self.lum,self.sch_al,self.Lstar,self.phistar)*Omega(self.lum,self.z,self.DLf,self.Omega_0_arr,1.0e-17*self.Flims_arr,self.alpha,self.fcmin)).sum()
        # logL = np.linspace(self.Lc,self.Lh,101)
        # fullint = self.setlncomp()
        fullint = 0.0
        for ii in range(self.nfields):
            integ_part = self.volume_part * Omega(self.logL[ii],self.zarr_rep,self.DLf,self.Omega_0[ii],1.0e-17*self.Flim[ii],self.alpha,self.fcmin)
            integ = TrueLumFunc(self.logL[ii],self.sch_al,self.Lstar,self.phistar) * integ_part
            fullint += trapz(trapz(integ,self.logL[ii],axis=0),self.zarr)
        return lnpart - fullint

    def lnlike_fix_comp(self):
        ''' Calculate the log likelihood and return the value and stellar mass of the model as well as other derived parameters when completeness parameters are fixed (faster)

        Returns
        -------
        log likelihood (float)
            The log likelihood includes a ln term and an integral term (based on Poisson statistics). '''
        self.getFlim()
        lnpart = np.log(TrueLumFunc(self.lum,self.sch_al,self.Lstar,self.phistar)*self.Om_arr).sum()
        fullint = 0.0
        for ii in range(self.nfields):
            integ = TrueLumFunc(self.logL[ii],self.sch_al,self.Lstar,self.phistar) * self.integ_part[ii]
            fullint += trapz(trapz(integ,self.logL[ii],axis=0),self.zarr)
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
            # pdb.set_trace()
            return lnl+lp
        else:
            return -np.inf

    def lnprob_fix_comp(self, theta):
        ''' Calculate the log probability for fixed completeness

        Returns
        -------
        log prior + log likelihood, (float)
            The log probability is just the sum of the logs of the prior and likelihood. '''
        self.set_parameters_from_list(theta)
        lp = self.lnprior()
        if np.isfinite(lp):
            lnl = self.lnlike_fix_comp()
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
        if not self.fix_comp:
            for i in range(self.nfields): theta_lims = np.vstack((theta_lims,self.Flim_lims))
            theta_lims = np.vstack((theta_lims,self.alpha_lims))
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
        if not self.fix_comp:
            for i in range(self.nfields): names += [r'$F_{{\rm 50},%d}$'%(i)]
            names += [r'$\alpha_C$']
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
        if not self.fix_comp:
            vals += list(self.Flim)
            vals += [self.alpha]
        self.nfreeparams = len(vals)
        return vals

    def fit_model(self):
        ''' Using emcee to find parameter estimations for given set of
        data measurements and errors
        '''
        self.log.info('Fitting Schechter model to true luminosity function using emcee')
        pos = self.get_init_walker_values()
        ndim = pos.shape[1]
        start = time.time()
        if self.fix_comp: func = 'lnprob_fix_comp'
        else: func = 'lnprob'
        sampler = emcee.EnsembleSampler(self.nwalkers, ndim, getattr(self,func))
        # Do real run
        sampler.run_mcmc(pos, self.nsteps, rstate0=np.random.get_state())
        end = time.time()
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

    def VeffLF(self):
        ''' Use V_Eff method to calculate properly weighted measured luminosity function '''
        self.getFlim()
        root = self.rootsf.ev(self.Flims_arr,self.alpha)
        sum_Omega = sum(self.Omega_0)
        self.phifunc = np.zeros_like(self.flux)
        for i in range(len(self.flux)):
            if self.min_comp_frac<=0.001: zmaxval = self.zmax
            else: zmaxval = min(self.zmax,V.getMaxz(10**self.lum[i],root[i]))
            if zmaxval>self.zmin: self.phifunc[i] = V.lumfunc(self.flux[i],self.dVdzf,sum_Omega,self.zmin,zmaxval,1.0e-17*self.Flims_arr[i],self.alpha,self.fcmin)
        self.Lavg, self.lfbinorig, self.var = V.getBootErrLog(self.lum,self.phifunc,self.zmin,self.zmax,self.nboot,self.nbins,Fmin=1.0e-17*np.max(self.Flim))

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
        Flims, alphas = np.zeros((rndsamples,self.nfields)), np.zeros(rndsamples)
        lf = []
        for i in np.arange(rndsamples):
            ind = np.random.randint(0, nsamples.shape[0])
            self.set_parameters_from_list(nsamples[ind, :])
            Flims[i], alphas[i] = self.Flim, self.alpha
            modlum = TrueLumFunc(self.lum,self.sch_al,self.Lstar,self.phistar)
            lf.append(modlum)
        self.medianLF = np.median(np.array(lf), axis=0)
        self.Flim, self.alpha = list(np.median(Flims,axis=0)), np.median(alphas)
        self.VeffLF()

    def add_LumFunc_plot(self,ax1):
        """ Set up the plot for the luminosity function """
        ax1.set_yscale('log')
        ax1.set_xlabel(r"$\log$ L (erg s$^{-1}$)")
        ax1.set_ylabel(r"$\phi_{\rm{true}}$ (Mpc$^{-3}$ dex$^{-1}$)")
        ax1.minorticks_on()

    def add_subplots(self,ax1,nsamples,rndsamples=200):
        ''' Add Subplots to Triangle plot below '''
        lf = []
        indsort = np.argsort(self.lum)
        Flims, alphas = np.zeros((rndsamples,self.nfields)), np.zeros(rndsamples)
        lstars = np.zeros(rndsamples)
        for i in np.arange(rndsamples):
            ind = np.random.randint(0, nsamples.shape[0])
            self.set_parameters_from_list(nsamples[ind, :])
            Flims[i], alphas[i] = self.Flim, self.alpha
            lstars[i] = self.Lstar
            modlum = TrueLumFunc(self.lum,self.sch_al,self.Lstar,self.phistar)
            lf.append(modlum)
            ax1.plot(self.lum[indsort],modlum[indsort],color='r',linestyle='solid',alpha=0.1)
        self.medianLF = np.median(np.array(lf), axis=0)
        self.Flim, self.alpha = list(np.median(Flims,axis=0)), np.median(alphas)
        self.roots_ln = self.rootsf.ev(self.Flim,self.alpha)
        self.VeffLF()
        ax1.plot(self.lum[indsort],self.medianLF[indsort],color='dimgray',linestyle='solid')
        # cond_veff = self.Lavg >= np.log10(V.get_L_constF(max(self.roots_ln),max(self.z)))
        # ax1.errorbar(self.Lavg[cond_veff],self.lfbinorig[cond_veff],yerr=np.sqrt(self.var[cond_veff]),fmt='b^')
        # ax1.errorbar(self.Lavg[~cond_veff],self.lfbinorig[~cond_veff],yerr=np.sqrt(self.var[~cond_veff]),fmt='b^',alpha=0.2)
        xmin = np.log10(V.get_L_constF(max(self.roots_ln),min(self.z)))
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
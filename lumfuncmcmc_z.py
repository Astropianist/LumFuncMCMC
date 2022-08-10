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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pdb
import seaborn as sns
sns.set_context("paper",font_scale=1.3) # options include: talk, poster, paper
sns.set_style("ticks")
sns.set_style({"xtick.direction": "in","ytick.direction": "in",
               "xtick.top":True, "ytick.right":True,
               "xtick.major.size":12, "xtick.minor.size":4,
               "ytick.major.size":12, "ytick.minor.size":4,
               })

def getQuadCoef(y1,y2,y3,z1,z2,z3):
    ''' From Leja et al. 2020: Function to get quadratic coefficients based on quantity's values at 3 pivot points

    Input
    -----
    y1,y2,y3: Floats or Numpy 1D Arrays (same size)
        Value of a given quantity (i.e., L* or phi*) at 3 pivot points
    z1,z2,z3: Floats
        The three pivot points (redshifts)

    Return
    ------
    a,b,c: Quadratic coefficients y=az^2+bz+c
    '''
    a =((y3-y1) + (y2-y1)*(z1-z3)/(z2-z1)) / (z3**2-z1**2 + (z2**2-z1**2)*(z1-z3)/(z2-z1))
    b = (y2-y1 - a*(z2**2-z1**2)) / (z2-z1)
    c = y1 - a*z1**2 - b*z1
    return a,b,c

def schechter_z(L,z,al,L1,L2,L3,phi1,phi2,phi3,z1,z2,z3):
    ''' Inspired by Leja et al. 2020; function to calculate Schechter value at a given luminosity (or array) and redshift with certain L* and phi* values at 3 pivot points

    Input
    -----
    L: float or 1D Numpy Array
        Log luminosity (log erg/s)
    z: float or 1D Numpy Array (if L is also an array, it must be the same size)
        Redshift
    al: float
        Schechther alpha parameter
    L1,L2,L3: floats
        Schechther log(Lstar) parameter at 3 pivot redshifts
    phi1,phi2,phi3: floats
        Schechther log(phistar) parameter at 3 pivot redshifts
    z1,z2,z3: Floats
        The three pivot points (redshifts)
    '''
    aphi,bphi,cphi = getQuadCoef(phi1,phi2,phi3,z1,z2,z3)
    alum,blum,clum = getQuadCoef(L1,L2,L3,z1,z2,z3)
    phistar = aphi*z**2 + bphi*z + cphi
    Lstar = alum*z**2 + blum*z + clum
    return TrueLumFunc(L,al,Lstar,phistar)

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
    fcmin: float
        Fractional completeness below which we really do not want to exaggerate the incompleteness levels

    Returns
    -------
    Omega(logL,z) : Float or 1-D array (same size as logL and/or z)
    '''
    L = 10**logL
    return Omega_0/V.sqarcsec * V.fleming(L/(4.0*np.pi*(3.086e24*dLzfunc(z))**2),Flim,alpha,fcmin)

class LumFuncMCMCz:
    def __init__(self,z,flux=None,flux_e=None,Flim=[2.35,3.12,2.20,2.86,2.85],
                 alpha=3.5,line_name="OIII",
                 line_plot_name=r'[OIII] $\lambda 5007$',lum=None,lum_e=None,Omega_0=[100.0,100.0,100.0,100.0,100.0],nbins=50,
                 nboot=100,sch_al=-1.6, sch_al_lims=[-3.0,1.0],Lstar=42.5,Lstar_lims=[41.0,45.0],
                 phistar=-3.0,phistar_lims=[-8.0,5.0],Lc=40.0,Lh=46.0,nwalkers=100,nsteps=1000,
                 fcmin=0.1,min_comp_frac=0.5,field_names=None,
                 field_ind=None,z1=1.20,z2=1.53,z3=1.86,fix_sch_al=False):
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
        min_comp_frac: Float
            Minimum completeness fraction considered
        field_names: 1-D Numpy array
            List of fields used in data
        field_ind: 1-D Numpy array
            Indices that convert the unique field names to the full list of fluxes
        z1,z2,z3: Floats
        The three pivot points (redshifts)
        '''
        self.z = np.concatenate(z)
        self.zmin, self.zmax = min(self.z), max(self.z)
        self.z1, self.z2, self.z3 = z1, z2, z3
        self.fcmin, self.min_comp_frac = fcmin, min_comp_frac
        self.Flim = Flim
        self.fields, self.nfields = field_names, len(self.Flim)
        self.field_ind = field_ind
        self.alpha = alpha
        self.line_name = line_name
        self.line_plot_name = line_plot_name
        self.Lc, self.Lh = Lc, Lh
        self.Omega_0 = Omega_0
        self.fix_sch_al = fix_sch_al
        self.nbins, self.nboot = nbins, nboot
        self.sch_al, self.sch_al_lims = sch_al, sch_al_lims
        self.Lstar, self.Lstar_lims = Lstar, Lstar_lims
        self.phistar, self.phistar_lims = phistar, phistar_lims
        self.L1, self.L2, self.L3 = np.random.uniform(self.Lstar_lims[0]+0.5,self.Lstar_lims[-1]-0.5,3)
        self.phi1, self.phi2, self.phi3 = np.random.uniform(self.phistar_lims[0]+3,self.phistar_lims[-1]-3,3)
        self.nwalkers, self.nsteps = nwalkers, nsteps
        self.getRoot()
        self.defineFlimOmArr()
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
        self.allind = np.arange(len(self.lum))
        self.setlnsimple()
        self.setup_logging()

    def setDLdVdz(self):
        ''' Create 1-D interpolated functions for luminosity distance (cm) and comoving volume differential (Mpc^3); also get function for minimum luminosity considered '''
        zint = np.linspace(0.95*self.zmin,1.05*self.zmax,len(self.z))
        self.minlumf = []
        self.DL = V.cosmo.luminosity_distance(self.z).value
        DLarr = V.cosmo.luminosity_distance(zint).value
        dVdzarr = V.cosmo.differential_comoving_volume(zint).value
        self.DLf = interp1d(zint,DLarr)
        self.dVdzf = interp1d(zint,dVdzarr)
        for ii in range(self.nfields):
            if self.min_comp_frac<=0.001: minlum = np.zeros_like(DLarr)
            else: minlum = np.log10(4.0*np.pi*(DLarr*3.086e24)**2 * self.roots_ln[ii])
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
                Omegaarr[i] = Omega(logL[i],zarr,self.DLf,self.Omega_0[ii],1.0e-17*self.Flim[ii],self.alpha,self.fcmin)
            self.Omegaf.append(RectBivariateSpline(logL,zarr,Omegaarr))

    def setlnsimple(self):
        '''Makes arrays needed for faster calculation of lnlike'''
        self.size_ln = 201
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

    def getRoot(self):
        ''' Get F50 fluxes'''
        self.roots_ln = np.array([])
        for i in range(self.nfields):
            root = fsolve(lambda x: V.fleming(x,1.0e-17*self.Flim[i],self.alpha,self.fcmin)-self.min_comp_frac,[1.0e-17*self.Flim[i]])[0]
            self.roots_ln = np.append(self.roots_ln,root)

    def defineFlimOmArr(self):
        '''Function to initially define arrays of same length as the entire input to facilitate different Flim calculation''' 
        self.Flims_arr, self.Omega_0_arr, self.roots_arr = np.zeros(self.field_ind[-1]), np.zeros(self.field_ind[-1],dtype=int), np.zeros(self.field_ind[-1])
        for ii in range(self.nfields):
            self.Flims_arr[self.field_ind[ii]:self.field_ind[ii+1]] = self.Flim[ii]
            self.Omega_0_arr[self.field_ind[ii]:self.field_ind[ii+1]] = self.Omega_0[ii]
            self.roots_arr[self.field_ind[ii]:self.field_ind[ii+1]] = self.roots_ln[ii]

    def setup_logging(self):
        '''Setup Logging for MCSED

        Builds
        -------
        self.log : class
            self.log.info() is for general print and self.log.error() is
            for raise cases
        '''
        self.log = logging.getLogger('lumfuncmcmc_z')
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
            self.log = logging.getLogger('lumfuncmcmc_z')
            self.log.setLevel(logging.DEBUG)
            self.log.addHandler(handler)

    def set_parameters_from_list(self,input_list):
        ''' For a given set of model parameters, set the needed class variables.

        Input
        -----
        theta : list
            list of input parameters for Schechter Fit'''
        self.L1, self.L2, self.L3 = input_list[0], input_list[1], input_list[2]
        self.phi1, self.phi2, self.phi3 = input_list[3], input_list[4], input_list[5]
        if not self.fix_sch_al: self.sch_al = input_list[6]

    def lnprior(self):
        ''' Simple, uniform prior for input variables

        Returns
        -------
        0.0 if all parameters are in bounds, -np.inf if any are out of bounds
        '''
        if self.fix_sch_al: flag = 1
        else: flag = ((self.sch_al >= self.sch_al_lims[0]) * (self.sch_al <= self.sch_al_lims[1]))
        for i in range(1,4):
            val = getattr(self, 'L' + str(i))
            lims = self.Lstar_lims
            flag *= ((val > lims[0]) * (val < lims[1]))
            val = getattr(self, 'phi' + str(i))
            lims = self.phistar_lims
            flag *= ((val > lims[0]) * (val < lims[1]))
        if not flag: 
            return -np.inf
        else: 
            return 0.0

    def lnlike(self):
        ''' Calculate the log likelihood and return the value and stellar mass of the model as well as other derived parameters when completeness parameters are fixed (faster)

        Returns
        -------
        log likelihood (float)
            The log likelihood includes a ln term and an integral term (based on Poisson statistics). '''
        lnpart = np.log(schechter_z(self.lum,self.z,self.sch_al,self.L1,self.L2,self.L3,self.phi1,self.phi2,self.phi3,self.z1,self.z2,self.z3)*self.Om_arr).sum()
        fullint = 0.0
        for ii in range(self.nfields):
            integ = schechter_z(self.logL[ii],self.zarr_rep,self.sch_al,self.L1,self.L2,self.L3,self.phi1,self.phi2,self.phi3,self.z1,self.z2,self.z3) * self.integ_part[ii]
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

    def get_init_walker_values(self, num=None):
        ''' Before running emcee, this function generates starting points
        for each walker in the MCMC process.

        Returns
        -------
        pos : np.array (2 dim)
            Two dimensional array with Nwalker x Ndim values
        '''
        theta_lims = np.vstack((self.Lstar_lims,self.Lstar_lims,self.Lstar_lims,self.phistar_lims,self.phistar_lims,self.phistar_lims))
        if not self.fix_sch_al: theta_lims = np.vstack((theta_lims,self.sch_al_lims))
        if num is None:
            num = self.nwalkers
        pos = (np.random.rand(num,len(theta_lims)) *
              (theta_lims[:, 1]-theta_lims[:, 0]) + theta_lims[:, 0])
        return pos

    def get_param_names(self):
        ''' Grab the names of the parameters for plotting

        Returns
        -------
        names : list
            list of all parameter names
        '''
        names =  [r'$\log {\rm{L}}1_*$',r'$\log {\rm{L}}2_*$',r'$\log {\rm{L}}3_*$',r'$\log \phi1_*$',r'$\log \phi2_*$',r'$\log \phi3_*$']
        if not self.fix_sch_al: names += [r'$\alpha$']
        return names

    def get_params(self):
        ''' Grab the the parameters in each class

        Returns
        -------
        vals : list
            list of all parameter values
        '''
        vals = [self.L1,self.L2,self.L3,self.phi1,self.phi2,self.phi3]
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
        start = time.time()
        sampler = emcee.EnsembleSampler(self.nwalkers, ndim, self.lnprob)
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
        self.phifunc = np.zeros_like(self.flux)
        sum_Omega = sum(self.Omega_0)
        for i in range(len(self.flux)):
            if self.min_comp_frac<=0.001: zmaxval = self.zmax
            else: zmaxval = min(self.zmax,V.getMaxz(10**self.lum[i],self.roots_arr[i]))
            if zmaxval>self.zmin: self.phifunc[i] = V.lumfunc(self.flux[i],self.dVdzf,sum_Omega,self.zmin,zmaxval,1.0e-17*self.Flims_arr[i],self.alpha,self.fcmin)
        self.Lavg, self.lfbinorig, self.var = V.getBootErrLog(self.lum,self.phifunc,self.zmin,self.zmax,self.nboot,self.nbins,Fmin=1.0e-17*np.max(self.Flim))

    def set_median_fit(self,lnprobcut=7.5,zlen=100,Llen=100):
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
            nsamples = self.samples[chi2sel, :-1]
            lnprobcut *= 2.0
        self.log.info("Shape of nsamples (with a lnprobcut applied)")
        self.log.info(nsamples.shape)
        self.Lout = np.linspace(min(self.lum)-0.2,max(self.lum)+0.2,Llen)
        self.zout = np.linspace(self.zmin,self.zmax,zlen)
        self.medianLF = np.zeros((zlen,Llen))
        self.set_parameters_from_list(np.percentile(nsamples,50.0,axis=0))
        for i in np.arange(zlen):
            self.medianLF[i] = schechter_z(self.Lout,self.zout[i],self.sch_al,self.L1,self.L2,self.L3,self.phi1,self.phi2,self.phi3,self.z1,self.z2,self.z3)
        self.VeffLF()

    def add_LumFunc_plot(self,ax1):
        """ Set up the plot for the luminosity function """
        ax1.set_yscale('log')
        ax1.set_xlabel(r"$\log$ L (erg s$^{-1}$)")
        ax1.set_ylabel(r"$\phi_{\rm{true}}$ (Mpc$^{-3}$ dex$^{-1}$)")
        ax1.minorticks_on()

    def add_subplots(self,ax1,nsamples,zlen=100,Llen=100):
        ''' Add Subplots to Triangle plot below '''
        self.Lout = np.linspace(min(self.lum)-0.08,max(self.lum)+0.01,Llen)
        self.zout = np.linspace(self.zmin,self.zmax,zlen)
        LLout,zzout = np.meshgrid(self.Lout,self.zout)
        self.medianLF = np.zeros((zlen,Llen))
        self.set_parameters_from_list(np.percentile(nsamples,50.0,axis=0))
        for i in np.arange(zlen):
            self.medianLF[i] = schechter_z(self.Lout,self.zout[i],self.sch_al,self.L1,self.L2,self.L3,self.phi1,self.phi2,self.phi3,self.z1,self.z2,self.z3)
        self.VeffLF()
        im = ax1.pcolormesh(LLout,self.medianLF,zzout,shading='auto',cmap='viridis')
        # cond_veff = self.Lavg >= np.log10(V.get_L_constF(max(self.roots_ln),max(self.z)))
        # ax1.errorbar(self.Lavg[cond_veff],self.lfbinorig[cond_veff],yerr=np.sqrt(self.var[cond_veff]),fmt='r^',markersize=10,elinewidth=3)
        xmax = min(max(self.L1,self.L2,self.L3)+0.5,self.Lout.max())
        cond = self.Lout<=xmax
        # ax1.set_ylim(bottom=np.percentile(self.medianLF[:,cond],1))
        ax1.set_ylim(bottom=max(np.percentile(self.medianLF[:,cond],1),3.1e-5*self.medianLF.max()))
        ax1.set_xlim(right=xmax)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical',label='Redshift')
        
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
            poss = [0.44-0.008*(len(indarr)-4), 0.78-0.001*(len(indarr)-4), 0.48+0.008*(len(indarr)-4), 0.19+0.001*(len(indarr)-4)]
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
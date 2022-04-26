import numpy as np 
import logging
import emcee
from uncertainties import unumpy, ufloat
import matplotlib
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline
from scipy.integrate import trapz
import time
import pdb
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner
import VmaxLumFunc as V
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
    def __init__(self,z,flux=None,flux_e=None,Flim=2.7e-17,alpha=4.56,
                 line_name="OIII",line_plot_name=r'[OIII] $\lambda 5007$', 
                 lum=None,lum_e=None,Omega_0=100.0,nbins=50,nboot=100,sch_al=-1.6,
                 sch_al_lims=[-3.0,1.0],Lstar=42.5,Lstar_lims=[41.0,45.0],phistar=-3.0,
                 phistar_lims=[-8.0,5.0],Lc=40.0,Lh=46.0,nwalkers=100,nsteps=1000,root=0.0,ln_simple=True,fcmin=0.1):
        ''' Initialize LumFuncMCMC class

        Init
        ----
        z : numpy array (1 dim)
            Array of redshifts for sample
        flux : numpy array (1 dim) or None Object
            Array of fluxes in 10^-17 erg/cm^2/s
        flux_e : numpy array (1 dim) or None Object
            Array of flux errors in 10^-17 erg/cm^2/s
        Flim: float
            50% completeness flux value
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
        Omega_0: float
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
        root: Float
        Minimum flux cutoff based on the completeness curve parameters and desired minimum completeness
        '''
        self.z = z
        self.zmin, self.zmax = min(self.z), max(self.z)
        self.ln_simple = ln_simple
        self.root = root
        self.setDLdVdz()
        print("Finished setting cosmological function interpolations")
        if flux is not None: 
            self.flux = 1.0e-17*flux
            if flux_e is not None:
                self.flux_e = 1.0e-17*flux_e
        else:
            self.lum, self.lum_e = lum, lum_e
            self.getFluxes()
        self.Flim = Flim
        self.alpha = alpha
        self.fcmin = fcmin
        self.line_name = line_name
        self.line_plot_name = line_plot_name
        if lum is None: 
            self.getLumin()
        print("Finished getting all fluxes and luminosities")
        self.Lc, self.Lh = Lc, Lh
        self.Omega_0 = Omega_0
        self.setOmegaLz()
        print("Finished calculating Omega interpolation")
        self.setlnsimple()
        print("Finished setting the fixed arrays for the ln simple calculation")
        self.lumcond = self.lum[self.flux>self.root]
        self.nbins, self.nboot = nbins, nboot
        self.sch_al, self.sch_al_lims = sch_al, sch_al_lims
        self.Lstar, self.Lstar_lims = Lstar, Lstar_lims
        self.phistar, self.phistar_lims = phistar, phistar_lims
        self.nwalkers, self.nsteps = nwalkers, nsteps
        self.setup_logging()

    def setDLdVdz(self):
        ''' Create 1-D interpolated functions for luminosity distance (cm) and comoving volume differential (Mpc^3); also get function for minimum luminosity considered '''
        self.DL = np.zeros(len(self.z))
        zint = np.linspace(0.95*self.zmin,1.05*self.zmax,len(self.z))
        dVdzarr, DLarr, minlum = np.zeros(len(zint)), np.zeros(len(zint)), np.zeros(len(zint))
        for i,zi in enumerate(self.z):
            self.DL[i] = V.dLz(zi) # In Mpc
            DLarr[i] = V.dLz(zint[i])
            dVdzarr[i] = V.dVdz(zint[i])
            minlum[i] = np.log10(4.0*np.pi*(DLarr[i]*3.086e24)**2 * self.root)
        self.DLf = interp1d(zint,DLarr)
        self.dVdzf = interp1d(zint,dVdzarr)
        self.minlumf = interp1d(zint,minlum)

    def setOmegaLz_old(self):
        ''' Create a 2-D interpolated function for Omega (fraction of sources that can be observed) '''
        logL = np.linspace(self.Lc,self.Lh,501)
        zarr = np.linspace(self.zmin,self.zmax,501)
        xx, yy = np.meshgrid(logL,zarr)
        Omegaarr = Omega(xx,yy,self.DLf,self.Omega_0,self.Flim,self.alpha)
        self.Omegaf = interp2d(logL,zarr,Omegaarr,kind='cubic')

    def setOmegaLz(self,size=501):
        ''' Create a 2-D interpolated function for Omega (fraction of sources that can be observed) '''
        logL = np.linspace(self.Lc,self.Lh,size)
        zarr = np.linspace(self.zmin,self.zmax,size)
        Omegaarr = np.empty((size,size))
        for i in range(size):
            Omegaarr[i] = Omega(logL[i],zarr,self.DLf,self.Omega_0,self.Flim,self.alpha)
        self.Omegaf = RectBivariateSpline(logL,zarr,Omegaarr)

    def setlnsimple(self):
        '''Makes arrays needed for faster calculation of lnlike'''
        if self.ln_simple: self.size_ln = 201
        else: self.size_ln = 101
        self.zarr = np.linspace(self.zmin,self.zmax,self.size_ln)
        minlums = self.minlumf(self.zarr)
        minlums[minlums<np.min(self.lum)] = np.min(self.lum)
        self.logL = np.empty((self.size_ln,self.size_ln))
        for i in range(self.size_ln):
            self.logL[:,i] = np.linspace(minlums[i],self.Lh,self.size_ln)
        # self.logL = np.linspace(np.min(self.lum),self.Lh,self.size_ln)
        volume_part = self.dVdzf(self.zarr)
        Omega_part = self.Omegaf.ev(self.logL,np.repeat(self.zarr[None],self.size_ln,axis=0))
        self.integ_part = volume_part * Omega_part

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
        self.sch_al = input_list[0]
        self.Lstar = input_list[1]
        self.phistar = input_list[2]

    def lnprior(self):
        ''' Simple, uniform prior for input variables

        Returns
        -------
        0.0 if all parameters are in bounds, -np.inf if any are out of bounds
        '''
        sch_al_flag = ((self.sch_al >= self.sch_al_lims[0]) *
                     (self.sch_al <= self.sch_al_lims[1]))
        Lstar_flag = ((self.Lstar >= self.Lstar_lims[0]) *
                      (self.Lstar <= self.Lstar_lims[1]))
        phistar_flag = ((self.phistar >= self.phistar_lims[0]) *
                     (self.phistar <= self.phistar_lims[1]))

        flag = sch_al_flag*Lstar_flag*phistar_flag
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
        # tic = time.time()
        lnpart = np.log(TrueLumFunc(self.lumcond,self.sch_al,self.Lstar,self.phistar)).sum()
        # toc = time.time()
        # print("Time for lnpart calculation:",toc-tic)
        # tic = time.time()
        # lnpart_old = sum(np.log(TrueLumFunc(self.lum,self.sch_al,self.Lstar,self.phistar)))
        # toc = time.time()
        # print("Time for old lnpart calculation:",toc-tic)
        # logL = np.linspace(self.Lc,self.Lh,101)
        # tic = time.time()
        fullint = 0.0
        for i, zi in enumerate(self.zmid):
            logL = np.linspace(max(min(self.lum),self.minlumf(zi)),self.Lstar+1.75,101)
            integ = TrueLumFunc(logL,self.sch_al,self.Lstar,self.phistar)*self.dVdzf(zi)*self.Omegaf(logL,zi)
            fullint += trapz(integ,logL)*self.dz
        # toc = time.time()
        # print("Time for fullint calculation:",toc-tic)
        # pdb.set_trace()
        return lnpart - fullint

    def lnlike_simple(self):
        ''' Calculate the log likelihood and return the value and stellar mass
        of the model as well as other derived parameters

        Returns
        -------
        log likelihood (float)
            The log likelihood includes a ln term and an integral term (based on Poisson statistics). '''
        tic = time.time()
        lnpart = np.log(TrueLumFunc(self.lum,self.sch_al,self.Lstar,self.phistar)).sum()
        tlf = TrueLumFunc(self.logL,self.sch_al,self.Lstar,self.phistar)
        integ = tlf * self.integ_part
        # fullint = trapz(trapz(integ,self.zarr,axis=1),self.logL)
        fullint = trapz(trapz(integ,self.logL,axis=0),self.zarr)
        toc = time.time()
        print("Time for lnlike_simple run:",toc-tic)
        pdb.set_trace()
        return lnpart - fullint

    def lnprob(self, theta):
        ''' Calculate the log probabilty 

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

    def lnprob_simple(self, theta):
        ''' Calculate the log probabilty with quicker version

        Returns
        -------
        log prior + log likelihood, (float)
            The log probability is just the sum of the logs of the prior and likelihood. '''
        self.set_parameters_from_list(theta)
        lp = self.lnprior()
        if np.isfinite(lp):
            lnl = self.lnlike_simple()
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
        theta = [self.sch_al, self.Lstar, self.phistar]
        theta_lims = np.vstack((self.sch_al_lims,self.Lstar_lims,self.phistar_lims))
        if num is None:
            num = self.nwalkers
        pos = (np.random.rand(num)[:, np.newaxis] *
              (theta_lims[:, 1]-theta_lims[:, 0]) + theta_lims[:, 0])
        return pos

    def get_param_names(self):
        ''' Grab the names of the parameters for plotting

        Returns
        -------
        names : list
            list of all parameter names
        '''
        return [r'$\alpha$',r'$\log L_*$',r'$\log \phi_*$']

    def get_params(self):
        ''' Grab the the parameters in each class

        Returns
        -------
        vals : list
            list of all parameter values
        '''
        vals = [self.sch_al,self.Lstar,self.phistar]
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
        if self.lnlike_simple: func_name = 'lnprob_simple'
        else: func_name = 'lnprob'
        sampler = emcee.EnsembleSampler(self.nwalkers, ndim, getattr(self,func_name))
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
        self.phifunc = np.zeros(len(self.lum))
        for i in range(len(self.lum)):
            self.phifunc[i] = V.lumfunc(self.flux[i],self.dVdzf,self.Omega_0,self.zmin,self.zmax,self.Flim,self.alpha,self.fcmin)
        self.Lavg, self.lfbinorig, self.var = V.getBootErrLog(self.lum,self.phifunc,self.zmin,self.zmax,self.nboot,self.nbins,self.root)

    def calcModLumFunc(self):
        ''' Calculate modeled ("observed") luminosity function given class Schechter parameters

        Returns
        -------
        ModLumFunc(self.lum) : 1-D Numpy Array
            Array of values giving modeled "observed" luminosity function at luminosities in sample
        '''
        Omegavals = np.zeros(len(self.z))
        for i,Li,zi in zip(np.arange(len(self.z)),self.lum,self.z): 
            Omegavals[i] = self.Omegaf(Li,zi)
        return TrueLumFunc(self.lum,self.sch_al,self.Lstar,self.phistar)*Omegavals

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
        chi2sel = (self.samples[:, -1] >
                   (np.max(self.samples[:, -1], axis=0) - lnprobcut))
        nsamples = self.samples[chi2sel, :]
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

    def add_LumFunc_plot(self,ax1):
        """ Set up the plot for the luminosity function """
        ax1.set_yscale('log')
        ax1.set_xlabel(r"$\log$ L (erg s$^{-1}$)",fontsize='x-small')
        ax1.set_ylabel(r"$\phi_{\rm{true}}$ (Number Mpc$^{-3}$ dex$^{-1}$)",fontsize='xx-small')
        ax1.minorticks_on()

    def add_subplots(self,ax1,nsamples,rndsamples=200):
        ''' Add Subplots to Triangle plot below '''
        lf = []
        indsort = np.argsort(self.lum)
        for i in np.arange(rndsamples):
            ind = np.random.randint(0, nsamples.shape[0])
            self.set_parameters_from_list(nsamples[ind, :])
            modlum = TrueLumFunc(self.lum,self.sch_al,self.Lstar,self.phistar)
            lf.append(modlum)
            ax1.plot(self.lum[indsort],modlum[indsort],color='r',linestyle='solid',alpha=0.1)
        self.medianLF = np.median(np.array(lf), axis=0)
        self.VeffLF()
        ax1.plot(self.lum[indsort],self.medianLF[indsort],color='dimgray',linestyle='solid')
        ax1.errorbar(self.Lavg,self.lfbinorig,yerr=np.sqrt(self.var),fmt='b^')
        
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
        chi2sel = (self.samples[:, -1] >
                   (np.max(self.samples[:, -1], axis=0) - lnprobcut))
        nsamples = self.samples[chi2sel, :]
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
        fig.set_figwidth(w-(len(indarr)-13)*0.025*w)
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.set_position([0.50-0.008*(len(indarr)-4), 0.78-0.001*(len(indarr)-4), 
                          0.48+0.008*(len(indarr)-4), 0.19+0.001*(len(indarr)-4)])
        self.add_LumFunc_plot(ax1)
        self.add_subplots(ax1,nsamples)
        fig.savefig("%s.%s" % (outname,imgtype), dpi=200)
        plt.close(fig)

    def add_fitinfo_to_table(self, percentiles, start_value=1, lnprobcut=7.5):
        ''' Assumes that "Ln Prob" is the last column in self.samples'''
        chi2sel = (self.samples[:, -1] >
                   (np.max(self.samples[:, -1], axis=0) - lnprobcut))
        nsamples = self.samples[chi2sel, :-1]
        # nsamples = self.samples[:,:-1]
        self.log.info("Number of table entries: %d"%(len(self.table[0])))
        self.log.info("Len(percentiles): %d; len(other axis): %d"%(len(percentiles), len(np.percentile(nsamples,percentiles[0],axis=0))))
        n = len(percentiles)
        for i, per in enumerate(percentiles):
            for j, v in enumerate(np.percentile(nsamples, per, axis=0)):
                self.table[-1][(i + start_value + j*n)] = v
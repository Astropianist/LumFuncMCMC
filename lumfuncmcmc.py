import numpy as np 
import logging
import emcee
import pickle
from uncertainties import unumpy, ufloat
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.integrate import trapz
from scipy.interpolate import RegularGridInterpolator as RGIScipy
import time
import matplotlib.pyplot as plt
import corner
import VmaxLumFunc as V
from scipy.optimize import fsolve
import seaborn as sns
sns.set_context("paper",font_scale=1.3) # options include: talk, poster, paper
sns.set_style("ticks")
sns.set_style({"xtick.direction": "in","ytick.direction": "in",
               "xtick.top":True, "ytick.right":True,
               "xtick.major.size":12, "xtick.minor.size":4,
               "ytick.major.size":12, "ytick.minor.size":4,
               })

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

def makeCompFunc(file_name='cosmos_completeness_grid.pickle'):
    with open(file_name,'rb') as f:
        dat = pickle.load(f)
    mag, dist, comp = dat['Mags'], dat['Dist'], dat['Comp']
    interp_comp = RGINNExt((dist, mag), comp)
    return interp_comp

def cgs2magAB(cgs, freq):
    return -2.5*np.log10(cgs/freq)-48.6

def magAB2cgs(mag, freq):
    return freq * 10^(-0.4*(mag+48.6)) 

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

# 
def Omega(logL,dLz,compfunc,Omega_0,freq):
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
        mags = cgs2magAB(flux_cgs, freq)
        comp = compfunc(mags)
    else: 
        comp = compfunc
    return Omega_0/V.sqarcsec * comp

class LumFuncMCMC:
    def __init__(self,z,del_red=None,flux=None,flux_e=None,line_name="OIII",
                 line_plot_name=r'[OIII] $\lambda 5007$',lum=None,
                 lum_e=None,Omega_0=43200.,nbins=50,nboot=100,sch_al=-1.6,
                 sch_al_lims=[-3.0,1.0],Lstar=42.5,Lstar_lims=[40.0,45.0],
                 phistar=-3.0,phistar_lims=[-8.0,5.0],Lc=40.0,Lh=46.0,
                 nwalkers=100,nsteps=1000,fix_sch_al=False,
                 min_comp_frac=0.5,diff_rand=True,field_name='COSMOS',
                 interp_comp=None,dist_orig=None,dist=None,
                 maglow=26.0,maghigh=19.0,magnum=15,distnum=100,comps=None,
                 size_ln=1001, wav_filt=5015.0):
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
        self.freq_filt = 3.0e18/self.wav_filt
        
        self.setDLdVdz()
        if flux is not None: 
            self.flux = 1.0e-17*flux
            if flux_e is not None:
                self.flux_e = 1.0e-17*flux_e
        else:
            self.lum, self.lum_e = lum, lum_e
            self.getFluxes()
        if lum is None: 
            self.getLumin()
        self.mags = cgs2magAB(self.flux, self.freq_filt) # For the completeness
        if interp_comp is None: self.interp_comp = makeCompFunc()
        else: self.interp_comp = interp_comp
        self.comps = self.interp_comp((self.dist, self.mags))
        
        self.get1DComp()
        self.setup_logging()

    def get1DComp(self):
        ''' Get LAE-point-averaged estimates of the 1-D completeness function (of magnitude) '''
        maggrid = np.linspace(self.maghigh, self.maglow, self.magnum)
        distgrid = np.sort(np.random.choice(self.dist_orig, size=self.distnum))
        distg, magg = np.meshgrid(distgrid, maggrid, indexing='ij')
        comps = self.interp_comp((distg.ravel(), magg.ravel()))
        comps = comps.reshape(self.distnum, self.magnum)
        roots = np.zeros(self.distnum)
        for i in range(self.distnum):
            func = interp1d(maggrid, comps[i])
            roots[i] = fsolve(lambda x: func(x)-self.min_comp_frac, [23.0])[0]
        minlums = np.log10(4.0*np.pi*(self.DL*3.086e24)**2 * magAB2cgs(roots, self.freq_filt))
        self.minlum = np.average(minlums)
        comp_avg_dist = np.average(comps,axis=0)
        self.comp1df = interp1d(maggrid, comp_avg_dist, bounds_error=False, fill_value=(comp_avg_dist[0], comp_avg_dist[-1]))
        self.comps1d = self.comp1df(self.mags)
        self.Omega_arr = Omega(self.lum,self.DL,self.comps,self.Omega_0,self.freq_filt)
        self.logL = np.linspace(self.minlum,self.Lh,self.size_ln)
        self.Omega_gen = Omega(self.logL,self.DL,self.comp1df,self.Omega_0,self.freq_filt)

    def setDLdVdz(self):
        ''' Create 1-D interpolated functions for luminosity distance (cm) and comoving volume differential (Mpc^3); also get function for minimum luminosity considered '''
        self.DL = V.cosmo.luminosity_distance(self.z).value
        self.dVdz = V.dVdz(self.z)
        self.volume = self.dVdz * self.del_red # Actual total volume of survey (redshift integral separate from luminosity function integral)--divided by 4pi (since we don't divide by 4pi for Omega)

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
        lnpart = np.log(TrueLumFunc(self.lum,self.sch_al,self.Lstar,self.phistar)*self.Omega_arr).sum()
        integ = TrueLumFunc(self.logL,self.sch_al,self.Lstar,self.phistar) * self.Omega_gen
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
        start = time.time()
        func = 'lnprob'
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
        self.phifunc = 1.0/(self.volume * self.Omega_arr)
        self.Lavg, self.lfbinorig, self.var = V.getBootErrLog(self.lum,self.phifunc,self.nboot,self.nbins,Lmin=self.minlum)

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
        lstars = np.zeros(rndsamples)
        for i in np.arange(rndsamples):
            ind = np.random.randint(0, nsamples.shape[0])
            self.set_parameters_from_list(nsamples[ind, :])
            lstars[i] = self.Lstar
            modlum = TrueLumFunc(self.lum,self.sch_al,self.Lstar,self.phistar)
            lf.append(modlum)
            ax1.plot(self.lum[indsort],modlum[indsort],color='r',linestyle='solid',alpha=0.1)
        self.medianLF = np.median(np.array(lf), axis=0)
        self.VeffLF()
        ax1.plot(self.lum[indsort],self.medianLF[indsort],color='dimgray',linestyle='solid')
        cond_veff = self.Lavg >= self.minlum
        ax1.errorbar(self.Lavg[cond_veff],self.lfbinorig[cond_veff],yerr=np.sqrt(self.var[cond_veff]),fmt='b^')
        ax1.errorbar(self.Lavg[~cond_veff],self.lfbinorig[~cond_veff],yerr=np.sqrt(self.var[~cond_veff]),fmt='b^',alpha=0.2)
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
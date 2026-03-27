""" File with default values for many parameters used in the code; these can be changed directly here or with command line options in run_lumfuncmcmc.py """

nwalkers = 100 #Number of walkers for MCMC
nsteps = 1000 #Number of steps per walker for MCMC
nbins = 50 #Number of bins for V/V_max calculation
nboot = 100 #Number of bootstrap experiments for determining error for V/V_max calculation

line_name="Lya"
line_plot_name=r'${\rm{Ly\alpha}}$'
Omega_0_sqarcmin = 36000.0 #Total area of survey in arcmin^2
frac_use = 0.9 #Fraction of survey area actually usable for science
conv_minsec = 3600 #Sq arcmin to Sq arcsec
Omega_0 = Omega_0_sqarcmin*conv_minsec #Effective survey area in arcsec^2

sch_al=-1.49 #Alpha parameter of Schechter
sch_al_lims=[-3.0,0.0]
Lstar=42.5 #log Lstar parameter of Schechter
Lstar_lims=[41.5,44.0]
phistar=-2.0 #log phistar parameter of Schechter
phistar_lims=[-5.0,-1.0]
Lc=40.0 #lower luminosity limit for integrals
Lh=46.0 #upper ...
min_comp_frac = 0.5 #Completeness limit where we stop including fluxes
redshift = 3.124 #2.449 #4.552
wav_filt = 5014.0 #4193.0 #6750.0 #Angstroms
wav_rest = 1215.67 # Lya
filt_width = 77.3 #Angstroms
filt_name = 'N501'
del_red = 0.06 #Effective width of filter
field_name = 'COSMOS'
maglow, maghigh = 30., 19. #For completeness
lum_lim = 45.0 #Max luminosity 
lum_min = -99.0 #Min luminosity--if a positive value, this replaces the min completeness fraction
contam_lim = 0.01 #Maximum possible contamination fraction (1 - this quantity)
contambin = 10 #Number of bins in magnitude for contamination determination
alnum, lsnum = 101, 101 #Number of values for alpha, L* grid
contam_type = 'L_LCA' #Undetermined spectral classifications are not included for contamination; just true LAEs (and AGN at the correct redshift are considered contamation)
T_EL = 1.0 #Should just keep as 1
logL_width=2.0 #Max orders of magnitude correction considered for transmission effects
trans_file = f'{filt_name}_Nicole.txt'
corr_file = 'CorrFull.dat' #For transmission corrections in case of V/V_max
# percentiles of each parameter to report in the output file
param_percentiles = [5, 16, 50, 84, 95]
extra_text = ''

#What files to create in output
output_dict = {'parameters'    : True,
               'settings'      : True, 
               'fitposterior'  : True,
               'bestfitLF'     : True,
               'VeffLF'        : True,
               'triangle plot' : True,
               'image format'  : 'png'}

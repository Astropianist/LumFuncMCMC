nwalkers = 100
nsteps = 1000
nbins = 50
nboot = 100
Flim=2.7e-17
alpha=-2.06
line_name="OIII"
line_plot_name=r'[OIII] $\lambda 5007$'
Omega_0=1.0e6
sch_al=-1.6
sch_al_lims=[-3.0,1.0]
Lstar=42.5
Lstar_lims=[40.0,45.0]
phistar=-2.0
phistar_lims=[-8.0,5.0]
Lc=35.0
Lh=60.0
min_comp_frac = 0.5
# percentiles of each parameter to report in the output file
param_percentiles = [5, 16, 50, 84, 95]

output_dict = {'parameters'    : True,
               'settings'      : True, 
               'fitposterior'  : True,
               'bestfitLF'     : True,
               'VeffLF'        : True,
               'triangle plot' : True,
               'image format'  : 'png'}
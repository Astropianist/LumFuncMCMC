nwalkers = 100
nsteps = 1000
nbins = 50
nboot = 100
# Flim = [2.35,3.12,2.20,2.86,2.85]
Flim_aegis = 2.35
Flim_rel = [1.0, 1.2, 0.83, 1.1, 1.1]
Flim = [rel*Flim_aegis for rel in Flim_rel]
# Flim=4.0 # For OIII
# Flim = 3.1 # For H-alpha
Flim_lims=[1.0,6.0]
alpha = 4.56
# alpha=2.12 # For OIII
# alpha=2.20 # For H-alpha
alpha_lims=[1.0,7.0]
line_name="OIII"
# line_name="Ha"
line_plot_name=r'[OIII] $\lambda 5007$'
# line_plot_name=r'${\rm{H\alpha}}$'
Omega_0_sqarcmin = [121.9,122.2,116.0,147.3,118.7]
frac_use = 0.85
conv_minsec = 3600
Omega_0 = [val*frac_use*conv_minsec for val in Omega_0_sqarcmin]
# Omega_0=1.9125e6
sch_al=-1.6
sch_al_lims=[-3.0,1.0]
Lstar=42.5
Lstar_lims=[40.0,45.0]
phistar=-2.0
phistar_lims=[-8.0,5.0]
Lc=40.0
Lh=46.0
min_comp_frac = 0.0
fcmin = 0.1
# percentiles of each parameter to report in the output file
param_percentiles = [5, 16, 50, 84, 95]

output_dict = {'parameters'    : True,
               'settings'      : True, 
               'fitposterior'  : True,
               'bestfitLF'     : True,
               'VeffLF'        : True,
               'triangle plot' : True,
               'image format'  : 'png'}

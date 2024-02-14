nwalkers = 100
nsteps = 1000
nbins = 50
nboot = 100

line_name="Lya"
line_plot_name=r'${\rm{Ly\alpha}}$'
Omega_0_sqarcmin = 36000.0
frac_use = 1.0
conv_minsec = 3600
Omega_0 = Omega_0_sqarcmin*frac_use*conv_minsec

sch_al=-1.49
sch_al_lims=[-3.0,0.0]
Lstar=42.5
Lstar_lims=[41.5,43.5]
phistar=-2.0
phistar_lims=[-5.0,-1.0]
Lc=40.0
Lh=46.0
min_comp_frac = 0.5
redshift = 3.1
wav_filt = 5015.0
wav_rest = 1215.67 # Lya
filt_width = 77.3
filt_name = 'N501'
del_red = 0.06
field_name = 'COSMOS'
maglow, maghigh = 30., 19.
flux_lim = 15.0
trans_file = 'N501_with_atm.txt'
corr_file = 'TransExp/CorrFull.dat'
# percentiles of each parameter to report in the output file
param_percentiles = [5, 16, 50, 84, 95]

output_dict = {'parameters'    : True,
               'settings'      : True, 
               'fitposterior'  : False,
               'bestfitLF'     : True,
               'VeffLF'        : True,
               'triangle plot' : True,
               'image format'  : 'png'}

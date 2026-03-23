""" Code to run luminosity function calculation on narrow-band data and create several output files"""

import argparse as ap
import numpy as np
import os.path as op
import logging
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.interpolate import interp1d
from scipy.stats import ks_2samp
from lumfuncmcmc import LumFuncMCMC, makeCompFunc, makeCompFuncMag, makeCompFuncSamp, cgs2magAB, magAB2cgs, cgs2lum, lum2cgs
import VmaxLumFunc as V
from scipy import odr
import configLF
from distutils.dir_util import mkpath
import pickle
from scipy.optimize import curve_fit
import glob
import re

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper",font_scale=1.3) # options include: talk, poster, paper
sns.set_style("ticks")
sns.set_style({"xtick.direction": "in","ytick.direction": "in",
               "xtick.top":True, "ytick.right":True,
               "xtick.major.size":12, "xtick.minor.size":4,
               "ytick.major.size":12, "ytick.minor.size":4,
               })

def setup_logging():
    '''Setup Logging for LumFuncMCMC, which allows us to track status of calls and
    when errors/warnings occur.

    Returns
    -------
    log : class
        log.info() is for general print and log.error() is for raise cases
    '''
    log = logging.getLogger('lumfuncmcmc')
    if not len(log.handlers):
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
        log = logging.getLogger('lumfuncmcmc')
        log.setLevel(logging.DEBUG)
        log.addHandler(handler)
    return log

def parse_args(argv=None):
    '''Parse arguments from commandline or a manually passed list

    Parameters
    ----------
    argv : list
        list of strings such as ['-f', 'input_file.txt', '-s', 'default.ssp']

    Returns
    -------
    args : class
        args class has attributes of each input, i.e., args.filename
        as well as astributes from the config file
    '''
    parser = ap.ArgumentParser(description="LumFuncMCMC",
                               formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument("-f", "--filename",
                        help='''File to be read for galaxy data''',
                        type=str, default=None)

    parser.add_argument("-o", "--output_name",
                        help='''Output name for given run''',
                        type=str, default='test')
    
    parser.add_argument("-fn", "--field_name",
                        help='''Name of field''',
                        type=str, default=None)

    parser.add_argument("-nw", "--nwalkers",
                        help='''Number of walkers for EMCEE''',
                        type=int, default=None)

    parser.add_argument("-ns", "--nsteps",
                        help='''Number of steps for EMCEE''',
                        type=int, default=None)

    parser.add_argument("-nbins", "--nbins",
                        help='''Number of bins for evaluating 
                        true measured luminosity function from V_eff method''',
                        type=int, default=None)

    parser.add_argument("-nboot", "--nboot",
                        help='''Number of bootstrap iterations for V_eff method''',
                        type=int, default=None)

    parser.add_argument("-o0", "--Omega_0",
                        help='''Effective survey area in square degrees''',
                        type=float, default=None)
    
    parser.add_argument("-fu", "--frac_use",
                        help='''Fraction of survey area actually not covered by masks''',
                        type=float, default=None)

    parser.add_argument("-mcf", "--min_comp_frac",
                        help='''Minimum completeness fraction considered''',
                        type=float, default=None)  
    
    parser.add_argument("-ll", "--lum_lim",
                        help='''Max luminosity considered''',
                        type=float, default=None)
    
    parser.add_argument("-lm", "--lum_min",
                        help='''Max luminosity considered''',
                        type=float, default=None)
    
    parser.add_argument("-z", "--redshift",
                        help='''Redshift of sample (narrow-band)''',
                        type=float, default=None)  
    
    parser.add_argument("-dz", "--del_red",
                        help='''Redshift of sample (narrow-band)''',
                        type=float, default=None)  

    parser.add_argument("-sa", "--sch_al",
                        help='''Schechter Alpha Param''',
                        type=float, default=None)
    
    parser.add_argument("-cl", "--contam_lim",
                        help='''Contamination limit''',
                        type=float, default=None)
    
    parser.add_argument("-cb", "--contambin",
                        help='''Contamination binning''',
                        type=int, default=None)

    parser.add_argument("-fsa", "--fix_sch_al",
                        help='''Fix Schechter Alpha''',
                        action='count',default=0)

    parser.add_argument("-sr", "--same_rand",
                        help='''Same random starting point''',
                        action='count',default=0)
    
    parser.add_argument("-ec", "--err_corr",
                        help='''Whether or not to use convolution''',
                        action='count',default=0)
    
    parser.add_argument("-to", "--trans_only",
                        help='''Whether or not to use transmission pdf only''',
                        action='count',default=0)

    parser.add_argument("-th", "--top_hat",
                        help='''Whether or not to use top hat filter''',
                        action='count',default=0)
    
    parser.add_argument("-no", "--norm_only",
                        help='''Whether or not to use normal (error) pdf only''',
                        action='count',default=0)
    
    parser.add_argument("-vo", "--veff_only",
                        help='''Whether or not to only do V_eff method''',
                        action='count',default=0)
    
    parser.add_argument("-e", "--environment",
                        help='''Whether or not to divide sample by environment''',
                        type=int,default=0)
    
    parser.add_argument("-c", "--corr",
                        help='''Whether or not to correct result for the transmission effects''',
                        action='count',default=0)
    
    parser.add_argument("-a", "--alls",
                        help='''Whether or not to create al ls file''',
                        action='count',default=0)
    
    parser.add_argument("-v", "--vgal",
                        help='''Whether or not to create vgal file''',
                        action='count',default=0)
    
    parser.add_argument("-va", "--varying",
                        help='''Whether or not to vary volume for veff''',
                        action='count',default=0)

    parser.add_argument("-neb", "--num_env_bins",
                        help='''Number of bins for environment designation''',
                        type=int, default=4)

    parser.add_argument("-ln", "--line_name",
                         help='''Name of line or band for LF measurement''',
                         type=str, default=None)
    
    parser.add_argument("-et", "--extra_text",
                         help='''Extra text for alls and vgal name''',
                         type=str, default=None)

    parser.add_argument("-tf", "--filt_name",
                         help='''Filter name''',
                         type=str, default=None)

    parser.add_argument("--interp_name_r1",
                        help='''Region-1 (non-overlap) completeness pickle (SHELA mode)''',
                        type=str, default=None)

    parser.add_argument("--interp_name_r2",
                        help='''Region-2 (overlap) completeness pickle (SHELA mode)''',
                        type=str, default=None)
    
    parser.add_argument("-ct", "--contam_type",
                         help='''How to calculate contamination''',
                         type=str, default=None) 

    parser.add_argument("-ne", "--num_err",
                        help='''Whether or not to divide sample by environment''',
                        type=int,default=-1) 
    
    parser.add_argument("-co", "--combo",
                        help='''Whether or not to run ''',
                        action='count',default=0) 
    
    parser.add_argument("-duc", "--dont_use_contam",
                        help='''Whether to turn off contam treatment ''',
                        action='count',default=0) 

    # Initialize arguments and log
    args = parser.parse_args(args=argv)
    args.log = setup_logging()

    if args.Omega_0 is not None: args.Omega_0 *= 3600**2 #Convert from deg^2 to arcsec^2

    # Use config values if none are set in the input
    arg_inputs = ['nwalkers','nsteps','nbins','nboot','line_name','line_plot_name','Omega_0','sch_al','sch_al_lims','Lstar','Lstar_lims','phistar','phistar_lims','Lc','Lh','min_comp_frac','param_percentiles','output_dict','field_name', 'del_red', 'redshift', 'maglow', 'maghigh', 'wav_filt', 'filt_width', 'lum_lim', 'filt_name', 'wav_rest', 'trans_file', 'corr_file', 'alnum', 'lsnum', 'T_EL', 'contam_lim', 'contambin', 'contam_type', 'logL_width', 'lum_min', 'frac_use']

    for arg_i in arg_inputs:
        try:
            if getattr(args, arg_i) in [None, 0]:
                setattr(args, arg_i, getattr(configLF, arg_i))
        except AttributeError:
            setattr(args, arg_i, getattr(configLF, arg_i))

    if args.environment == 2: args.num_env_bins = 2
    args.interp_name = f'{args.field_name.lower()}_completeness_{args.filt_name.lower()}_grid_extrap.pickle'
    # Values here are specific to ODIN; if using for another survey, this code needs to be edited
    if args.filt_name=='N501': args.redshift, args.wav_filt, args.filt_width, args.aper_corr = 3.124, 5014.0, 77.17, -0.2352
    elif args.filt_name=='N419': args.redshift, args.wav_filt, args.filt_width, args.aper_corr = 2.449, 4193.0, 75.46, -0.2876
    else: args.redshift, args.wav_filt, args.filt_width, args.aper_corr = 4.552, 6750.0, 101.31, -0.2138
    if args.field_name.lower() != 'cosmos': args.aper_corr = 0.0
    args.del_red = args.filt_width / args.wav_rest
    args.trans_file = f'{args.filt_name}_Nicole.txt'
    delz = args.del_red * 1.5
    if args.varying: args.corr_file = f'CorrFull{args.filt_name}{args.field_name.upper()}_delz{delz:0.2f}_ngal2500000.dat'
    else: args.corr_file = f'CorrFull{args.filt_name}{args.field_name.upper()}_delz{delz:0.2f}_ngal2500000_var0.dat'
    # args.corr_file = op.join('TransExp', f'{args.filt_name}Corr_ng100000_bn20_al-1.1_delz0.08_ml41.83_Lc40.0_corr0_var1.dat')
    return args

def flin(B, x):
    ''' Linear function '''
    return B[0]*x + B[1]

def power(x, a, b):
    ''' Power function '''
    return a*x**b

def doubp(x, a1, b1, b2, x0):
    ''' Double power law '''
    a2 = a1*x0**(b1-b2)
    y = np.zeros_like(x)
    y[x<x0] = a1*x[x<x0]**b1
    y[x>=x0] = a2*x[x>=x0]**b2
    return y

def doubpv2(x, a1, b1, a2, x0):
    ''' Different double power law'''
    b2 = (a1-a2)*x0 + b1
    y = np.zeros_like(x)
    y[x<x0] = a1*x[x<x0] + b1
    y[x>=x0] = a2*x[x>=x0] + b2
    return y

def test_funcs(func=doubpv2, p0=(-1.0, 40.0, -3.0, 42.5)):
    ''' Just testing double power law on fitting the luminosity function; the new contamination method removes the need for this as the Schechter curve fits well '''
    args = parse_args()
    dir_name_first = 'LFMCMCOdin'
    output_filename = f'ODIN_fsa{args.fix_sch_al}_sa{args.sch_al:0.2f}_ml{args.lum_min}_ll{args.lum_lim}_ec2_contam_{args.contam_lim}_cb{args.contambin}{args.extra_text}'
    dir_name = op.join(dir_name_first, output_filename)
    vfile = '%s/%s_VeffLF_%s_nb%d_nw%d_ns%d_ml%0.2f_ec_%d_env%d_bin%d_c%d.dat' % (dir_name, args.output_name, output_filename, args.nbins, args.nwalkers, args.nsteps, args.lum_min, 2, args.environment, 1, args.corr)
    dat = Table.read(vfile, format='ascii')
    lum, lf, lfe = dat['Luminosity'], dat['BinLF'], dat['BinLFErr']
    loglf = np.log10(lf)
    loglfe = 1/(lf*np.log(10)) * lfe
    cond = np.logical_and(~np.isinf(loglf), ~np.isinf(loglfe))
    popt, _ = curve_fit(func, lum[cond], loglf[cond], p0=p0, sigma=loglfe[cond])
    print(popt)
    plt.errorbar(lum, loglf, yerr=loglfe, fmt='b.')
    lumarr = np.linspace(lum.min()-0.2, lum.max()+0.2, 1001)
    plt.plot(lumarr, func(lumarr, *popt), 'r-')
    plt.xlim(lumarr.min(), lumarr.max())
    plt.show()

def plotLumDistribRaw(lum_comp, lum_incomp, lum_bright, bins=40, filt_name='N419'):
    ''' Plotting raw luminosity function (straight from data) assuming single redshift for all sources; inputs are included luminosities (above minimum threshold), luminosities below threshold, and luminosities above maximum threshold (for contamination) '''
    # if filt_name=='N673': labb = 'Above bright luminosity cutoff (removed)'
    fig = plt.figure()
    labb = 'Contamination over 50% (removed)'
    plt.hist([lum_comp, lum_incomp, lum_bright], histtype='barstacked', bins=bins, color=['blue', 'lightgrey', 'gold'], label=['Above 50% completeness (kept)', 'Below 50% completeness (removed)', labb])
    plt.xlabel(r'Log luminosity (erg s$^{-1}$)')
    plt.ylabel(f'Number of sources for {filt_name}')
    plt.legend(loc='best', frameon=False)
    fig.savefig(f'LumDistRaw{filt_name}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

def plotFluxDistribRaw(flux_comp, flux_incomp, flux_bright, flux_low, bins=40, filt_name='N419', extra_text=''):
    ''' Plotting raw flux distributions '''
    # if filt_name=='N673': labb = 'Above bright luminosity cutoff (removed)'
    fig = plt.figure(figsize=(6,6))
    val = 50
    if filt_name=='N673': val = 32
    labb = f'Contamination over {val}% (removed)'
    plt.hist([np.log10(flux_comp), np.log10(flux_incomp), np.log10(flux_bright)], histtype='barstacked', bins=bins, color=['blue', 'lightgrey', 'gold'], label=[fr'Above {flux_low:0.2f} $\times 10^{{-17}}$ erg cm$^{{-2}}$ s$^{{-1}}$ (kept)', fr'Below {flux_low:0.2f} $\times 10^{{-17}}$ erg cm$^{{-2}}$ s$^{{-1}}$ (removed)', labb])
    plt.xlabel(r'Log flux ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$)', fontsize='large')
    plt.ylabel(f'Number of sources', fontsize='large')
    plt.xlim(-0.3, 2.05)
    plt.ylim(0, 725)
    plt.legend(loc='best', frameon=False, fontsize='x-small')
    fig.savefig(f'FluxDistRaw{filt_name}{extra_text}.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

def getDensityFrac(args, datfile):
    ''' Retrieve fraction to modify area of survey for specific environment'''
    dens = datfile['Density']
    pc = datfile['Protocluster']
    if args.environment: numbins = args.num_env_bins
    else: numbins = 1
    pers = np.linspace(0., 100., numbins+1)
    dens_vals = np.percentile(dens, pers)
    dens_vals[-1] += 1.0e-6 # Need to include max value in one of the bins
    densavg = np.median(dens)
    density_frac = np.ones(numbins)
    for i in range(numbins):
        cond_env = np.logical_and(dens>=dens_vals[i], dens<dens_vals[i+1])
        if args.environment==2: cond_env = abs(pc-i)<1.0e-6
        densiavg = np.median(dens[cond_env])
        density_frac[i] = densavg / densiavg
    return density_frac

def getContCorr(flux, fluxe, nb, nbe, filter='N501', extra_text=''):
    ''' Calculate and plot narrow-band to line flux relation (since completeness and contamination is done with narrow-band fluxes but the luminosity function requires line fluxes)'''
    linear = odr.Model(flin)
    data = odr.Data(flux, nb, wd=1.0/fluxe**2, we=1.0/nbe**2)
    myodr = odr.ODR(data, linear, beta0=[1.5, 0.0])
    out = myodr.run()
    out.pprint()
    fig, ax = plt.subplots()
    ax.scatter(flux, nb, c='b', s=2, label='')
    ax.errorbar(flux, nb, yerr=nbe, xerr=fluxe, fmt='none', linestyle='none', capsize=2, alpha=0.2, label='')
    fmin, fmax = flux.min(), flux.max()
    farr = np.linspace(fmin, fmax, 1001)
    ax.plot(farr, flin(out.beta, farr), 'r-', label=rf'$f_{{\rm NB}} = {out.beta[0]:0.2f}f_{{\rm line}} - {-out.beta[1]:0.2f}$')
    ax.plot(farr, farr, 'k--', label='1-1')
    ax.set_xlabel(fr'{filter} Line Flux ($10^{{-17}}$ erg cm$^{{-2}}$ s$^{{-1}}$)')
    ax.set_ylabel(rf'{filter} NB Flux ($10^{{-17}}$ erg cm$^{{-2}}$ s$^{{-1}}$)')
    ax.legend(loc='best', frameon=False)
    ax.set_xlim(fmin, fmax)
    ax.set_ylim(nb.min(), nb.max())
    fig.savefig(f'{filter}_Cont_Corr{extra_text}.png', bbox_inches='tight', dpi=300)
    out.beta[1]*=1.0e-17
    plt.close(fig)
    return out.beta

def _infer_shela_field_from_filename(filename):
    match = re.search(r'(SHELA_P\d+)', op.basename(filename))
    return match.group(1) if match is not None else None

def _get_shela_fits_filename(args):
    field = _infer_shela_field_from_filename(args.filename)
    if field is None:
        field = args.field_name
    pattern = f'{field}_{args.filt_name}_voronoi_sd_maglim_25.*.fits'
    matches = sorted(glob.glob(pattern))
    if len(matches) == 0:
        raise FileNotFoundError(f'Could not find SHELA FITS file matching {pattern}')
    return matches[0]

def _resolve_shela_comp_files(args):
    ''' Resolve region-1 and region-2 completeness pickle files.

    Priority:
      1) args.interp_name_r1 / args.interp_name_r2 if provided
      2) infer from args.interp_name using common suffixes
    '''
    fn1 = getattr(args, 'interp_name_r1', None)
    fn2 = getattr(args, 'interp_name_r2', None)
    if fn1 is not None and fn2 is not None and op.exists(fn1) and op.exists(fn2):
        return fn1, fn2

    base = args.interp_name
    if base is None:
        raise ValueError('interp_name is None; cannot infer SHELA region completeness files.')
    stem, ext = op.splitext(base)
    candidates = [
        (f'{stem}_region1{ext}', f'{stem}_region2{ext}'),
        (f'{stem}_nonoverlap{ext}', f'{stem}_overlap{ext}'),
        (f'{stem}_r1{ext}', f'{stem}_r2{ext}'),
    ]
    for c1, c2 in candidates:
        if op.exists(c1) and op.exists(c2):
            return c1, c2
    raise FileNotFoundError(
        'Could not resolve region completeness pickle files. '
        'Provide --interp_name_r1 and --interp_name_r2 or use one of '
        'the supported suffix pairs: _region1/_region2, _nonoverlap/_overlap, _r1/_r2.'
    )

def _get_shela_region_info(args, datfile):
    ''' Compute source regions and area fractions from SHELA FITS mask/exptime. '''
    fits_fn = _get_shela_fits_filename(args)
    if 'RA' not in datfile.colnames or 'DEC' not in datfile.colnames:
        raise KeyError('SHELA mode requires RA and DEC columns in input catalog.')
    ra = np.asarray(datfile['RA'], dtype=float)
    dec = np.asarray(datfile['DEC'], dtype=float)

    with fits.open(fits_fn) as hdul:
        mask = np.asarray(hdul[2].data)
        exptime = np.asarray(hdul[3].data)
        wcs = WCS(hdul[3].header)

    pos = exptime > 0
    if np.count_nonzero(pos) == 0:
        raise ValueError(f'Exposure map in {fits_fn} has no pixels with EXPTIME > 0.')
    thr = 1.5 * np.median(exptime[pos])

    x, y = wcs.wcs_world2pix(ra, dec, 0)
    xi = np.rint(x).astype(int)
    yi = np.rint(y).astype(int)
    inside = (xi >= 0) & (xi < exptime.shape[1]) & (yi >= 0) & (yi < exptime.shape[0])
    src_exptime = np.zeros(ra.size, dtype=float)
    src_exptime[inside] = exptime[yi[inside], xi[inside]]
    src_valid = inside & (src_exptime > 0)
    src_region2 = src_valid & (src_exptime >= thr)

    pix_scales_deg = proj_plane_pixel_scales(wcs)
    area_per_pix_deg2 = abs(pix_scales_deg[0] * pix_scales_deg[1])
    unmasked = mask == 0
    r2_pix = unmasked & pos & (exptime >= thr)
    r1_pix = unmasked & pos & (~r2_pix)
    area1_deg2 = np.count_nonzero(r1_pix) * area_per_pix_deg2
    area2_deg2 = np.count_nonzero(r2_pix) * area_per_pix_deg2
    area_tot_deg2 = area1_deg2 + area2_deg2
    if area_tot_deg2 <= 0:
        raise ValueError(f'Computed zero unmasked area for {fits_fn}.')
    frac1, frac2 = area1_deg2/area_tot_deg2, area2_deg2/area_tot_deg2

    return {
        'fits_file': fits_fn,
        'threshold': thr,
        'source_valid': src_valid,
        'source_region2': src_region2,
        'frac1': frac1,
        'frac2': frac2,
        'omega0_deg2': area_tot_deg2,
    }

def read_input_file(args):
    """ Function to read in input ascii file with properly named columns.
    Columns should include a (linear) flux (header 'LineorBandName_flux') 
    in 1.0e-17 erg/cm^2/s or log luminosity (header 'LineorBandName_lum') 
    in log erg/s. Errors can be included with headers 
    'LineorBandName_flux_e' or 'LineorBandName_lum_e', with the same units.
    The last required column is distance in arcmin from center of field.
    The header should simply be 'dist'

    Input
    -----
    args : class
        The args class is carried from function to function with information
        from command line input and config.py

    Return
    ------
    flux: Numpy 1-D Array
        Source fluxes (1.0e-17 erg/cm^2/s or None if not in input file)
    flux_e: Numpy 1-D Array
        Source flux errors (1.0e-17 erg/cm^2/s or None if not in input file)
    lum: Numpy 1-D Array
        Source log luminosities (log erg/s or None if not in input file)
    lum_e: Numpy 1-D Array
        Source log luminosity errors (log erg/s or oNone if not in input file)
    dist: Numpy 1-D Array
        Source distance from center of field in arcmin
    interp_comp: Scipy Regular Grid Interpolation function (modified)
        Interpolation function for completeness
    """
    
    fluxs, fluxes, dists, distos, compss, denss, areas, nbs, nbes = [], [], [], [], [], [], [], [], []
    comp_regions = []
    interp_comp_simp2 = []
    frac1_list, frac2_list, omega0_list = [], [], []

    datfile = Table.read(args.filename,format='ascii')
    DL = V.cosmo.luminosity_distance(args.redshift).value
    if args.environment: numbins = args.num_env_bins
    else: numbins = 1
    density_frac = getDensityFrac(args, datfile)
    interp_comp, interp_comp_simp_orig, interp_comp_simp, nbcontam, cf = [], [], [], [], []
    flux_lim, cgscontam = [], []

    shela_mode = ('shela' in str(args.field_name).lower()) or ('SHELA_' in op.basename(args.filename))
    shela_info = None
    if shela_mode:
        shela_info = _get_shela_region_info(args, datfile)
        args.Omega_0 = shela_info['omega0_deg2'] * 3600.0**2
        comp_file_r1, comp_file_r2 = _resolve_shela_comp_files(args)
        print(f"SHELA mode enabled with FITS: {shela_info['fits_file']}")
        print(f"SHELA EXPTIME threshold: {shela_info['threshold']}")
        print(f"SHELA area fractions: frac1={shela_info['frac1']:0.4f}, frac2={shela_info['frac2']:0.4f}")
        print(f"SHELA Omega_0 (deg^2): {shela_info['omega0_deg2']:0.6f}")

    for i in range(numbins):
        if shela_mode:
            interp_compi, interp_comp_simp_origi, interp_comp_simpi, nbcontami, cfi = makeCompFuncMag(DL,
                file_name=comp_file_r1,
                binnum=args.contambin,
                filter=args.filt_name,
                contam_type=args.contam_type,
                contam_lim=args.contam_lim,
                density_frac=density_frac[i],
                aper_corr=args.aper_corr,
                use_contam=not args.dont_use_contam,
                label='Region 1', mag_min=27.9, mag_max=21.8, wave=args.wav_filt, dwave=args.filt_width
            )
            _, _, interp_comp_simpi2, _, _ = makeCompFuncMag(DL,
                file_name=comp_file_r2,
                binnum=args.contambin,
                filter=args.filt_name,
                contam_type=args.contam_type,
                contam_lim=args.contam_lim,
                density_frac=density_frac[i],
                aper_corr=args.aper_corr,
                use_contam=not args.dont_use_contam,
                label='Region 2', mag_min=27.9, mag_max=21.8, wave=args.wav_filt, dwave=args.filt_width
            )
        else:
            if args.num_err<0:
                interp_compi, interp_comp_simp_origi, interp_comp_simpi, nbcontami, cfi = makeCompFunc(
                    DL, binnum=args.contambin, filter=args.filt_name, contam_type=args.contam_type,
                    file_name=args.interp_name, contam_lim=args.contam_lim, mag_max=21.8, mag_min=29.5,
                    density_frac=density_frac[i], aper_corr=args.aper_corr, use_contam=not args.dont_use_contam
                )
            else:
                interp_compi, interp_comp_simp_origi, interp_comp_simpi, nbcontami, cfi = makeCompFuncSamp(
                    args.num_err, DL, filter=args.filt_name, file_name=args.interp_name.replace('extrap', 'extrap_samp'),
                    contam_lim=args.contam_lim, mag_max=21.8, mag_min=29.5, aper_corr=args.aper_corr
                )
            interp_comp_simpi2 = None

        interp_comp.append(interp_compi)
        interp_comp_simp.append(interp_comp_simpi)
        interp_comp_simp_orig.append(interp_comp_simp_origi)
        interp_comp_simp2.append(interp_comp_simpi2)
        nbcontam.append(nbcontami)
        cf.append(cfi)
        if args.lum_lim<0.0: flux_limi = np.inf
        else: flux_limi = lum2cgs(args.lum_lim, DL) * 1.0e17
        print("Original flux limit:", flux_limi)
        cgscontami = magAB2cgs(nbcontami, args.wav_filt, args.filt_width)
        flux_limi = min(flux_limi, cgscontami*1.0e17)
        print("Final flux limit:", flux_limi)
        lum_limi = cgs2lum(flux_limi*1.0e-17, DL)
        print("Final luminosity limit:", lum_limi)
        flux_lim.append(flux_limi)
        cgscontam.append(cgscontami)

    fluxfull, fluxefull = datfile[f'{args.line_name}_flux'], datfile[f'{args.line_name}_flux_e']
    distfull = datfile['dist'] if 'dist' in datfile.colnames else np.zeros(len(datfile))
    nbfull, nbefull = datfile['NB_flux'], datfile['NB_flux_e']
    dens = datfile['Density']
    pc = datfile['Protocluster']

    pers = np.linspace(0., 100., numbins+1)
    dens_vals = np.percentile(dens, pers)
    dens_vals[-1] += 1.0e-6 # Need to include max value in one of the bins
    weights = np.ones(numbins)

    for i in range(numbins):
        cond_env = np.logical_and(dens>=dens_vals[i], dens<dens_vals[i+1])
        if args.environment==2: cond_env = abs(pc-i)<1.0e-6
        flux, fluxe, dist = fluxfull[cond_env], fluxefull[cond_env], distfull[cond_env]
        nb, nbe = nbfull[cond_env], nbefull[cond_env]
        cond_init = np.logical_and(flux>0.0, nb<flux_lim[i])

        region2 = None
        if shela_mode:
            valid = shela_info['source_valid'][cond_env]
            cond_init = np.logical_and(cond_init, valid)
            region2 = shela_info['source_region2'][cond_env][cond_init]

        lum = cgs2lum(1.0e-17*flux[cond_init], DL)
        mag = cgs2magAB(1.0e-17*nb[cond_init], args.wav_filt, args.filt_width)
        if shela_mode:
            comp1 = interp_comp_simp[i](mag)
            comp2 = interp_comp_simp2[i](mag)
            comps = np.where(region2, comp2, comp1)
        else:
            comps = interp_comp_simp[i].ev(dist[cond_init], mag)

        if args.lum_min>0:
            cond = lum>=args.lum_min
        else:
            cond = comps>=args.min_comp_frac
        fluxmin = lum2cgs(args.lum_min, DL)*1.0e17
        plotFluxDistribRaw(flux[cond_init][cond], flux[cond_init][~cond], flux[nb>=flux_lim[i]], fluxmin, filt_name=args.filt_name, extra_text=args.extra_text)

        densi = dens[cond_env][cond_init][cond]
        areai = 1/densi
        vals = np.percentile(areai, [5,95])
        conda = np.logical_and(areai>=vals[0],areai<=vals[-1])

        fluxs.append(flux[cond_init][cond]); fluxes.append(fluxe[cond_init][cond]); dists.append(dist[cond_init][cond]); distos.append(dist[cond_init]); compss.append(comps[cond]); denss.append(densi); areas.append(areai[conda].sum())
        nbs.append(nb[cond_init][cond]); nbes.append(nbe[cond_init][cond])
        if shela_mode:
            comp_regions.append(region2[cond])
            frac1_list.append(shela_info['frac1'])
            frac2_list.append(shela_info['frac2'])
            omega0_list.append(shela_info['omega0_deg2'] * 3600.0**2)
        else:
            comp_regions.append(None)
            frac1_list.append(1.0)
            frac2_list.append(None)
            omega0_list.append(args.Omega_0)

    areas = np.array(areas)
    for i in range(numbins):
        weights[i] = areas[i]/areas.sum()
    print("Weights for different density regions:", weights)
    return fluxs, fluxes, None, None, dists, interp_comp, interp_comp_simp_orig, interp_comp_simp, distos, compss, dens_vals, denss, flux_lim, weights, cgscontam, cf, density_frac, nbs, nbes, comp_regions, interp_comp_simp2, frac1_list, frac2_list, omega0_list

def getVeffCombo(args=None, numtot=25):
    ''' Do V/V_max method with several iterations of completeness and contamination (to consider uncertainties in those quantities) '''
    if args is None: args = parse_args()
    assert args.trans_only
    ecnum = 2
    dir_name_first = 'LFMCMCOdin'
    output_filename_orig = f'ODIN_fsa{args.fix_sch_al}_sa{args.sch_al:0.2f}_ml{args.lum_min}_ll{args.lum_lim}_ec{ecnum}_contam_{args.contam_lim}_cb{args.contambin}{args.extra_text}'
    i = 0
    if args.corr: 
        corrfile = Table.read(args.corr_file, format='ascii')
        logL, corr, corre = corrfile['logL'], corrfile['Corr'], corrfile['CorrErr']
        cond = np.logical_and(np.isfinite(corr), np.isfinite(corre))
        corrf = interp1d(logL[cond], corr[cond], kind='linear', bounds_error=False, fill_value=(corr[cond][0], corr[cond][-1]))
        corref = interp1d(logL[cond], corre[cond], kind='linear', bounds_error=False, fill_value=(corre[cond][0], corre[cond][-1]))
    else:
        corrf, corref = None, None
    lums, phis = np.zeros(0), np.zeros(0)
    
    for j in range(numtot):
        args.num_err = j
        flux, flux_e, lum, lum_e, dist, interp_comp, interp_comp_simp_orig, interp_comp_simp, dist_orig, comps, dens_vals, dens, flux_lim, weights, cgscontam, cf, density_frac, nb, nb_e, comp_region, interp_comp_simp2, frac1, frac2, omega0 = read_input_file(args)
        alls_file_name = f'Likes_alls_field{args.field_name}_z{args.redshift}_ml{args.lum_min}_ll{args.lum_lim}_env{args.environment}_neb{len(flux)}_bin{i}_contam_{args.contam_lim}_cb{args.contambin}{args.extra_text}_{j}.pickle'
        vgal_file_name = f'Likes_vgal_field{args.field_name}_z{args.redshift}_ml{args.lum_min}_contam_{args.contam_lim}_cb{args.contambin}{args.extra_text}_{j}.pickle'

        beta = getContCorr(flux[i], flux_e[i], nb[i], nb_e[i], filter=args.filt_name, extra_text=args.extra_text)

        if args.lum_min>0: minlum = args.lum_min
        else: minlum = None
        # Initialize LumFuncMCMC class
        LFmod = LumFuncMCMC(args.redshift, del_red = args.del_red, flux=flux[i], flux_e=flux_e[i], nb=nb[i], nb_e=nb_e[i], lum=lum, lum_e=lum_e, line_name=args.line_name, line_plot_name=args.line_plot_name, Omega_0=omega0[i],nbins=args.nbins, nboot=args.nboot, sch_al=args.sch_al, sch_al_lims=args.sch_al_lims, Lstar=args.Lstar, Lstar_lims=args.Lstar_lims, phistar=args.phistar, phistar_lims=args.phistar_lims, Lc=args.Lc, Lh=args.Lh, nwalkers=args.nwalkers, nsteps=args.nsteps, fix_sch_al=args.fix_sch_al, min_comp_frac=args.min_comp_frac, field_name=args.field_name, diff_rand=not args.same_rand, interp_comp=interp_comp, interp_comp_simp=interp_comp_simp[i], interp_comp_simp2=interp_comp_simp2[i], comp_region=comp_region[i], dist_orig=dist_orig[i], dist=dist[i], maglow=args.maglow, maghigh=args.maghigh, comps=comps[i], wav_filt=args.wav_filt, filt_width=args.filt_width, wav_rest=args.wav_rest, err_corr=args.err_corr, trans_only=args.trans_only, norm_only=args.norm_only, trans_file=args.trans_file, corrf=corrf, corref=corref, flux_lim=flux_lim[i], logL_width=args.logL_width, T_EL=args.T_EL, alls_file_name=alls_file_name, vgal_file_name=vgal_file_name, weight=weights[i], contam_lim=args.contam_lim, contambin=args.contambin, cgscontam=cgscontam[i], interp_comp_simp_orig=interp_comp_simp_orig[i], cf=cf[i], varying=args.varying, density_frac=density_frac[i], aper_corr=args.aper_corr, beta=beta, extra_text=args.extra_text, minlum=minlum, transsim=1, frac_use=args.frac_use, frac1=frac1[i], frac2=frac2[i])
        print("Initialized LumFuncMCMC class")
        LFmod.VeffLF(combo=True)
        lums, phis = np.concatenate((lums, LFmod.lum)), np.concatenate((phis, LFmod.phifunc))

    LFmod.VeffLF(phifunc=phis, lum=lums)

    T = Table([LFmod.Lavg, LFmod.lfbinorig/numtot, np.sqrt(LFmod.var)/numtot, LFmod.lfbinorig_orig/numtot, np.sqrt(LFmod.var_orig)/numtot],
                        names=['Luminosity', 'BinLF', 'BinLFErr', 'BinLFOrig', 'BinLFErrOrig'])
    output_filename = output_filename_orig + '_combo'
    dir_name = op.join(dir_name_first, output_filename)
    mkpath(dir_name)
    T.write('%s/%s_VeffLF_nb%d_nw%d_ns%d_ml%0.2f_ec_%d_env%d_bin%d_c%d.dat' % (dir_name, args.output_name, args.nbins, args.nwalkers, args.nsteps, args.lum_min, ecnum, args.environment, i+1, args.corr),
            overwrite=True, format='ascii.fixed_width_two_line')
    print("Finished writing VeffLF file")

def main(args=None):
    """ Read input file, run luminosity function routine, and create the appropriate output """
    # Get Inputs
    # if argv == None:
    #     argv = sys.argv
    #     argv.remove('run_lumfuncmcmc.py')

    if args is None: args = parse_args()

    # Make output folder if it doesn't exist
    if args.err_corr: ecnum = 1
    elif args.trans_only: ecnum = 2
    elif args.norm_only: ecnum = 3
    else: ecnum = 0
    dir_name_first = 'LFMCMCOdin'
    output_filename = f'ODIN_fsa{args.fix_sch_al}_sa{args.sch_al:0.2f}_ml{args.lum_min}_ll{args.lum_lim}_ec{ecnum}_contam_{args.contam_lim}_cb{args.contambin}{args.extra_text}'
    if args.top_hat: output_filename += '_th'
    if args.num_err>=0: output_filename += f'_{args.num_err}'
    # if args.filt_name=='N673': output_filename = f'ODIN_fsa{args.fix_sch_al}_sa{args.sch_al:0.2f}_ml{args.lum_min}_ll{args.lum_lim}_ec{ecnum}'
    dir_name = op.join(dir_name_first, output_filename)
    mkpath(dir_name)
    
    # Read input file into arrays
    flux, flux_e, lum, lum_e, dist, interp_comp, interp_comp_simp_orig, interp_comp_simp, dist_orig, comps, dens_vals, dens, flux_lim, weights, cgscontam, cf, density_frac, nb, nb_e, comp_region, interp_comp_simp2, frac1, frac2, omega0 = read_input_file(args)
    print("Read Input File")
    if args.corr: 
        corrfile = Table.read(args.corr_file, format='ascii')
        logL, corr, corre = corrfile['logL'], corrfile['Corr'], corrfile['CorrErr']
        cond = np.logical_and(np.isfinite(corr), np.isfinite(corre))
        corrf = interp1d(logL[cond], corr[cond], kind='linear', bounds_error=False, fill_value=(corr[cond][0], corr[cond][-1]))
        corref = interp1d(logL[cond], corre[cond], kind='linear', bounds_error=False, fill_value=(corre[cond][0], corre[cond][-1]))
    else:
        corrf, corref = None, None
    if not args.veff_only: lumlf, bestlf = [], []
    else: lumlf, bestlf = None, None
    if args.environment:
        dir_name = op.join(dir_name, str(args.num_env_bins))
        mkpath(dir_name)
        lavg, lfbinorig, var, minlums, labels_env = [], [], [], [], []
        for k in range(len(flux)):
            for kk in range(k+1, len(flux)):
                print(f"For k={k} and kk={kk}:", ks_2samp(flux[k], flux[kk]))
    for i in range(len(flux)):
        alls_file_name = f'Likes_alls_field{args.field_name}_z{args.redshift}_ml{args.lum_min}_ll{args.lum_lim}_env{args.environment}_neb{len(flux)}_bin{i}_contam_{args.contam_lim}_cb{args.contambin}{args.extra_text}.pickle'
        vgal_file_name = f'Likes_vgal_field{args.field_name}_z{args.redshift}_ml{args.lum_min}_contam_{args.contam_lim}_cb{args.contambin}{args.extra_text}.pickle'
        if args.top_hat: alls_file_name, vgal_file_name = alls_file_name.replace('.pickle', f'_th.pickle'), vgal_file_name.replace('.pickle', f'_th.pickle')
        if args.num_err>=0: alls_file_name, vgal_file_name = alls_file_name.replace('.pickle', f'_{args.num_err}.pickle'), vgal_file_name.replace('.pickle', f'_{args.num_err}.pickle')
        print("Alls file name:", alls_file_name)

        # ccorr = Table()
        # ccorr['flux'], ccorr['flue_e'], ccorr['nb'], ccorr['nb_e'] = flux[i], flux_e[i], nb[i], nb_e[i]
        # ccorr.write(f'FinalFluxes{args.filt_name}.dat', format='ascii', overwrite=True)
        beta = getContCorr(flux[i], flux_e[i], nb[i], nb_e[i], filter=args.filt_name, extra_text=args.extra_text)

        if args.lum_min>0: minlum = args.lum_min
        else: minlum = None
        # Initialize LumFuncMCMC class
        LFmod = LumFuncMCMC(args.redshift, del_red = args.del_red, flux=flux[i], flux_e=flux_e[i], nb=nb[i], nb_e=nb_e[i], lum=lum, lum_e=lum_e, line_name=args.line_name, line_plot_name=args.line_plot_name, Omega_0=omega0[i],nbins=args.nbins, nboot=args.nboot, sch_al=args.sch_al, sch_al_lims=args.sch_al_lims, Lstar=args.Lstar, Lstar_lims=args.Lstar_lims, phistar=args.phistar, phistar_lims=args.phistar_lims, Lc=args.Lc, Lh=args.Lh, nwalkers=args.nwalkers, nsteps=args.nsteps, fix_sch_al=args.fix_sch_al, min_comp_frac=args.min_comp_frac, field_name=args.field_name, diff_rand=not args.same_rand, interp_comp=interp_comp, interp_comp_simp=interp_comp_simp[i], interp_comp_simp2=interp_comp_simp2[i], comp_region=comp_region[i], dist_orig=dist_orig[i], dist=dist[i], maglow=args.maglow, maghigh=args.maghigh, comps=comps[i], wav_filt=args.wav_filt, filt_width=args.filt_width, wav_rest=args.wav_rest, err_corr=args.err_corr, trans_only=args.trans_only, norm_only=args.norm_only, trans_file=args.trans_file, corrf=corrf, corref=corref, flux_lim=flux_lim[i], logL_width=args.logL_width, T_EL=args.T_EL, alls_file_name=alls_file_name, vgal_file_name=vgal_file_name, weight=weights[i], contam_lim=args.contam_lim, contambin=args.contambin, cgscontam=cgscontam[i], interp_comp_simp_orig=interp_comp_simp_orig[i], cf=cf[i], varying=args.varying, density_frac=density_frac[i], aper_corr=args.aper_corr, beta=beta, extra_text=args.extra_text, minlum=minlum, transsim=args.veff_only, frac_use=args.frac_use, frac1=frac1[i], frac2=frac2[i])
        print("Initialized LumFuncMCMC class")
        _ = LFmod.get_params()

        if args.alls:
            if args.top_hat: als, lss, likes = LFmod.calclikeLsalTH(alnum=args.alnum, lsnum=args.lsnum)
            else: als, lss, likes = LFmod.calclikeLsal(alnum=args.alnum, lsnum=args.lsnum)
            alls_output = {}
            alls_output['Alphas'], alls_output['Lstars'], alls_output['likelihoods'] = als, lss, likes
            # pickle.dump(alls_output, open(f'Likes_alls_field{args.field_name}_z{args.redshift}_ml{args.lum_min}_fl{args.flux_lim}_env{args.environment}_bin{i}.pickle', 'wb'))

            # alls_input = pickle.load(open(f'Likes_alls_field{args.field_name}_z{args.redshift}_ml{args.lum_min}_fl{args.flux_lim}_better.pickle', 'rb'))
            pickle.dump(alls_output, open(alls_file_name, 'wb'))
            continue
        if args.vgal:
            if args.top_hat: als2, lss2, vgal = LFmod.calcVgalPhistarTH(alnum=args.alnum, lsnum=args.lsnum)
            else: als2, lss2, vgal = LFmod.calcVgalPhistar(alnum=args.alnum, lsnum=args.lsnum)
            # assert np.all(als==als2)
            # assert np.all(lss==lss2)
            alls_output = {}
            alls_output['Alphas'], alls_output['Lstars'], alls_output['Vgal'] = als2, lss2, vgal
            pickle.dump(alls_output, open(vgal_file_name, 'wb'))
            continue

        if args.veff_only:
            if args.environment: 
                LFmod.VeffLF(varying=args.varying)
                lavg.append(LFmod.Lavg); lfbinorig.append(LFmod.lfbinorig); var.append(LFmod.var); minlums.append(LFmod.minlum)
                if args.environment==1: labels_env.append(fr'{dens_vals[i]:0.2f} $\leq \sigma <$ {dens_vals[i+1]:0.2f}')
                else: labels_env.append(f'Protocluster: {i}')
                continue
            LFmod.plotVeff('%s/%s_Veff_%s_nb%d_nw%d_ns%d_ml%0.2f_ec_%d_env%d_bin%d_c%d' % (dir_name, args.output_name, output_filename, args.nbins, args.nwalkers, args.nsteps, args.lum_min, ecnum, args.environment, i+1, args.corr), imgtype = args.output_dict['image format'], varying=args.varying)
            if args.output_dict['VeffLF']:
                T = Table([LFmod.Lavg, LFmod.lfbinorig, np.sqrt(LFmod.var)],
                            names=['Luminosity', 'BinLF', 'BinLFErr'])
                T.write('%s/%s_VeffLF_%s_nb%d_nw%d_ns%d_ml%0.2f_ec_%d_env%d_bin%d_c%d.dat' % (dir_name, args.output_name, output_filename, args.nbins, args.nwalkers, args.nsteps, args.lum_min, ecnum, args.environment, i+1, args.corr),
                        overwrite=True, format='ascii.fixed_width_two_line')
                print("Finished writing VeffLF file")
            continue

        # If the run has already been completed and there is a fitposterior file, don't bother with fitting everything again
        fn = '%s/%s_fitposterior_%s_nb%d_nw%d_ns%d_ml%0.2f_ec_%d_env%d_bin%d.dat' % (dir_name, args.output_name, output_filename, args.nbins, args.nwalkers, args.nsteps, args.lum_min, ecnum, args.environment, i+1)
        if op.isfile(fn):
            dat = Table.read(fn,format='ascii')
            LFmod.samples = np.lib.recfunctions.structured_to_unstructured(dat.as_array())
            if args.output_dict['triangle plot']:
                LFmod.triangle_plot('%s/%s_triangle_%s_nb%d_nw%d_ns%d_ml%0.2f_ec_%d_env%d_bin%d_c%d' % (dir_name, args.output_name, output_filename, args.nbins, args.nwalkers, args.nsteps, args.lum_min, ecnum, args.environment, i+1, args.corr), imgtype = args.output_dict['image format'])
                print("Finished making Triangle Plot with Best-fit LF (and V_eff-method-based data)")
            else:
                LFmod.set_median_fit()
                print("Finished setting median fit and V_eff parameters")
            # LFmod.triangle_plot('%s/triangle_%s_nb%d_nw%d_ns%d_ml%0.2f_ec_%d_env%d_bin%d' % (dir_name, output_filename, args.nbins, args.nwalkers, args.nsteps, args.lum_min, ecnum, args.environment, i+1), imgtype = args.output_dict['image format'])
            if args.environment: 
                lavg.append(LFmod.Lavg); lfbinorig.append(LFmod.lfbinorig); var.append(LFmod.var); minlums.append(LFmod.minlum)
                lumlf.append(LFmod.lum); bestlf.append(LFmod.medianLF)
                if args.environment==1: labels_env.append(fr'{dens_vals[i]:0.2f} $\leq \sigma <$ {dens_vals[i+1]:0.2f}')
                else: labels_env.append(f'Protocluster: {i}')
            # T = Table([LFmod.Lavg, LFmod.lfbinorig, np.sqrt(LFmod.var)],
            #             names=['Luminosity', 'BinLF', 'BinLFErr'])
            # T.write('%s/VeffLF_%s_nb%d_nw%d_ns%d_ml%0.2f.dat' % (dir_name, output_filename, args.nbins, args.nwalkers, args.nsteps, args.lum_min),
            #         overwrite=True, format='ascii.fixed_width_two_line')
            # print("Finished writing VeffLF file")
            continue

        # Build names for parameters and labels for table
        names = LFmod.get_param_names()
        percentiles = args.param_percentiles
        labels = ['Line']
        for name in names:
            labels = labels + [name + '_%02d' % per for per in percentiles]
        formats = {}
        for label in labels:
            formats[label] = '%0.3f'
        formats['Line'] = '%s'
        print('Labels:', labels)
        
        LFmod.table = Table(names=labels, dtype=['S10'] +
                                ['f8']*(len(labels)-1))
        print("Finished making names and labels for LF table and about to start fitting the model!")
        #### Run the actual model!!! ####
        LFmod.fit_model()
        print("Finished fitting model and about to create outputs")
        #### Get desired outputs ####
        if args.output_dict['triangle plot']:
            LFmod.triangle_plot('%s/%s_triangle_%s_nb%d_nw%d_ns%d_ml%0.2f_ec_%d_env%d_bin%d_c%d' % (dir_name, args.output_name, output_filename, args.nbins, args.nwalkers, args.nsteps, args.lum_min, ecnum, args.environment, i+1, args.corr), imgtype = args.output_dict['image format'])
            print("Finished making Triangle Plot with Best-fit LF (and V_eff-method-based data)")
        else:
            LFmod.set_median_fit()
            print("Finished setting median fit and V_eff parameters")
        names.append('Ln Prob')
        if args.output_dict['fitposterior']: 
            T = Table(LFmod.samples, names=names)
            T.write('%s/%s_fitposterior_%s_nb%d_nw%d_ns%d_ml%0.2f_ec_%d_env%d_bin%d.dat' % (dir_name, args.output_name, output_filename, args.nbins, args.nwalkers, args.nsteps, args.lum_min, ecnum, args.environment, i+1),
                    overwrite=True, format='ascii.fixed_width_two_line')
            print("Finished writing fitposterior file")
        if args.output_dict['bestfitLF']:
            T = Table([LFmod.lum, LFmod.lum_e, LFmod.medianLF],
                        names=['Luminosity', 'Luminosity_Err', 'MedianLF'])
            T.write('%s/%s_bestfitLF_%s_nb%d_nw%d_ns%d_ml%0.2f_ec_%d_env%d_bin%d.dat' % (dir_name, args.output_name, output_filename, args.nbins, args.nwalkers, args.nsteps, args.lum_min, ecnum, args.environment, i+1),
                    overwrite=True, format='ascii.fixed_width_two_line')
            print("Finished writing bestfitLF file")
        if args.output_dict['VeffLF']:
            T = Table([LFmod.Lavg, LFmod.lfbinorig, np.sqrt(LFmod.var)],
                        names=['Luminosity', 'BinLF', 'BinLFErr'])
            T.write('%s/%s_VeffLF_%s_nb%d_nw%d_ns%d_ml%0.2f_ec_%d_env%d_bin%d_c%d.dat' % (dir_name, args.output_name, output_filename, args.nbins, args.nwalkers, args.nsteps, args.lum_min, ecnum, args.environment, i+1, args.corr),
                    overwrite=True, format='ascii.fixed_width_two_line')
            print("Finished writing VeffLF file")

        if args.environment:
            lavg.append(LFmod.Lavg); lfbinorig.append(LFmod.lfbinorig); var.append(LFmod.var); minlums.append(LFmod.minlum)
            lumlf.append(LFmod.lum); bestlf.append(LFmod.medianLF)
            if args.environment==1: labels_env.append(fr'{dens_vals[i]:0.2f} $\leq \sigma <$ {dens_vals[i+1]:0.2f}')
            else: labels_env.append(f'Protocluster: {i}')

        LFmod.table.add_row([args.line_name] + [0.]*(len(labels)-1))
        LFmod.add_fitinfo_to_table(percentiles)
        print(LFmod.table)

        if args.output_dict['parameters']:
            LFmod.table.write('%s/%s_%s_env%d_bin%d.dat' %(dir_name, args.output_name, output_filename, args.environment, i+1),
                            format='ascii.fixed_width_two_line',
                            formats=formats, overwrite=True)
            print("Finished writing LF main table")
        if args.output_dict['settings']:
            filename = open('%s/%s_%s_env%d_bin%d.dat.args' %(dir_name, args.output_name, output_filename, args.environment, i+1), 'w')
            try: del args.log
            except: pass
            filename.write( str( vars(args) ) )
            filename.close()
            print("Finished writing settings to file")
    
    if args.environment:
        LFmod.plotVeffEnv(lavg, lfbinorig, var, minlums, labels_env, '%s/%s_Veff_%s_nb%d_nw%d_ns%d_ml%0.2f_ec_%d_env%d_split_%d_c%d_bins' % (dir_name, args.output_name, output_filename, args.nbins, args.nwalkers, args.nsteps, args.lum_min, ecnum, args.environment, args.num_env_bins, args.corr), imgtype=args.output_dict['image format'], lflums=lumlf, lfs=bestlf)

if __name__ == '__main__':
    args = parse_args()
    if args.combo: getVeffCombo(args=args)
    else: main(args=args)
    # test_funcs()
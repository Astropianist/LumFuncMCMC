import sys
import argparse as ap
import numpy as np
import os.path as op
import logging
from astropy.table import Table
from scipy.interpolate import interp1d
from scipy.stats import ks_2samp
from lumfuncmcmc import LumFuncMCMC, makeCompFunc, cgs2magAB, magAB2cgs, cgs2lum
import VmaxLumFunc as V
from scipy.optimize import fsolve
import configLF
from distutils.dir_util import mkpath
import pickle

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
                        help='''Effective survey area in square arcseconds''',
                        type=float, default=None)

    parser.add_argument("-mcf", "--min_comp_frac",
                        help='''Minimum completeness fraction considered''',
                        type=float, default=None)  
    
    parser.add_argument("-ll", "--lum_lim",
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
    
    parser.add_argument("-ct", "--contam_type",
                         help='''How to calculate contamination''',
                         type=str, default=None)  

    # Initialize arguments and log
    args = parser.parse_args(args=argv)
    args.log = setup_logging()

    # Use config values if none are set in the input
    arg_inputs = ['nwalkers','nsteps','nbins','nboot','line_name','line_plot_name','Omega_0','sch_al','sch_al_lims','Lstar','Lstar_lims','phistar','phistar_lims','Lc','Lh','min_comp_frac','param_percentiles','output_dict','field_name', 'del_red', 'redshift', 'maglow', 'maghigh', 'wav_filt', 'filt_width', 'lum_lim', 'filt_name', 'wav_rest', 'trans_file', 'corr_file', 'alnum', 'lsnum', 'T_EL', 'contam_lim', 'contambin', 'contam_type']

    for arg_i in arg_inputs:
        try:
            if getattr(args, arg_i) in [None, 0]:
                setattr(args, arg_i, getattr(configLF, arg_i))
        except AttributeError:
            setattr(args, arg_i, getattr(configLF, arg_i))

    if args.environment == 2: args.num_env_bins = 2
    args.interp_name = f'{args.field_name.lower()}_completeness_{args.filt_name.lower()}_grid_extrap.pickle'
    if args.filt_name=='N501': args.redshift, args.wav_filt, args.filt_width = 3.124, 5014.0, 77.17
    elif args.filt_name=='N419': args.redshift, args.wav_filt, args.filt_width = 2.449, 4193.0, 75.46
    else: args.redshift, args.wav_filt, args.filt_width = 4.552, 6750.0, 101.31
    args.del_red = args.filt_width / args.wav_rest
    args.trans_file = f'{args.filt_name}_Nicole.txt'
    # args.corr_file = f'CorrFull{args.filt_name}_delz0.1_ngal2500000.dat'
    args.corr_file = op.join('TransExp', f'{args.filt_name}Corr_ng100000_bn20_al-1.1_delz0.08_ml41.83_Lc40.0_corr0_var1.dat')

    return args

def plotLumDistribRaw(lum_comp, lum_incomp, lum_bright, bins=40, filt_name='N419'):
    # if filt_name=='N673': labb = 'Above bright luminosity cutoff (removed)'
    labb = 'Contamination over 99% (removed)'
    plt.hist([lum_comp, lum_incomp, lum_bright], histtype='barstacked', bins=bins, color=['blue', 'lightgrey', 'mistyrose'], label=['Above 50% completeness (kept)', 'Below 50% completeness (removed)', labb])
    plt.xlabel(r'Log luminosity (erg s$^{-1}$)')
    plt.ylabel(f'Number of sources for {filt_name}')
    plt.legend(loc='best', frameon=False)
    plt.savefig(f'LumDistRaw{filt_name}.png', bbox_inches='tight', dpi=300)

def getDensityFrac(args, datfile):
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
    
    fluxs, fluxes, dists, distos, compss, denss, areas = [], [], [], [], [], [], []
    datfile = Table.read(args.filename,format='ascii')
    DL = V.cosmo.luminosity_distance(args.redshift).value
    if args.environment: numbins = args.num_env_bins
    else: numbins = 1
    density_frac = getDensityFrac(args, datfile)
    interp_comp, interp_comp_simp_orig, interp_comp_simp, nbcontam, cf = [], [], [], [], []
    flux_lim, cgscontam = [], []
    for i in range(numbins):
        interp_compi, interp_comp_simp_origi, interp_comp_simpi, nbcontami, cfi = makeCompFunc(DL, binnum=args.contambin, filter=args.filt_name, contam_type=args.contam_type, file_name=args.interp_name, contam_lim=args.contam_lim, mag_max=21.8, mag_min=29.5, density_frac=density_frac[i])
        interp_comp.append(interp_compi); interp_comp_simp.append(interp_comp_simpi); interp_comp_simp_orig.append(interp_comp_simp_origi); nbcontam.append(nbcontami); cf.append(cfi)
        if args.lum_lim<0.0: flux_limi = np.inf
        else: flux_limi = 10**args.lum_lim / (4.0*np.pi*(3.086e24*DL)**2) * 1.0e17 #From log luminosity to 1.0e-17 cgs flux
        print("Original flux limit:", flux_limi)
        cgscontami = magAB2cgs(nbcontam[i], args.wav_filt, args.filt_width)
        flux_limi = min(flux_limi, cgscontami*1.0e17)
        print("Final flux limit:", flux_limi)
        lum_limi = cgs2lum(flux_limi*1.0e-17, DL)
        print("Final luminosity limit:", lum_limi)
        flux_lim.append(flux_limi); cgscontam.append(cgscontami)
    fluxfull, fluxefull, distfull = datfile[f'{args.line_name}_flux'], datfile[f'{args.line_name}_flux_e'], datfile['dist']
    dens = datfile['Density']
    pc = datfile['Protocluster']
    
    pers = np.linspace(0., 100., numbins+1)
    dens_vals = np.percentile(dens, pers)
    dens_vals[-1] += 1.0e-6 # Need to include max value in one of the bins
    weights = np.ones(numbins)
    # cond_init = np.logical_and(fluxfull>0.0, fluxfull<flux_lim)
    # mag = cgs2magAB(1.0e-17*fluxfull[cond_init], args.wav_filt, args.filt_width)
    # comps = interp_comp_simp.ev(distfull[cond_init], mag)
    # cond = comps>=args.min_comp_frac
    # densfull = dens[cond_init][cond]
    # densfullavg = np.median(densfull)
    # density_frac = np.ones(numbins)
    for i in range(numbins):
        cond_env = np.logical_and(dens>=dens_vals[i], dens<dens_vals[i+1])
        if args.environment==2: cond_env = abs(pc-i)<1.0e-6
        flux, fluxe, dist = fluxfull[cond_env], fluxefull[cond_env], distfull[cond_env]
        cond_init = np.logical_and(flux>0.0, flux<flux_lim[i])
        lum = np.log10(1.0e-17*flux[cond_init] * 4.0*np.pi*(3.086e24*DL)**2)
        lumb = np.log10(1.0e-17*flux[flux>=flux_lim[i]] * 4.0*np.pi*(3.086e24*DL)**2)
        mag = cgs2magAB(1.0e-17*flux[cond_init], args.wav_filt, args.filt_width)
        comps = interp_comp_simp[i].ev(dist[cond_init], mag)
        # compsorig = interp_comp_simp_orig.ev(dist[cond_init], mag)
        cond = comps>=args.min_comp_frac
        # plotLumDistribRaw(lum[cond], lum[~cond], lumb, filt_name=args.filt_name)
        densi = dens[cond_env][cond_init][cond]
        # densiavg = np.median(densi)
        # density_frac[i] = densfullavg / densiavg
        areai = 1/densi
        vals = np.percentile(areai, [5,95])
        conda = np.logical_and(areai>=vals[0],areai<=vals[-1])

        fluxs.append(flux[cond_init][cond]); fluxes.append(fluxe[cond_init][cond]); dists.append(dist[cond_init][cond]); distos.append(dist[cond_init]); compss.append(comps[cond]); denss.append(densi); areas.append(areai[conda].sum())
    areas = np.array(areas)
    for i in range(numbins):
        weights[i] = areas[i]/areas.sum()
    print("Weights for different density regions:", weights)
    return fluxs, fluxes, None, None, dists, interp_comp, interp_comp_simp_orig, interp_comp_simp, distos, compss, dens_vals, denss, flux_lim, weights, cgscontam, cf, density_frac

def main(argv=None):
    """ Read input file, run luminosity function routine, and create the appropriate output """
    # Get Inputs
    if argv == None:
        argv = sys.argv
        argv.remove('run_lumfuncmcmc.py')

    args = parse_args(argv)

    # Make output folder if it doesn't exist
    if args.err_corr: ecnum = 1
    elif args.trans_only: ecnum = 2
    elif args.norm_only: ecnum = 3
    else: ecnum = 0
    dir_name_first = 'LFMCMCOdin'
    output_filename = f'ODIN_fsa{args.fix_sch_al}_sa{args.sch_al:0.2f}_mcf{int(100*args.min_comp_frac)}_ll{args.lum_lim}_ec{ecnum}_contam_{args.contam_lim}_cb{args.contambin}{args.extra_text}'
    # if args.filt_name=='N673': output_filename = f'ODIN_fsa{args.fix_sch_al}_sa{args.sch_al:0.2f}_mcf{int(100*args.min_comp_frac)}_ll{args.lum_lim}_ec{ecnum}'
    dir_name = op.join(dir_name_first, output_filename)
    mkpath(dir_name)
    
    # Read input file into arrays
    flux, flux_e, lum, lum_e, dist, interp_comp, interp_comp_simp_orig, interp_comp_simp, dist_orig, comps, dens_vals, dens, flux_lim, weights, cgscontam, cf, density_frac = read_input_file(args)
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
        lavg, lfbinorig, var, minlum, labels_env = [], [], [], [], []
        for k in range(len(flux)):
            for kk in range(k+1, len(flux)):
                print(f"For k={k} and kk={kk}:", ks_2samp(flux[k], flux[kk]))
    for i in range(len(flux)):
        alls_file_name = f'Likes_alls_field{args.field_name}_z{args.redshift}_mcf{args.min_comp_frac}_ll{args.lum_lim}_env{args.environment}_neb{len(flux)}_bin{i}_contam_{args.contam_lim}_cb{args.contambin}{args.extra_text}.pickle'
        vgal_file_name = f'Likes_vgal_field{args.field_name}_z{args.redshift}_contam_{args.contam_lim}_cb{args.contambin}{args.extra_text}.pickle'
        print("Alls file name:", alls_file_name)

        # Initialize LumFuncMCMC class
        LFmod = LumFuncMCMC(args.redshift, del_red = args.del_red, flux=flux[i], 
                            flux_e=flux_e[i], lum=lum, 
                            lum_e=lum_e, line_name=args.line_name,
                            line_plot_name=args.line_plot_name, 
                            Omega_0=args.Omega_0,nbins=args.nbins, 
                            nboot=args.nboot, sch_al=args.sch_al, 
                            sch_al_lims=args.sch_al_lims, Lstar=args.Lstar, 
                            Lstar_lims=args.Lstar_lims, phistar=args.phistar, 
                            phistar_lims=args.phistar_lims, Lc=args.Lc, 
                            Lh=args.Lh, nwalkers=args.nwalkers, 
                            nsteps=args.nsteps, fix_sch_al=args.fix_sch_al,
                            min_comp_frac=args.min_comp_frac, 
                            field_name=args.field_name, 
                            diff_rand=not args.same_rand, 
                            interp_comp=interp_comp, interp_comp_simp=interp_comp_simp[i], dist_orig=dist_orig[i], 
                            dist=dist[i], maglow=args.maglow, maghigh=args.maghigh, comps=comps[i], wav_filt=args.wav_filt, filt_width=args.filt_width, wav_rest=args.wav_rest,
                            err_corr=args.err_corr, trans_only=args.trans_only,
                            norm_only=args.norm_only, trans_file=args.trans_file,
                            corrf=corrf, corref=corref, flux_lim=flux_lim[i],
                            logL_width=4.0, T_EL=args.T_EL, alls_file_name=alls_file_name, vgal_file_name=vgal_file_name, weight=weights[i], contam_lim=args.contam_lim, contambin=args.contambin, cgscontam=cgscontam[i], interp_comp_simp_orig=interp_comp_simp_orig[i], cf=cf[i], varying=args.varying, density_frac=density_frac[i])
        print("Initialized LumFuncMCMC class")
        _ = LFmod.get_params()

        if args.alls:
            als, lss, likes = LFmod.calclikeLsal(alnum=args.alnum, lsnum=args.lsnum)
            alls_output = {}
            alls_output['Alphas'], alls_output['Lstars'], alls_output['likelihoods'] = als, lss, likes
            # pickle.dump(alls_output, open(f'Likes_alls_field{args.field_name}_z{args.redshift}_mcf{args.min_comp_frac}_fl{args.flux_lim}_env{args.environment}_bin{i}.pickle', 'wb'))

            # alls_input = pickle.load(open(f'Likes_alls_field{args.field_name}_z{args.redshift}_mcf{args.min_comp_frac}_fl{args.flux_lim}_better.pickle', 'rb'))
            pickle.dump(alls_output, open(alls_file_name, 'wb'))
            continue
        if args.vgal:
            als2, lss2, vgal = LFmod.calcVgalPhistar(alnum=args.alnum, lsnum=args.lsnum)
            # assert np.all(als==als2)
            # assert np.all(lss==lss2)
            alls_output = {}
            alls_output['Alphas'], alls_output['Lstars'], alls_output['Vgal'] = als2, lss2, vgal
            pickle.dump(alls_output, open(vgal_file_name, 'wb'))
            continue

        if args.veff_only:
            if args.environment: 
                LFmod.VeffLF(varying=args.varying)
                lavg.append(LFmod.Lavg); lfbinorig.append(LFmod.lfbinorig); var.append(LFmod.var); minlum.append(LFmod.minlum)
                if args.environment==1: labels_env.append(fr'{dens_vals[i]:0.2f} $\leq \sigma <$ {dens_vals[i+1]:0.2f}')
                else: labels_env.append(f'Protocluster: {i}')
                continue
            LFmod.plotVeff('%s/%s_Veff_%s_nb%d_nw%d_ns%d_mcf%d_ec_%d_env%d_bin%d_c%d' % (dir_name, args.output_name, output_filename, args.nbins, args.nwalkers, args.nsteps, int(100*args.min_comp_frac), ecnum, args.environment, i+1, args.corr), imgtype = args.output_dict['image format'], varying=args.varying)
            if args.output_dict['VeffLF']:
                T = Table([LFmod.Lavg, LFmod.lfbinorig, np.sqrt(LFmod.var)],
                            names=['Luminosity', 'BinLF', 'BinLFErr'])
                T.write('%s/%s_VeffLF_%s_nb%d_nw%d_ns%d_mcf%d_ec_%d_env%d_bin%d_c%d.dat' % (dir_name, args.output_name, output_filename, args.nbins, args.nwalkers, args.nsteps, int(100*args.min_comp_frac), ecnum, args.environment, i+1, args.corr),
                        overwrite=True, format='ascii.fixed_width_two_line')
                print("Finished writing VeffLF file")
            continue

        # If the run has already been completed and there is a fitposterior file, don't bother with fitting everything again
        fn = '%s/%s_fitposterior_%s_nb%d_nw%d_ns%d_mcf%d_ec_%d_env%d_bin%d.dat' % (dir_name, args.output_name, output_filename, args.nbins, args.nwalkers, args.nsteps, int(100*args.min_comp_frac), ecnum, args.environment, i+1)
        if op.isfile(fn):
            dat = Table.read(fn,format='ascii')
            LFmod.samples = np.lib.recfunctions.structured_to_unstructured(dat.as_array())
            if args.output_dict['triangle plot']:
                LFmod.triangle_plot('%s/%s_triangle_%s_nb%d_nw%d_ns%d_mcf%d_ec_%d_env%d_bin%d_c%d' % (dir_name, args.output_name, output_filename, args.nbins, args.nwalkers, args.nsteps, int(100*args.min_comp_frac), ecnum, args.environment, i+1, args.corr), imgtype = args.output_dict['image format'])
                print("Finished making Triangle Plot with Best-fit LF (and V_eff-method-based data)")
            else:
                LFmod.set_median_fit()
                print("Finished setting median fit and V_eff parameters")
            # LFmod.triangle_plot('%s/triangle_%s_nb%d_nw%d_ns%d_mcf%d_ec_%d_env%d_bin%d' % (dir_name, output_filename, args.nbins, args.nwalkers, args.nsteps, int(100*args.min_comp_frac), ecnum, args.environment, i+1), imgtype = args.output_dict['image format'])
            if args.environment: 
                lavg.append(LFmod.Lavg); lfbinorig.append(LFmod.lfbinorig); var.append(LFmod.var); minlum.append(LFmod.minlum)
                lumlf.append(LFmod.lum); bestlf.append(LFmod.medianLF)
                if args.environment==1: labels_env.append(fr'{dens_vals[i]:0.2f} $\leq \sigma <$ {dens_vals[i+1]:0.2f}')
                else: labels_env.append(f'Protocluster: {i}')
            # T = Table([LFmod.Lavg, LFmod.lfbinorig, np.sqrt(LFmod.var)],
            #             names=['Luminosity', 'BinLF', 'BinLFErr'])
            # T.write('%s/VeffLF_%s_nb%d_nw%d_ns%d_mcf%d.dat' % (dir_name, output_filename, args.nbins, args.nwalkers, args.nsteps, int(100*args.min_comp_frac)),
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
            LFmod.triangle_plot('%s/%s_triangle_%s_nb%d_nw%d_ns%d_mcf%d_ec_%d_env%d_bin%d_c%d' % (dir_name, args.output_name, output_filename, args.nbins, args.nwalkers, args.nsteps, int(100*args.min_comp_frac), ecnum, args.environment, i+1, args.corr), imgtype = args.output_dict['image format'])
            print("Finished making Triangle Plot with Best-fit LF (and V_eff-method-based data)")
        else:
            LFmod.set_median_fit()
            print("Finished setting median fit and V_eff parameters")
        names.append('Ln Prob')
        if args.output_dict['fitposterior']: 
            T = Table(LFmod.samples, names=names)
            T.write('%s/%s_fitposterior_%s_nb%d_nw%d_ns%d_mcf%d_ec_%d_env%d_bin%d.dat' % (dir_name, args.output_name, output_filename, args.nbins, args.nwalkers, args.nsteps, int(100*args.min_comp_frac), ecnum, args.environment, i+1),
                    overwrite=True, format='ascii.fixed_width_two_line')
            print("Finished writing fitposterior file")
        if args.output_dict['bestfitLF']:
            T = Table([LFmod.lum, LFmod.lum_e, LFmod.medianLF],
                        names=['Luminosity', 'Luminosity_Err', 'MedianLF'])
            T.write('%s/%s_bestfitLF_%s_nb%d_nw%d_ns%d_mcf%d_ec_%d_env%d_bin%d.dat' % (dir_name, args.output_name, output_filename, args.nbins, args.nwalkers, args.nsteps, int(100*args.min_comp_frac), ecnum, args.environment, i+1),
                    overwrite=True, format='ascii.fixed_width_two_line')
            print("Finished writing bestfitLF file")
        if args.output_dict['VeffLF']:
            T = Table([LFmod.Lavg, LFmod.lfbinorig, np.sqrt(LFmod.var)],
                        names=['Luminosity', 'BinLF', 'BinLFErr'])
            T.write('%s/%s_VeffLF_%s_nb%d_nw%d_ns%d_mcf%d_ec_%d_env%d_bin%d_c%d.dat' % (dir_name, args.output_name, output_filename, args.nbins, args.nwalkers, args.nsteps, int(100*args.min_comp_frac), ecnum, args.environment, i+1, args.corr),
                    overwrite=True, format='ascii.fixed_width_two_line')
            print("Finished writing VeffLF file")

        if args.environment:
            lavg.append(LFmod.Lavg); lfbinorig.append(LFmod.lfbinorig); var.append(LFmod.var); minlum.append(LFmod.minlum)
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
        LFmod.plotVeffEnv(lavg, lfbinorig, var, minlum, labels_env, '%s/%s_Veff_%s_nb%d_nw%d_ns%d_mcf%d_ec_%d_env%d_split_%d_c%d_bins' % (dir_name, args.output_name, output_filename, args.nbins, args.nwalkers, args.nsteps, int(100*args.min_comp_frac), ecnum, args.environment, args.num_env_bins, args.corr), imgtype=args.output_dict['image format'], lflums=lumlf, lfs=bestlf)

if __name__ == '__main__':
    main()
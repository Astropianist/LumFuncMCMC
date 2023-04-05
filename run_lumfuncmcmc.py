import sys
import argparse as ap
import numpy as np
import os.path as op
import logging
from astropy.table import Table
from scipy.interpolate import interp1d
from lumfuncmcmc import LumFuncMCMC, makeCompFunc, cgs2magAB
import VmaxLumFunc as V
from scipy.optimize import fsolve
import configLF
from distutils.dir_util import mkpath

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

    parser.add_argument("-o", "--output_filename",
                        help='''Output filename for given run''',
                        type=str, default='test.dat')
    
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
    
    parser.add_argument("-z", "--redshift",
                        help='''Redshift of sample (narrow-band)''',
                        type=float, default=None)  
    
    parser.add_argument("-dz", "--del_red",
                        help='''Redshift of sample (narrow-band)''',
                        type=float, default=None)  

    parser.add_argument("-sa", "--sch_al",
                        help='''Schechter Alpha Param''',
                        type=float, default=None)

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

    parser.add_argument("-ln", "--line_name",
                         help='''Name of line or band for LF measurement''',
                         type=str, default=None)               

    # Initialize arguments and log
    args = parser.parse_args(args=argv)
    args.log = setup_logging()

    # Use config values if none are set in the input
    arg_inputs = ['nwalkers','nsteps','nbins','nboot','line_name','line_plot_name','Omega_0','sch_al','sch_al_lims','Lstar','Lstar_lims','phistar','phistar_lims','Lc','Lh','min_comp_frac','param_percentiles','output_dict','field_name', 'del_red', 'redshift', 'maglow', 'maghigh', 'wav_filt', 'filt_width', 'flux_lim', 'filt_name', 'wav_rest']

    for arg_i in arg_inputs:
        try:
            if getattr(args, arg_i) in [None, 0]:
                setattr(args, arg_i, getattr(configLF, arg_i))
        except AttributeError:
            setattr(args, arg_i, getattr(configLF, arg_i))

    return args

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
    
    datfile = Table.read(args.filename,format='ascii')
    interp_comp = makeCompFunc()
    flux, fluxe, dist = datfile[f'{args.line_name}_flux'], datfile[f'{args.line_name}_flux_e'], datfile['dist']
    cond_init = np.logical_and(flux>0.0,flux<args.flux_lim)
    mag = cgs2magAB(1.0e-17*flux[cond_init], args.wav_filt, args.filt_width)
    comps = interp_comp((dist[cond_init], mag))
    cond = comps>=args.min_comp_frac
    return flux[cond_init][cond], fluxe[cond_init][cond], None, None, dist[cond_init][cond], interp_comp, dist[cond_init], comps[cond]

def main(argv=None):
    """ Read input file, run luminosity function routine, and create the appropriate output """
    # Make output folder if it doesn't exist
    dir_name = 'LFMCMCOdin'
    mkpath(dir_name)
    # Get Inputs
    if argv == None:
        argv = sys.argv
        argv.remove('run_lumfuncmcmc.py')

    args = parse_args(argv)
    # Read input file into arrays
    flux, flux_e, lum, lum_e, dist, interp_comp, dist_orig, comps = read_input_file(args)
    print("Read Input File")

    # Initialize LumFuncMCMC class
    LFmod = LumFuncMCMC(args.redshift, del_red = args.del_red, flux=flux, 
                        flux_e=flux_e, lum=lum, 
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
                        interp_comp=interp_comp, dist_orig=dist_orig, 
                        dist=dist, maglow=args.maglow, maghigh=args.maghigh, comps=comps, wav_filt=args.wav_filt, filt_width=args.filt_width, wav_rest=args.wav_rest,
                        err_corr=args.err_corr, trans_only=args.trans_only)
    print("Initialized LumFuncMCMC class")

    # If the run has already been completed and there is a fitposterior file, don't bother with fitting everything again
    fn = '%s/fitposterior_%s_nb%d_nw%d_ns%d_mcf%d_ec_%d.dat' % (dir_name, args.output_filename.split('.')[0], args.nbins, args.nwalkers, args.nsteps, int(100*args.min_comp_frac), args.err_corr)
    if op.isfile(fn):
        dat = Table.read(fn,format='ascii')
        LFmod.samples = np.lib.recfunctions.structured_to_unstructured(dat.as_array())
        LFmod.triangle_plot('%s/triangle_%s_nb%d_nw%d_ns%d_mcf%d_ec_%d' % (dir_name, args.output_filename.split('.')[0], args.nbins, args.nwalkers, args.nsteps, int(100*args.min_comp_frac), args.err_corr), imgtype = args.output_dict['image format'])
        # T = Table([LFmod.Lavg, LFmod.lfbinorig, np.sqrt(LFmod.var)],
        #             names=['Luminosity', 'BinLF', 'BinLFErr'])
        # T.write('%s/VeffLF_%s_nb%d_nw%d_ns%d_mcf%d.dat' % (dir_name, args.output_filename.split('.')[0], args.nbins, args.nwalkers, args.nsteps, int(100*args.min_comp_frac)),
        #         overwrite=True, format='ascii.fixed_width_two_line')
        # print("Finished writing VeffLF file")
        return

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
        LFmod.triangle_plot('%s/triangle_%s_nb%d_nw%d_ns%d_mcf%d_ec_%d' % (dir_name, args.output_filename.split('.')[0], args.nbins, args.nwalkers, args.nsteps, int(100*args.min_comp_frac), args.err_corr), imgtype = args.output_dict['image format'])
        print("Finished making Triangle Plot with Best-fit LF (and V_eff-method-based data)")
    else:
        LFmod.set_median_fit()
        print("Finished setting median fit and V_eff parameters")
    names.append('Ln Prob')
    if args.output_dict['fitposterior']: 
        T = Table(LFmod.samples, names=names)
        T.write('%s/fitposterior_%s_nb%d_nw%d_ns%d_mcf%d_ec_%d.dat' % (dir_name, args.output_filename.split('.')[0], args.nbins, args.nwalkers, args.nsteps, int(100*args.min_comp_frac), args.err_corr),
                overwrite=True, format='ascii.fixed_width_two_line')
        print("Finished writing fitposterior file")
    if args.output_dict['bestfitLF']:
        T = Table([LFmod.lum, LFmod.lum_e, LFmod.medianLF],
                    names=['Luminosity', 'Luminosity_Err', 'MedianLF'])
        T.write('%s/bestfitLF_%s_nb%d_nw%d_ns%d_mcf%d_ec_%d.dat' % (dir_name, args.output_filename.split('.')[0], args.nbins, args.nwalkers, args.nsteps, int(100*args.min_comp_frac), args.err_corr),
                overwrite=True, format='ascii.fixed_width_two_line')
        print("Finished writing bestfitLF file")
    if args.output_dict['VeffLF']:
        T = Table([LFmod.Lavg, LFmod.lfbinorig, np.sqrt(LFmod.var)],
                    names=['Luminosity', 'BinLF', 'BinLFErr'])
        T.write('%s/VeffLF_%s_nb%d_nw%d_ns%d_mcf%d_ec_%d.dat' % (dir_name, args.output_filename.split('.')[0], args.nbins, args.nwalkers, args.nsteps, int(100*args.min_comp_frac), args.err_corr),
                overwrite=True, format='ascii.fixed_width_two_line')
        print("Finished writing VeffLF file")

    LFmod.table.add_row([args.line_name] + [0.]*(len(labels)-1))
    LFmod.add_fitinfo_to_table(percentiles)
    print(LFmod.table)

    if args.output_dict['parameters']:
        LFmod.table.write('%s/%s' %(dir_name, args.output_filename),
                          format='ascii.fixed_width_two_line',
                          formats=formats, overwrite=True)
        print("Finished writing LF main table")
    if args.output_dict['settings']:
        filename = open('%s/%s.args' %(dir_name, args.output_filename), 'w')
        del args.log
        filename.write( str( vars(args) ) )
        filename.close()
        print("Finished writing settings to file")

if __name__ == '__main__':
    main()
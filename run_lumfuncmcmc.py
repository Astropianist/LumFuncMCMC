import sys
import argparse as ap
import numpy as np
import os.path as op
import logging
from astropy.table import Table
from lumfuncmcmc import LumFuncMCMC
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

    # Initialize arguments and log
    args = parser.parse_args(args=argv)
    args.log = setup_logging()

    # Use config values if none are set in the input
    arg_inputs = ['nwalkers','nsteps','nbins','nboot','Flim','alpha','line_name','line_plot_name','Omega_0','sch_al','sch_al_lims','Lstar','Lstar_lims','phistar','phistar_lims','Lc','Lh',
    'param_percentiles', 'output_dict']

    for arg_i in arg_inputs:
        try:
            if getattr(args, arg_i) in [None, 0]:
                setattr(args, arg_i, getattr(configLF, arg_i))
        except AttributeError:
            setattr(args, arg_i, getattr(configLF, arg_i))

    return args

def read_input_file(args):
    datfile = Table.read(args.filename,format='ascii')
    z = datfile['z']
    try:
        flux = datfile['%s_flux'%(args.line_name)]
        cond = flux>0.0
        flux_e = datfile['%s_flux_e'%(args.line_name)]
        flux, flux_e = flux[cond], flux_e[cond]
    except:
        flux, flux_e = None, None
    try: 
        lum = datfile['%s_lum'%(args.line_name)]
        cond = lum>0.0
        lum = lum[cond]
    except: 
        lum = None
    try: 
        lum_e = datfile['%s_lum_e'%(args.line_name)]
        cond = lum_e>0.0
        lum_e = lum_e[cond]
    except: 
        lum_e = None
    z = z[cond]
    return z, flux, flux_e, lum, lum_e

def main(argv=None):

    # Make output folder if it doesn't exist
    mkpath('LFMCMCOut')
    # Get Inputs
    if argv == None:
        argv = sys.argv
        argv.remove('run_lumfuncmcmc.py')

    args = parse_args(argv)
    # Read input file into arrays
    z, flux, flux_e, lum, lum_e = read_input_file(args)
    print "Read Input File"

    # Initialize LumFuncMCMC class
    LFmod = LumFuncMCMC(z, flux=flux, flux_e=flux_e, lum=lum, lum_e=lum_e, 
                        Flim=args.Flim, alpha=args.alpha, line_name=args.line_name,
                        line_plot_name=args.line_plot_name, Omega_0=args.Omega_0,
                        nbins=args.nbins, nboot=args.nboot, sch_al=args.sch_al, 
                        sch_al_lims=args.sch_al_lims, Lstar=args.Lstar, 
                        Lstar_lims=args.Lstar_lims, phistar=args.phistar, 
                        phistar_lims=args.phistar_lims, Lc=args.Lc, Lh=args.Lh, 
                        nwalkers=args.nwalkers, nsteps=args.nsteps)
    print "Initialized LumFuncMCMC class"
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
    LFmod.table = Table(names=labels, dtype=['S10'] +
                              ['f8']*(len(labels)-1))
    print "Finished making names and labels for LF table and about to start fitting the model!"
    #### Run the actual model!!! ####
    LFmod.fit_model()
    print "Finished fitting model and about to create outputs"
    #### Get desired outputs ####
    if args.output_dict['triangle plot']:
        LFmod.triangle_plot('LFMCMCOut/triangle_%s_%d_%d' % (args.line_name, args.nbins, args.nboot), imgtype = args.output_dict['image format'])
        print "Finished making Triangle Plot with Best-fit LF (and V_eff-method-based data)"
    else:
        LFmod.set_median_fit()
        print "Finished setting median fit and V_eff parameters"
    names.append('Ln Prob')
    if args.output_dict['fitposterior']: 
        T = Table(LFmod.samples, names=names)
        T.write('LFMCMCOut/fitposterior_%s_%d_%d.dat' % (args.line_name, args.nbins, args.nboot),
                overwrite=True, format='ascii.fixed_width_two_line')
        print "Finished writing fitposterior file"
    if args.output_dict['bestfitLF']:
        T = Table([LFmod.lum, LFmod.lum_e, LFmod.medianLF],
                    names=['Luminosity', 'Luminosity_Err', 'MedianLF'])
        T.write('LFMCMCOut/bestfitLF_%s_%d_%d.dat' % (args.line_name, args.nbins, args.nboot),
                overwrite=True, format='ascii.fixed_width_two_line')
        print "Finished writing bestfitLF file"
    if args.output_dict['VeffLF']:
        T = Table([LFmod.Lavg, LFmod.lfbinorig, np.sqrt(LFmod.var)],
                    names=['Luminosity', 'BinLF', 'BinLFErr'])
        T.write('LFMCMCOut/VeffLF_%s_%d_%d.dat' % (args.line_name, args.nbins, args.nboot),
                overwrite=True, format='ascii.fixed_width_two_line')
        print "Finished writing VeffLF file"

    LFmod.table.add_row([args.line_name] + [0.]*(len(labels)-1))
    LFmod.add_fitinfo_to_table(percentiles)
    print(LFmod.table)

    if args.output_dict['parameters']:
        LFmod.table.write('LFMCMCOut/%s' % args.output_filename,
                          format='ascii.fixed_width_two_line',
                          formats=formats, overwrite=True)
        print "Finished writing LF main table"
    if args.output_dict['settings']:
        filename = open('LFMCMCOut/%s.args' % args.output_filename, 'w')
        del args.log
        filename.write( str( vars(args) ) )
        filename.close()
        print "Finished writing settings to file"

if __name__ == '__main__':
    main()
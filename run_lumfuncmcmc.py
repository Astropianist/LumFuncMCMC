import sys
import argparse as ap
import numpy as np
import os.path as op
import logging
from astropy.table import Table
from lumfuncmcmc import LumFuncMCMC
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

    # parser.add_argument("-o0", "--Omega_0",
    #                     help='''Effective survey area in square arcseconds''',
    #                     type=float, default=None)

    parser.add_argument("-mcf", "--min_comp_frac",
                        help='''Minimum completeness fraction considered''',
                        type=float, default=None)  

    parser.add_argument("-al", "--alpha",
                        help='''Minimum completeness fraction considered''',
                        type=float, default=None)

    parser.add_argument('-fl','--Flim', type=float, nargs='*', default=None, help="Flim for each of the five fields")    

    parser.add_argument("-sa", "--sch_al",
                        help='''Schechter Alpha Param''',
                        type=float, default=None)

    parser.add_argument("-fsa", "--fix_sch_al",
                        help='''Fix Schechter Alpha''',
                        action='count',default=0)

    parser.add_argument("-fc", "--fix_comp",
                        help='''Fix Completeness''',
                        action='count',default=0)

    parser.add_argument("-sr", "--same_rand",
                        help='''Same random starting point''',
                        action='count',default=0)

    parser.add_argument("-ln", "--line_name",
                         help='''Name of line or band for LF measurement''',
                         type=str, default=None)               

    # Initialize arguments and log
    args = parser.parse_args(args=argv)
    args.log = setup_logging()

    # Use config values if none are set in the input
    arg_inputs = ['nwalkers','nsteps','nbins','nboot','Flim','alpha','line_name','line_plot_name','Omega_0','sch_al','sch_al_lims','Lstar','Lstar_lims','phistar','phistar_lims','Lc','Lh','min_comp_frac', 'param_percentiles','output_dict','Flim_lims','alpha_lims','fcmin']

    for arg_i in arg_inputs:
        try:
            if getattr(args, arg_i) in [None, 0]:
                setattr(args, arg_i, getattr(configLF, arg_i))
        except AttributeError:
            setattr(args, arg_i, getattr(configLF, arg_i))

    if args.line_name=='OIII':
        args.line_plot_name = r'[OIII] $\lambda 5007$'
    if args.line_name=='Ha':
        args.line_plot_name = r'${\rm{H\alpha}}$'

    return args

def read_input_file(args, dust_fn=None):
    """ Function to read in input ascii file with properly named columns.
    Columns should include redshifts (header 'z') and a (linear) flux (header 
    'LineorBandName_flux') in 1.0e-17 erg/cm^2/s or log luminosity (header 
    'LineorBandName_lum') in log erg/s. Errors can be included with headers
    'LineorBandName_flux_e' or 'LineorBandName_lum_e', with the same units.

    Input
    -----
    args : class
        The args class is carried from function to function with information
        from command line input and config.py

    Return
    ------
    z: Numpy 1-D Array
        Source redshifts
    flux: Numpy 1-D Array
        Source fluxes (1.0e-17 erg/cm^2/s or None if not in input file)
    flux_e: Numpy 1-D Array
        Source flux errors (1.0e-17 erg/cm^2/s or None if not in input file)
    lum: Numpy 1-D Array
        Source log luminosities (log erg/s or None if not in input file)
    lum_e: Numpy 1-D Array
        Source log luminosity errors (log erg/s or oNone if not in input file)
    root: Float
        Minimum flux cutoff based on the completeness curve parameters and desired minimum completeness
    """
    
    datfile = Table.read(args.filename,format='ascii')
    fields, zfull = datfile['Field'], datfile['z']
    idfull = datfile['ID']
    field_names = np.unique(fields)
    field_ind = np.array([0])
    if abs(args.min_comp_frac-0.0)<1.0e-6:
        roots = np.zeros(len(field_names))
    else:
        roots = np.array([])
        for i in range(len(field_names)):
            root = fsolve(lambda x: V.fleming(x,args.Flim[i],args.alpha,args.fcmin)-args.min_comp_frac,[args.Flim[i]])[0]
            roots = np.append(roots,root)
    try:
        fluxfull = datfile['%s_flux'%(args.line_name)]
        fluxfull_e = datfile['%s_flux_e'%(args.line_name)]
        flux, flux_e = [], []
        for i,field in enumerate(field_names):
            fluxmin = roots[i]
            cond = np.logical_and(fields==field,fluxfull>fluxmin)
            flux.append(fluxfull[cond]); flux_e.append(fluxfull_e[cond])
            condlen = len(fluxfull[cond])
            field_ind = np.append(field_ind,field_ind[i]+condlen)
    except:
        flux, flux_e = None, None
    if '%s_lum'%(args.line_name) in datfile.columns: 
        lumfull = datfile['%s_lum'%(args.line_name)]
        if '%s_lum_e'%(args.line_name) in datfile.columns:
            lumfull_e = datfile['%s_lum'%(args.line_name)]
        lum, lum_e = [], []
        for field in field_names:
            cond = np.logical_and(fields==field,lumfull>0)
            lum.append(lumfull[cond])
            if lumfull_e is not None: lum_e.append(lumfull_e[cond])
            condlen = len(lumfull[cond])
            field_ind = np.append(field_ind,field_ind[i]+condlen)
        if len(lum_e)==0: lum_e = None
    else: 
        lum, lum_e = None, None
    z, ids, fds = [], [], []
    for i,field in enumerate(field_names):
        fluxmin = roots[i]
        try: cond = np.logical_and(fields==field,fluxfull>fluxmin)
        except: cond = np.logical_and(fields==field,lumfull>0.0)
        z.append(zfull[cond])
        ids.append(idfull[cond])
        fds.append(fields[cond])
    ids, fds = np.concatenate(ids), np.concatenate(fds)
    fdid = np.array([fd+str(id) for fd,id in zip(fds,ids)])
    if dust_fn is not None: 
        dustf = Table.read(dust_fn,format='ascii')
        fd_dust, id_dust = dustf['Field'], dustf['ID']
        fdid_dust = np.array([fd+str(id) for fd,id in zip(fd_dust,id_dust)])
        # ind_overlap = np.where(np.in1d(fdid_dust,fdid))
        ind_overlap = np.empty(len(fdid),dtype=int)
        for i in range(len(fdid)):
            ind_overlap[i] = np.where(fdid_dust==fdid[i])[0][0]
        assert np.all(fdid_dust[ind_overlap]==fdid)
        if args.line_name=='Ha': AHa, AHaerr = 2.07*2.66*dustf['E(B-V)'][ind_overlap], 2.07*2.66*dustf['E(B-V)err'][ind_overlap]
        else: AHa, AHaerr = 2.07*3.46*dustf['E(B-V)'][ind_overlap], 2.07*3.46*dustf['E(B-V)err'][ind_overlap]
        SFR_MC, SFR_MC_E = dustf['SFR100'][ind_overlap], dustf['SFR100err'][ind_overlap]
    else:
        AHa, AHaerr, SFR_MC, SFR_MC_E = None, None, None, None

    return z, flux, flux_e, lum, lum_e, field_names, field_ind, AHa, AHaerr, SFR_MC, SFR_MC_E

def main(argv=None):
    """ Read input file, run luminosity function routine, and create the appropriate output """
    # Make output folder if it doesn't exist
    mkpath('LFMCMCOut')
    # Get Inputs
    if argv == None:
        argv = sys.argv
        argv.remove('run_lumfuncmcmc.py')

    args = parse_args(argv)
    # Read input file into arrays
    z, flux, flux_e, lum, lum_e, field_names, field_ind, AHa, AHaerr, SFR_MC, SFR_MC_E = read_input_file(args, dust_fn='combined_all_Swift_NoDust_Donley_removed.dat')
    print("Read Input File")

    # Initialize LumFuncMCMC class
    LFmod = LumFuncMCMC(z, flux=flux, flux_e=flux_e, lum=lum, lum_e=lum_e, 
                        Flim=args.Flim, alpha=args.alpha, line_name=args.line_name,
                        line_plot_name=args.line_plot_name, Omega_0=args.Omega_0,
                        nbins=args.nbins, nboot=args.nboot, sch_al=args.sch_al, 
                        sch_al_lims=args.sch_al_lims, Lstar=args.Lstar, 
                        Lstar_lims=args.Lstar_lims, phistar=args.phistar, 
                        phistar_lims=args.phistar_lims, Lc=args.Lc, Lh=args.Lh, 
                        nwalkers=args.nwalkers, nsteps=args.nsteps, 
                        fix_sch_al=args.fix_sch_al, fix_comp=args.fix_comp, 
                        min_comp_frac=args.min_comp_frac, Flim_lims=args.Flim_lims,
                        alpha_lims=args.alpha_lims, field_names=field_names, field_ind=field_ind,
                        diff_rand=not args.same_rand)
    print("Initialized LumFuncMCMC class")

    # If the run has already been completed and there is a fitposterior file, don't bother with fitting everything again
    fn = 'LFMCMCOut/fitposterior_%s_nb%d_nw%d_ns%d_mcf%d.dat' % (args.output_filename.split('.')[0], args.nbins, args.nwalkers, args.nsteps, int(100*args.min_comp_frac))
    if op.isfile(fn):
        dat = Table.read(fn,format='ascii')
        LFmod.samples = np.lib.recfunctions.structured_to_unstructured(dat.as_array())
        LFmod.triangle_plot('LFMCMCOut/triangle_%s_nb%d_nw%d_ns%d_mcf%d' % (args.output_filename.split('.')[0], args.nbins, args.nwalkers, args.nsteps, int(100*args.min_comp_frac)), imgtype = args.output_dict['image format'])
        # T = Table([LFmod.Lavg, LFmod.lfbinorig, np.sqrt(LFmod.var)],
        #             names=['Luminosity', 'BinLF', 'BinLFErr'])
        # T.write('LFMCMCOut/VeffLF_%s_nb%d_nw%d_ns%d_mcf%d.dat' % (args.output_filename.split('.')[0], args.nbins, args.nwalkers, args.nsteps, int(100*args.min_comp_frac)),
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
        LFmod.triangle_plot('LFMCMCOut/triangle_%s_nb%d_nw%d_ns%d_mcf%d' % (args.output_filename.split('.')[0], args.nbins, args.nwalkers, args.nsteps, int(100*args.min_comp_frac)), imgtype = args.output_dict['image format'])
        print("Finished making Triangle Plot with Best-fit LF (and V_eff-method-based data)")
    else:
        LFmod.set_median_fit()
        print("Finished setting median fit and V_eff parameters")
    names.append('Ln Prob')
    if args.output_dict['fitposterior']: 
        T = Table(LFmod.samples, names=names)
        T.write('LFMCMCOut/fitposterior_%s_nb%d_nw%d_ns%d_mcf%d.dat' % (args.output_filename.split('.')[0], args.nbins, args.nwalkers, args.nsteps, int(100*args.min_comp_frac)),
                overwrite=True, format='ascii.fixed_width_two_line')
        print("Finished writing fitposterior file")
    if args.output_dict['bestfitLF']:
        T = Table([LFmod.lum, LFmod.lum_e, LFmod.medianLF],
                    names=['Luminosity', 'Luminosity_Err', 'MedianLF'])
        T.write('LFMCMCOut/bestfitLF_%s_nb%d_nw%d_ns%d_mcf%d.dat' % (args.output_filename.split('.')[0], args.nbins, args.nwalkers, args.nsteps, int(100*args.min_comp_frac)),
                overwrite=True, format='ascii.fixed_width_two_line')
        print("Finished writing bestfitLF file")
    if args.output_dict['VeffLF']:
        T = Table([LFmod.Lavg, LFmod.lfbinorig, np.sqrt(LFmod.var)],
                    names=['Luminosity', 'BinLF', 'BinLFErr'])
        T.write('LFMCMCOut/VeffLF_%s_nb%d_nw%d_ns%d_mcf%d.dat' % (args.output_filename.split('.')[0], args.nbins, args.nwalkers, args.nsteps, int(100*args.min_comp_frac)),
                overwrite=True, format='ascii.fixed_width_two_line')
        print("Finished writing VeffLF file")

    LFmod.table.add_row([args.line_name] + [0.]*(len(labels)-1))
    LFmod.add_fitinfo_to_table(percentiles)
    print(LFmod.table)

    if args.output_dict['parameters']:
        LFmod.table.write('LFMCMCOut/%s' % args.output_filename,
                          format='ascii.fixed_width_two_line',
                          formats=formats, overwrite=True)
        print("Finished writing LF main table")
    if args.output_dict['settings']:
        filename = open('LFMCMCOut/%s.args' % args.output_filename, 'w')
        del args.log
        filename.write( str( vars(args) ) )
        filename.close()
        print("Finished writing settings to file")

if __name__ == '__main__':
    main()
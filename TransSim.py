import numpy as np 
from uncertainties import unumpy, ufloat
import matplotlib.pyplot as plt 
from astropy.table import Table 
import argparse as ap
# from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
import lumfuncmcmc as L
from VmaxLumFunc import cosmo
import configLF as C
import os.path as op
from distutils.dir_util import mkpath
from astropy.table import Table
from itertools import cycle
import seaborn as sns
sns.set_context("paper",font_scale=1.3) # options include: talk, poster, paper
sns.set_style("ticks")
sns.set_style({"xtick.direction": "in","ytick.direction": "in",
               "xtick.top":True, "ytick.right":True,
               "xtick.major.size":12, "xtick.minor.size":4,
               "ytick.major.size":12, "ytick.minor.size":4,
               })
### color palettes
colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
colors += ["cloudy blue", "browny orange", "dark sea green"]
sns.set_palette(sns.xkcd_palette(colors))
orig_palette_arr = sns.color_palette()
orig_palette = cycle(tuple(orig_palette_arr))
markers = cycle(tuple(['o','^','*','s','+','v','<','>']))

def parse_args():
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
    parser = ap.ArgumentParser(description="TransSim",
                               formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument("-it", "--interp_type", help='''Method for interpolation''', type=str, default='cubic')
    parser.add_argument("-tf", "--filt_name", help='''Filter name''', type=str, default='N501')
    parser.add_argument("-dz", "--delz", help='''Width in redshift distribution''', type=float, default=0.1)
    parser.add_argument("-v", "--varying", help='''Vary the volume used in Veff''', action='count', default=0)
    parser.add_argument("-c", "--corrf", help='''Add correction for reverse experiment (kind of)''', action='count', default=0)
    parser.add_argument("-af", "--alpha_fixed", help='''Fixed alpha used in run that fit the Schechter curve used''', type=float, default=-1.6)
    parser.add_argument("-mcf", "--min_comp_frac", help='''Minimum completeness fraction considered''', type=float, default=1.0e-6)
    parser.add_argument("-Lc", "--Lc", help='''Lower value used for limiting luminosities''', type=float, default=40.0)
    # parser.add_argument("-nl", "--numlum", help='''Number of luminosities used for array for initial luminosity selection''', type=int, default=100000)
    parser.add_argument("-ng", "--numgal", help='''Number of galaxies selected for experiment''', type=int, default=100000)
    parser.add_argument("-bn", "--binnum", help='''Number of bins for Veff and correction''', type=int, default=20)
    args = parser.parse_args()
    args.field_name = 'COSMOS'
    args.interp_name = f'{args.field_name.lower()}_completeness_{args.filt_name.lower()}_grid_extrap.pickle'
    if args.filt_name=='N501': args.redshift, args.wav_filt, args.filt_width = 3.124, 5014.0, 77.17
    elif args.filt_name=='N419': args.redshift, args.wav_filt, args.filt_width = 2.449, 4193.0, 75.46
    else: args.redshift, args.wav_filt, args.filt_width = 4.552, 6750.0, 101.31
    args.trans_file = f'{args.filt_name}_Nicole.txt'
    return args

def add_LumFunc_plot(ax1):
    """ Set up the plot for the luminosity function """
    ax1.set_yscale('log')
    ax1.set_xlabel(r"$\log$ L (erg s$^{-1}$)")
    ax1.set_ylabel(r"$\phi_{\rm{true}}$ (Mpc$^{-3}$ dex$^{-1}$)")
    ax1.minorticks_on()

def plotVeffComp(logLs, lfs, vars, delz, alpha, minlum_use, lc, ngal, bn, image_dir=op.join('TransExp', 'VeffPlots')):
    mkpath(image_dir)
    fig, ax = plt.subplots()
    add_LumFunc_plot(ax)
    ax.errorbar(logLs, lfs[0], yerr=np.sqrt(vars[0]), fmt='bs', linestyle='none', label='Original')
    ax.errorbar(logLs, lfs[1], yerr=np.sqrt(vars[1]), fmt='r^', linestyle='none', label='Convolved')
    ax.legend(loc='best', frameon=False)
    fig.savefig(op.join(image_dir, f'VeffComp_ng{ngal}_bn{bn}_al{alpha}_delz{delz}_ml{minlum_use:0.2f}_Lc{lc}.png'), bbox_inches='tight', dpi=200)
    plt.close('all')

def plotTransCurve(file_name='N501_with_atm.txt', image_dir='TransExp'):
    trans_dat = Table.read(file_name, format='ascii')
    lam, tra = trans_dat['lambda'], trans_dat['transmission']
    fig, ax = plt.subplots()
    ax.plot(lam, tra, 'b-')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel('Tranmission')
    pn = file_name.split('.')[0]
    fig.savefig(op.join(image_dir,f'TransCurve_{pn}.png'), bbox_inches='tight', dpi=150)

def get1DComp(interp_comp, maghigh=19., maglow=30., magnum=25, distnum=100):
    maggrid = np.linspace(maghigh, maglow, magnum)
    R = np.sqrt(C.Omega_0_sqarcmin/np.pi)
    dists = R * np.sqrt(np.random.rand(distnum))
    distgrid = np.sort(dists)
    distg, magg = np.meshgrid(distgrid, maggrid, indexing='ij')
    comps = interp_comp.ev(distg, magg)
    # comps = comps.reshape(distnum, magnum)
    comp_avg_dist = np.average(comps, axis=0)
    comps1df = interp1d(maggrid, comp_avg_dist, bounds_error=False, fill_value=(comp_avg_dist[0], 0))
    return comps1df

def calc_mags(logL, dL, wav_filt, filt_width):
    lum = 10**logL
    flux_cgs = lum/(4.0*np.pi*(3.086e24*dL)**2)
    mags = L.cgs2magAB(flux_cgs, wav_filt, filt_width)
    return mags

def calc_lum(mag, dL, wav_filt, filt_width):
    flux = L.magAB2cgs(mag, wav_filt, filt_width)
    lum = flux * (4.0*np.pi*(3.086e24*dL)**2)
    return np.log10(lum)

def select_gal(args, al, ls, phis, zmin, zmax, interp_comp, numgal=1000000, numlum=1000000, Lc=40.0, Lh=44.0, zc=3.125, maglow=30.0, corrf=None):
    # zmin, zmax = (wavmin - C.wav_rest) / C.wav_rest, (wavmax - C.wav_rest) / C.wav_rest
    reds = np.random.uniform(zmin, zmax, numgal)
    logL = np.random.uniform(Lc, Lh, numlum)
    tlf = L.TrueLumFunc(logL, al, ls, phis)
    comps1df = get1DComp(interp_comp, maglow=maglow)
    dL = cosmo.luminosity_distance(zc).value
    mags = calc_mags(logL, dL, args.wav_filt, args.filt_width)
    tlf *= comps1df(mags)
    if corrf is not None: 
        corrs = corrf(logL)
        tlf *= 10**corrs
    lums = np.random.choice(logL, size=numgal, p=tlf/tlf.sum())
    # breakpoint()
    return reds, lums, comps1df, dL

def calc_new_lums(lums, reds, file_name='N501_with_atm.txt', interp_type='cubic', size_lprime=51):
    logLfuncz, delzfv2, _ = L.getRealLumRed(file_name, interp_type, C.wav_rest)
    _, _, delzf = L.getBoundsTransPDF(logL_width=8.0,wav_rest=C.wav_rest,num_discrete=size_lprime,file_name=file_name)
    logLs = logLfuncz(reds)
    # mags = calc_mags(logLs, dL)
    # comp = comps1df(mags)
    # logLs_use = np.random.choice(logLs, logLs.size, p=comp/comp.sum())
    # _, _, delzf = L.getBoundsTransPDF(logLs.max(), file_name=file_name)
    return lums - logLs, logLs, delzf, delzfv2

def bin_lums(lums, binnum=10, minlum=41.5, maxlum=43.5):
    bin_edges = np.linspace(minlum, maxlum, binnum+1)
    bin_centers = np.array([(bin_edges[i]+bin_edges[i+1])/2.0 for i in range(binnum)])
    hist, _ = np.histogram(lums, bin_edges)
    return bin_centers, hist

def plot_hists(lums, lums_mod, delz, al, logL, bins=50, image_dir='TransExp'):
    fig, ax = plt.subplots(ncols=2)
    ax[0].hist(lums, bins=bins, color='r', alpha=0.5, density=True, label='Drawn from TLF')
    ax[0].hist(lums_mod, bins=bins, color='b', alpha=0.5, density=True, label='Convolved')
    ax[1].hist(logL, bins=bins, density=True, color='b')
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[0].set_xlabel('Log(L)')
    ax[1].set_xlabel(r'$\Delta$Log(L)')
    ax[0].set_ylabel('PDF')
    # ax[1].set_ylabel('PDF')
    ax[0].legend(loc='best', frameon=False)
    plt.tight_layout()
    fig.savefig(op.join(image_dir,f'LumTransEff_delz{delz}_al{al}.png'), bbox_inches='tight', dpi=200)
    plt.close('all')

def get_corrections(args, al, ls, phis, Lc=40.0, Lh=44.0, minlumorig=41.5, varying=0, image_dir='TransExp', maglow=30.0, corrf=None):
    delz, file_name, numgal, numlum, binnum, min_comp_frac, interp_type = args.delz, args.trans_file, args.numgal, args.numgal, args.binnum, args.min_comp_frac, args.interp_type
    minlum = max(Lc, minlumorig)
    interp_comp, interp_comp_simp = L.makeCompFunc(file_name=args.interp_name)
    R = np.sqrt(C.Omega_0_sqarcmin/np.pi)
    dists = R * np.sqrt(np.random.rand(numlum))
    zcent = (args.wav_filt - C.wav_rest) / C.wav_rest
    zmin, zmax = zcent - delz, zcent + delz
    reds, lums, comps1df, dL = select_gal(args, al, ls, phis, zmin, zmax, interp_comp_simp, numgal=numgal, numlum=numlum, Lc=Lc, Lh=Lh, maglow=maglow, corrf=corrf, zc=args.redshift)
    maxlum = lums.max()
    # dL_full = cosmo.luminosity_distance(reds).value
    # minlums_accept = calc_lum(maglow, dL_full)
    minlum_onered = calc_lum(maglow, dL, args.wav_filt, args.filt_width)
    # bin_centers_orig, hist_orig = bin_lums(lums)
    lums_mod, logLs, delzf, delzfv2 = calc_new_lums(lums, reds, file_name=file_name, interp_type=interp_type)
    # bin_centers, hist = bin_lums(lums_mod)
    plot_hists(lums, lums_mod, delz, al, logLs)
    mkpath(image_dir)
    # outname_list = [f'Veff_al{al}_delz{delz}_vary{varying}', f'VeffTrans_al{al}_delz{delz}_vary{varying}']
    # minlum_use = max(Lc, minlum_onered)
    minlum_use = minlum_onered
    print("minlum_use:", minlum_use)
    delz_eff = [np.average(delzf(lums-minlum_use)), np.average(delzf(lums_mod-minlum_use))]
    delz_effv2 = [np.average(delzfv2(lums-minlum_use)), np.average(delzfv2(lums_mod-minlum_use))]
    # delz_use = [C.del_red, 2*delz]
    print("Delz_eff:", delz_eff)
    print("Delz_effv2:", delz_effv2)
    lumlist = [lums, lums_mod]
    lf, vars = [], []
    for i, lumi in enumerate(lumlist):
        lumobj = L.LumFuncMCMC(args.redshift, del_red=delz_eff[i], lum=lumi, Omega_0=C.Omega_0, sch_al=al, Lstar=ls, phistar=phis, fix_sch_al=True, min_comp_frac=min_comp_frac, dist_orig=dists, dist=dists, logL_width=logLs.max(), transsim=True, minlum=minlum, maxlum=maxlum, nbins=binnum, interp_comp=interp_comp, interp_comp_simp=interp_comp_simp)
        lumobj.VeffLF(varying=varying)
        lf.append(lumobj.lfbinorig)
        vars.append(lumobj.var)
    plotVeffComp(lumobj.Lavg, lf, vars, delz, al, minlum_use, Lc, numgal, binnum)
    lf0 = unumpy.uarray(lf[0], np.sqrt(vars[0]))
    lf1 = unumpy.uarray(lf[1], np.sqrt(vars[1]))
    print("vars[0]:", vars[0])
    print("vars[1]:", vars[1])
    corr = unumpy.uarray([np.nan]*binnum, [np.nan*binnum])
    cond_corr = np.logical_and(lf[0]>0, lf[1]>0)
    corr[cond_corr] = unumpy.log10(lf0[cond_corr]/lf1[cond_corr])
    # print("corr:", corr)
    # assert np.all(bin_centers == bin_centers_orig), breakpoint()
    # corr = np.log10(hist_orig/hist)
    # breakpoint()
    # return bin_centers, corr
    # breakpoint()
    return lumobj.Lavg, corr, minlum_use

def plot_corr(bin_centers, corr, plotname, image_dir='TransExp', corre=None, lcs=None, bcs=None, corrfull=None, correfull=None):
    mkpath(image_dir)
    fig, ax = plt.subplots()
    if corrfull is not None: 
        col = next(orig_palette)
        ax.plot(bcs, corrfull, color=col, linestyle='--', marker='none', label='Overall')
        ax.fill_between(bcs, corrfull-correfull, corrfull+correfull, color=col, alpha=0.2, label='')
    if type(bin_centers)==list:
        for bc, co, coe, lc in zip(bin_centers, corr, corre, lcs):
            ax.errorbar(bc, co, coe, color=next(orig_palette), marker=next(markers), label=f'Lower limit: {lc}')
        ax.legend(loc='best', frameon=False)
    else: ax.errorbar(bin_centers, unumpy.nominal_values(corr), yerr=unumpy.std_devs(corr), fmt='b-*')
    ax.set_xlabel('Log Luminosity (erg/s)')
    ax.set_ylabel('Log Correction (True/Obs)')
    fig.savefig(op.join(image_dir, plotname), bbox_inches='tight', dpi=300)
    plt.close('all')

def getOverallCorr(bcall, corrall, correall, num=1001):
    corrfs, correfs = [], []
    corrs, corres = np.zeros((len(bcall), num)), np.zeros((len(bcall), num))
    bcmin, bcmax = np.inf, -np.inf
    for i in range(len(bcall)):
        cond = ~np.isnan(corrall[i])
        bcmin = min(bcmin, bcall[i][cond].min())
        bcmax = max(bcmax, bcall[i][cond].max())
    bcs = np.linspace(bcmin, bcmax, num)
    for i in range(len(bcall)):
        corrfs.append(interp1d(bcall[i], corrall[i], kind='cubic', bounds_error=False, fill_value=np.nan))
        correfs.append(interp1d(bcall[i], correall[i], kind='cubic', bounds_error=False, fill_value=np.nan))
        corrs[i] = corrfs[i](bcs)
        corres[i] = correfs[i](bcs)
    ws = 1/corres
    corrfull = np.nansum(corrs*ws, axis=0) / np.nansum(ws, axis=0)
    correfull = np.sqrt(len(bcall)) / np.nansum(ws, axis=0)

    return bcs, corrfull, correfull

def showAllCorr(filter='N501', ngal=2500000):
    if filter=='N501': alpha, ml = -1.6, 40.48 
    elif filter=='N419': alpha, ml = -2.0, 40.37
    else: alpha, ml = -1.2, 40.73
    image_dir = op.join('TransExp', str(ngal))
    Lcvals = [40.0, 41.0, 42.0, 42.5, 42.8, 43.1]
    fn_base = f'{filter}Corr_ng{ngal}_bn20_al{alpha:0.1f}_delz0.1_ml{ml:0.2f}'
    fn_base43 = f'{filter}Corr_ng{ngal}_bn8_al{alpha:0.1f}_delz0.1_ml{ml:0.2f}'
    bcall, corrall, correall = [], [], []
    for Lc in Lcvals:
        # if Lc < 40.9: fn = op.join(image_dir, fn_base+'.dat')
        if Lc>43: fn = op.join(image_dir, f'{fn_base43}_Lc{Lc:0.1f}_corr0.dat')
        else: fn = op.join(image_dir, f'{fn_base}_Lc{Lc:0.1f}_corr0.dat')
        dat = Table.read(fn, format='ascii')
        bc, co, coe = dat['logL'], dat['Corr'], dat['CorrErr']
        bcall.append(bc); corrall.append(co); correall.append(coe)
    bcs, corrfull, correfull = getOverallCorr(bcall, corrall, correall)
    corrdat = Table()
    corrdat['logL'] = bcs
    corrdat['Corr'] = corrfull
    corrdat['CorrErr'] = correfull
    corrdat.write(op.join(image_dir, f'CorrFull{filter}.dat'), format='ascii')
    plot_corr(bcall, corrall, plotname=f'MixCorrsOverall{filter}.png', image_dir=image_dir, corre=correall, lcs=Lcvals, bcs=bcs, corrfull=corrfull, correfull=correfull)

def main():
    args = parse_args()
    filter = args.filt_name
    image_dir = 'TransExp'
    mkpath(image_dir)
    alpha_fixed, delz, varying, Lc, numgal, binnum = args.alpha_fixed, args.delz, args.varying, args.Lc, args.numgal, args.binnum
    if filter=='N501':
        if alpha_fixed==-1.6: this_work = [alpha_fixed, 42.47, -2.74]
        else: 
            print("Not one of the sanctioned alpha fixed values")
            return
    elif filter=='N419':
        if alpha_fixed==-2.0: this_work = [alpha_fixed, 42.56, -3.19]
        else: 
            print("Not one of the sanctioned alpha fixed values")
            return
    else:
        if alpha_fixed==-1.1: this_work = [alpha_fixed, 42.53, -2.71]
        else: 
            print("Not one of the sanctioned alpha fixed values")
            return
    # plotTransCurve('N673_Nicole.txt', image_dir='')
    # plotTransCurve('N419_Nicole.txt', image_dir='')
    # bin_centers, corr_perf = get_corrections(*this_work, delz=0.0317, file_name=perf_filt)
    # plot_corr(bin_centers, corr_perf, f'TopHatCorr_al{alpha_fixed}.png')
    if args.corrf:
        corrfile = Table.read(op.join(image_dir, 'CorrFull.dat'), format='ascii')
        logL, corr = corrfile['logL'], corrfile['Corr']
        corrf = interp1d(logL, corr, kind='linear', bounds_error=False, fill_value=(corr[0], corr[-1]))
    else: 
        corrf = None
    bin_centers, corr_n501, minlum_use = get_corrections(args, *this_work, varying=varying, Lc=Lc, corrf=corrf)
    plot_corr(bin_centers, corr_n501, f'{filter}CorrVeff_ng{numgal}_bn{binnum}_al{alpha_fixed}_delz{delz}_ml{minlum_use:0.2f}_Lc{Lc}_corr{args.corrf}.png', image_dir=image_dir)

    # Write corrections to a file
    dat = Table()
    dat['logL'], dat['Corr'], dat['CorrErr'] = bin_centers, unumpy.nominal_values(corr_n501), unumpy.std_devs(corr_n501)
    dat.write(op.join(image_dir, f'{filter}Corr_ng{numgal}_bn{binnum}_al{alpha_fixed}_delz{delz}_ml{minlum_use:0.2f}_Lc{Lc}_corr{args.corrf}.dat'), format='ascii')

if __name__ == '__main__':
    # main()
    showAllCorr('N673')
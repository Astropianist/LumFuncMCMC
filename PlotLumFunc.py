import numpy as np 
from scipy.integrate import quad
from uncertainties import unumpy, ufloat
import matplotlib.pyplot as plt 
from astropy.table import Table 
from astropy.io import fits
import os.path as op
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
colors_overall = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
colors_overall += ["cloudy blue", "browny orange", "dark sea green"]
sns.set_palette(sns.xkcd_palette(colors_overall))
orig_palette_arr = sns.color_palette()
orig_palette = cycle(tuple(orig_palette_arr))
markers_overall = ['o','^','*','s','+','v','<','>', '1', '8', 'P']
markers = cycle(tuple(markers_overall))

Lsun = 3.8e33

def plotLLComp(orig_file, ll_min=42.8, ll_max=44.0, num_files=13, sm=3, sM=30):
    lls = np.linspace(ll_min, ll_max, num_files)
    fig, ax = plt.subplots()
    al, ls, phis = [], [], []
    for ll in lls:
        file_name = orig_file.replace('43.2', f'{ll:0.1f}')
        dat = Table.read(file_name, format='ascii')
        al.append(dat[r'$\alpha$_50'])
        ls.append(dat[r'$\log L_*$_50'])
        phis.append(dat[r'$\log \phi_*$_50'])
        del dat
    al, ls, phis = np.array(al), np.array(ls), np.array(phis)
    sizes = (sM-sm)/(phis.max() - phis.min()) * (phis-phis.min()) + sm
    sc = ax.scatter(al, ls, c=lls, s=sizes)
    plt.colorbar(sc, label='Upper luminosity cutoff')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\log L_*$')
    fig.savefig('N673_LLComp.png', bbox_inches='tight', dpi=300)

def getIntegs(al, ls, phis, llow=42.0, lhigh=46.0):
    ans_num, _ = quad(lambda x: TrueLumFunc(x, al, ls, phis), llow, lhigh)
    ans_lum, _ = quad(lambda x: 10**x * TrueLumFunc(x, al, ls, phis), llow, lhigh)
    return ans_num, ans_lum

def getIntegsv2(al, ls, phis, llow=42.0, lhigh=46.0):
    ans_num, _ = quad(lambda x: schechter(x, al, 10**phis, 10**ls), 10**llow, 10**lhigh)
    ans_lum, _ = quad(lambda x: x * schechter(x, al, 10**phis, 10**ls), 10**llow, 10**lhigh)
    return ans_num, ans_lum

def getIntegInfo(fitpost, rndsamples=100, llow=42.0, lhigh=46.0, sa=-1.6):
    dat = Table.read(fitpost,format='ascii')
    samples = np.lib.recfunctions.structured_to_unstructured(dat.as_array())
    del dat
    nsamples = getnsamples(samples)
    ind = np.random.randint(0, nsamples.shape[0], rndsamples)
    nums, lums = np.zeros(rndsamples), np.zeros(rndsamples)
    if nsamples.shape[1]==4: alpha = nsamples[:,2]
    else: alpha = np.repeat(sa, nsamples.shape[0])
    for i, indi in enumerate(ind):
        nums[i], lums[i] = getIntegsv2(alpha[indi], nsamples[indi, 0], nsamples[indi, 1], llow=llow, lhigh=lhigh)
    lums /= Lsun
    numvals = np.percentile(np.log10(nums), [16,50,84])
    lumvals = np.percentile(np.log10(lums), [16,50,84])
    print("[16th, 50th, 84th] percentile of log integral over number of galaxies: ", numvals)
    print("[16th, 50th, 84th] percentile of log integral over luminosity of galaxies: ", lumvals)
    print("Log Number: upper and lower errors:", numvals[2]-numvals[1], numvals[1]-numvals[0])
    print("Log Luminosity: upper and lower errors:", lumvals[2]-lumvals[1], lumvals[1]-lumvals[0])

def add_LumFunc_plot(ax1, no_ylabel=False):
    """ Set up the plot for the luminosity function """
    ax1.set_yscale('log')
    ax1.set_xlabel(r"$\log$ L (erg s$^{-1}$)")
    if not no_ylabel: ax1.set_ylabel(r"$\phi_{\rm{true}}$ (Mpc$^{-3}$ dex$^{-1}$)")
    ax1.minorticks_on()

def getSamples(logL, nsamples, rndsamples=200, sa=-1.6):
    lf = []
    if nsamples.shape[1]==4: alpha = nsamples[:,2]
    else: alpha = np.repeat(sa, nsamples.shape[0])
    for i in np.arange(rndsamples):
        ind = np.random.randint(0, nsamples.shape[0])
        modlum = TrueLumFunc(logL, alpha[ind], nsamples[ind, 0], nsamples[ind, 1])
        lf.append(modlum)
    medianLF = np.median(np.array(lf), axis=0)
    return lf, medianLF

def getnsamples(samples, lnprobcut=7.5):
    nsamples = []
    while len(nsamples)<len(samples)//4: 
        chi2sel = (samples[:, -1] >
                (np.max(samples[:, -1], axis=0) - lnprobcut))
        nsamples = samples[chi2sel, :]
        lnprobcut *= 2.0
    return nsamples

def plotProtoEvol(fitpostprotorig, reds, Lmin=42.0, Lmax=43.5, Lnum=1001, sa=-1.6):
    fppo = [fpp.split('/') for fpp in fitpostprotorig]
    fitpostprot = [op.join(fpp[0], fpp[1], '2', fpp[2].replace('env0', 'env2').replace('_all_', '_pc_')) for fpp in fppo]
    fitpostnotprot = [fp.replace('bin1', 'bin2') for fp in fitpostprot]
    logL = np.linspace(Lmin, Lmax, Lnum)
    samples_prot, samples_notprot = [], []
    for fpf, fpnf in zip(fitpostprot, fitpostnotprot):
        dat = Table.read(fpf, format='ascii')
        dat2 = Table.read(fpnf, format='ascii')
        samples_prot.append(np.lib.recfunctions.structured_to_unstructured(dat.as_array()))
        samples_notprot.append(np.lib.recfunctions.structured_to_unstructured(dat2.as_array()))
        del dat, dat2
    fig, ax = plt.subplots()
    add_LumFunc_plot(ax)
    for i, z in enumerate(reds):
        coli = next(orig_palette)
        nsamples_prot = getnsamples(samples_prot[i])
        nsamples_notprot = getnsamples(samples_notprot[i])
        lfp, lfpbest = getSamples(logL, nsamples_prot, sa=sa)
        lfnp, lfnpbest = getSamples(logL, nsamples_notprot, sa=sa)
        ax.plot(logL, lfpbest, linestyle='-', color=coli, label=rf'$z={z}$ PC')
        ax.plot(logL, lfnpbest, linestyle=':', color=coli, label=rf'$z={z}$ Not PC')
        for lfpi, lfnpi in zip(lfp, lfnp):
            ax.plot(logL, lfpi, linestyle='-', color=coli, alpha=0.05, label='')
            ax.plot(logL, lfnpi, linestyle=':', color=coli, alpha=0.05, label='')
    ax.set_xlim(Lmin, Lmax)
    ax.set_ylim(1.0e-6, 3.0e-2)
    ax.legend(loc='best', frameon=False)
    fig.savefig("CosmicEvolCOSMOS_PCv2.png", bbox_inches='tight', dpi=300)

def plotEvolution(fitpostfs, reds, Lmin=42.0, Lmax=43.5, Lnum=1001, sa=-1.6):
    logL = np.linspace(Lmin, Lmax, Lnum)
    samples = []
    for fpf in fitpostfs:
        dat = Table.read(fpf,format='ascii')
        samples.append(np.lib.recfunctions.structured_to_unstructured(dat.as_array()))
        del dat
    fig, ax = plt.subplots()
    add_LumFunc_plot(ax)
    for i, z in enumerate(reds):
        coli = next(orig_palette)
        nsamples = getnsamples(samples[i])
        lf, lfbest = getSamples(logL, nsamples, sa=sa)
        ax.plot(logL, lfbest, linestyle='-', color=coli, label=rf'$z={z}$')
        for lfi in lf:
            ax.plot(logL, lfi, linestyle='-', color=coli, alpha=0.05, label='')
    ax.set_xlim(Lmin, Lmax)
    ax.set_ylim(1.0e-6, 3.0e-2)
    ax.legend(loc='best', frameon=False)
    fig.savefig("CosmicEvolCOSMOS_newcontam.png", bbox_inches='tight', dpi=300)

def plotDensityEvol(fit_names_orig, reds, dens_vals, Lmin=42.0, Lmax=43.5, Lnum=1001, ymin=1.0e-6, ymax=3.0e-2, sa=-1.6):
    ns = len(dens_vals[0])-1
    fno = [fn.split('/') for fn in fit_names_orig]
    fit_names = [op.join(fn[0], fn[1], str(ns), fn[2].replace('env0', 'env1').replace('_all_', f'_e{ns}_')) for fn in fno]
    logL = np.linspace(Lmin, Lmax, Lnum)
    fitpostall = []
    # ns = int(fit_names[0].split('/')[2])
    cols = []
    for i in range(len(reds)):
        assert len(dens_vals[i]) == ns + 1
        fiti = []
        for j in range(ns):
            if i==0: cols.append(next(orig_palette))
            fitf = fit_names[i].replace('bin1', f'bin{j+1}')
            dat = Table.read(fitf,format='ascii')
            samples = np.lib.recfunctions.structured_to_unstructured(dat.as_array())
            fiti.append(getnsamples(samples))
            del dat
        fitpostall.append(fiti)
    fig, ax = plt.subplots(nrows=1, ncols=len(reds), sharex=True, sharey=True, figsize=(12, 4))
    for i, z in enumerate(reds):
        if i==0: no_ylabel=False
        else: no_ylabel=True
        add_LumFunc_plot(ax[i], no_ylabel=no_ylabel)
        for j in range(ns):
            lf, lfbest = getSamples(logL, fitpostall[i][j], sa=sa)
            ax[i].plot(logL, lfbest, linestyle='-', color=cols[j], label=fr'{dens_vals[i][j]:0.2f} $\leq \sigma <$ {dens_vals[i][j+1]:0.2f}')
            for lfi in lf:
                ax[i].plot(logL, lfi, linestyle='-', color=cols[j], alpha=0.05, label='')
        ax[i].text(0.5, 0.98, fr'$z={z}$', horizontalalignment='center', verticalalignment='top', transform=ax[i].transAxes)
        ax[i].legend(loc='best', frameon=False, fontsize='small')
    ax[0].set_xlim(Lmin, Lmax)
    ax[0].set_ylim(ymin, ymax)
    plt.tight_layout()
    
    fig.savefig("CosmicDensEvolCOSMOS_newcontam_3bins.png", bbox_inches='tight', dpi=300)

def calc_phi_err(phi, logphierr):
    return np.log(10) * phi * logphierr

def plotStuffNew(fitpostfs, reds, sobfile='sty378_supp/SC4K_full_LFs_Table_C1.fits', sobothers='sty378_supp/SSC4K_compilation_Table_C2.fits', Lmin=42.0, Lmax=43.5, Lnum=1001, sobkeys=['IA427 ($z=2.5$)', 'IA505 ($z=3.2$)', 'IA679 ($z=4.6$)'], sobzs=[2.5, 3.2, 4.6], maxdiff=0.21, llims=[43.1, 43.1, 43.2], ymin=1.0e-6, ymax=3.0e-2, sa=-1.6):
    logL = np.linspace(Lmin, Lmax, Lnum)
    sob = fits.getdata(sobfile, 1)
    sobs = sob['Sample']
    logLsob, logLsobe, phisob = sob['log_Lum_bin'], sob['delta_bin'], 10**sob['Phi_final']
    phiseu, phisel = calc_phi_err(phisob, sob['Phi_final_err_up']), calc_phi_err(phisob, sob['Phi_final_err_down'])
    sobo = fits.getdata(sobothers, 1)
    zavg = (sobo['z_min'] + sobo['z_max']) / 2
    ref, logLso, logLsoe, phiso = sobo['Reference'], sobo['LogL'], sobo['D_LogL'], 10**sobo['LogPhi']
    phisoeu, phisoel = calc_phi_err(phiso, sobo['D_LogPhi_up']), calc_phi_err(phiso, sobo['D_LogPhi_down'])
    refuniq = np.unique(ref)
    colref, markref = [], []
    for refi in refuniq:
        colref.append(next(orig_palette))
        markref.append(next(markers))
    samples = []
    for fpf in fitpostfs:
        dat = Table.read(fpf,format='ascii')
        samples.append(np.lib.recfunctions.structured_to_unstructured(dat.as_array()))
        del dat
    fig, ax = plt.subplots(nrows=1, ncols=len(reds), sharex=True, sharey=True, figsize=(12, 4))
    for i, z in enumerate(reds):
        if i==0: no_ylabel=False
        else: no_ylabel=True
        add_LumFunc_plot(ax[i], no_ylabel=no_ylabel)
        coli = next(orig_palette)
        nsamples = getnsamples(samples[i])
        lf, lfbest = getSamples(logL, nsamples, sa=sa)
        ax[i].plot(logL, lfbest, linestyle='-', color=coli, label=rf'Nagaraj+24 $z={z}$')
        for lfi in lf:
            ax[i].plot(logL, lfi, linestyle='-', color=coli, alpha=0.05, label='')
        condsob = sobs == sobkeys[i]
        ax[i].errorbar(logLsob[condsob], phisob[condsob], yerr=np.row_stack((phisel[condsob], phiseu[condsob])), xerr=logLsobe[condsob]/2, linestyle='none', marker=markers_overall[0], color='k', label='Sobral+2018', capsize=2)
        condsobo = abs(zavg-z)<maxdiff
        refsj = np.unique(ref[condsobo])
        for rj in refsj:
            if rj=='Konno+2016': continue
            if rj=='Konno+2016_Sobral+2017': rjlab = r'Konno+2016$^{\rm CC}$'
            else: rjlab = rj
            ind = np.where(refuniq==rj)[0][0]
            condsofull = np.logical_and(condsobo, ref==rj)
            ax[i].errorbar(logLso[condsofull], phiso[condsofull], yerr=np.row_stack((phisoel[condsofull], phisoeu[condsofull])), xerr=logLsoe[condsofull]/2, linestyle='none', marker=markref[ind], color=colref[ind], label=rjlab, capsize=2)
        # ax[i].vlines(llims[i], ymin, ymax, colors='k', label='')
        # Now using contamination algorithm so don't need a 
        condvi = logL>=llims[i]
        if i==2: ax[i].fill_between(logL[condvi], ymin*np.ones_like(logL[condvi]), ymax*np.ones_like(logL[condvi]), color='k', alpha=0.1, label='')
        ax[i].legend(loc='best', frameon=False, fontsize='x-small')
    ax[0].set_xlim(Lmin, Lmax)
    ax[0].set_ylim(ymin, ymax)
    plt.tight_layout()
    
    fig.savefig("FullLitCompv2.png", bbox_inches='tight', dpi=300)

def plotStuff(logLV, lfV, lfeV, logL, bflf, this_work=None, sobral1=None, sobral2=None):
    fig, ax = plt.subplots()
    add_LumFunc_plot(ax)
    if bflf is None: veff_label = 'This work'
    else: veff_label = r'V$_{\rm eff}$'
    ax.errorbar(logLV, lfV, yerr=lfeV, fmt='b^', label=veff_label)
    if bflf is not None:
        ax.plot(logL, bflf, 'r-', label=fr'Median MCMC Fit')
    if this_work is not None:
        lf = TrueLumFunc(logL, this_work['alpha'], this_work['logLstar'], this_work['logphistar'])
        ax.plot(logL, lf, 'k-', label=this_work['label'])
    for arg, col in zip([sobral1, sobral2], ['orange', 'gray']): 
        # breakpoint()
        lf = TrueLumFunc(logL, arg['alpha'], arg['logLstar'], arg['logphistar'])
        if type(lf[0])==float: ax.plot(logL, lf, color=col, label=arg['label'])
        else: 
            ax.plot(logL, unumpy.nominal_values(lf), color=col, label=arg['label'])
            ax.fill_between(logL, unumpy.nominal_values(lf) - unumpy.std_devs(lf), unumpy.nominal_values(lf) + unumpy.std_devs(lf), color=col, alpha=0.2)
    ax.legend(loc='best', frameon=False)
    fig.savefig(f'LumFuncCompN501.png', bbox_inches='tight', dpi=300)

def TrueLumFunc(logL,alpha,logLstar,logphistar):
    ''' Calculate true luminosity function (Schechter form)

    Input
    -----
    logL : float or numpy 1-D array
        Value or array of log luminosities in erg/s
    alpha: float
        Schechther alpha parameter
    logLstar: float
        Schechther log(Lstar) parameter
    logphistar: float
        Schechter log(phistar) parameter

    Returns
    -------
    Phi(logL,z) : Float or 1-D array (same size as logL and/or z)
        Value or array giving luminosity function in Mpc^-3/dex
    '''
    if type(alpha)==float: return np.log(10.0) * 10**logphistar * 10**((logL-logLstar)*(alpha+1))*np.exp(-10**(logL-logLstar))
    else: return unumpy.log(10.0) * 10**logphistar * 10**((logL-logLstar)*(alpha+1))*unumpy.exp(-10**(logL-logLstar))

def schechter(L, al, phistar, Lstar):
    """ Schechter function """
    return phistar/Lstar * (L/Lstar)**al * np.exp(-L/Lstar)

def main(alpha_fixed=-1.6):
    run_use = f'ODIN_fsa0_sa{alpha_fixed:0.2f}_mcf50_ll43.1_ec2'
    prefix = 'final_ll_431_all_'
    postfix = '_nb50_nw200_ns5000_mcf50_ec_2_env0_bin1'
    dir_use = op.join('LFMCMCOdin', run_use)
    dat_veff = Table.read(op.join(dir_use, prefix + 'VeffLF_' + run_use + postfix + '_c1.dat'), format='ascii')
    dat_bf = Table.read(op.join(dir_use, prefix + 'bestfitLF_' + run_use + postfix + '.dat'), format='ascii')
    sobral_sc4k = {'alpha': ufloat(-1.8, 0.2), 'logLstar': ufloat(42.69, 0.045), 'logphistar': ufloat(-2.73, 0.115), 'label': 'Sobral+18 SC4K only'}
    sobral_ssc4k = {'alpha': ufloat(-1.63, 0.165), 'logLstar': ufloat(42.77, 0.105), 'logphistar': ufloat(-3.06, 0.235), 'label': 'Sobral+18 S-SC4K'}
    # if alpha_fixed==-1.6: this_work = {'alpha': alpha_fixed, 'logLstar': 42.435, 'logphistar': -2.701, 'label': 'Median MCMC (no trans)'}
    # if alpha_fixed==-1.6: this_work = {'alpha': alpha_fixed, 'logLstar': 42.392, 'logphistar': -2.393, 'label': 'Median MCMC (with trans)'}
    # elif alpha_fixed==-1.8: this_work = {'alpha': alpha_fixed, 'logLstar': 42.513, 'logphistar': -2.856, 'label': 'Median MCMC (no trans)'}
    # else: 
        # print("Not one of the sanctioned alpha fixed values")
        # this_work = None
    logL = dat_bf['Luminosity']
    bf = dat_bf['MedianLF']
    inds = np.argsort(logL)
    logLVs = dat_veff['Luminosity']
    # logL = np.linspace(logLVs.min(), logLVs.max(), 1001)
    plotStuff(logLVs, dat_veff['BinLF'], dat_veff['BinLFErr'], logL[inds], bf[inds], sobral1=sobral_sc4k, sobral2=sobral_ssc4k, this_work=None)

def NewProc():
    fits_z24 = op.join('LFMCMCOdin', 'ODIN_fsa0_sa-1.49_mcf50_ll45.0_ec2_contam_0.01_cb10newcontam', 'N419_ll_431_all_fitposterior_ODIN_fsa0_sa-1.49_mcf50_ll45.0_ec2_contam_0.01_cb10newcontam_nb50_nw200_ns5000_mcf50_ec_2_env0_bin1.dat')
    fits_z31 = op.join('LFMCMCOdin', 'ODIN_fsa0_sa-1.49_mcf50_ll45.0_ec2_contam_0.01_cb10newcontam', 'N501_ll_431_all_fitposterior_ODIN_fsa0_sa-1.49_mcf50_ll45.0_ec2_contam_0.01_cb10newcontam_nb50_nw200_ns5000_mcf50_ec_2_env0_bin1.dat')
    fits_z45 = op.join('LFMCMCOdin', 'ODIN_fsa0_sa-1.49_mcf50_ll43.2_ec2', 'N673_ll_431_all_fitposterior_ODIN_fsa0_sa-1.49_mcf50_ll43.2_ec2_nb50_nw200_ns5000_mcf50_ec_2_env0_bin1.dat')
    # dat_z45 = op.join('LFMCMCOdin', 'ODIN_fsa0_sa-1.49_mcf50_ll43.2_ec2', 'N673_ll_431_all_ODIN_fsa0_sa-1.49_mcf50_ll43.2_ec2_env0_bin1.dat')
    reds = [2.4, 3.1, 4.5]
    plotEvolution([fits_z24, fits_z31, fits_z45], reds)
    # plotProtoEvol([fits_z24, fits_z31, fits_z45], reds)
    # plotStuffNew([fits_z24, fits_z31, fits_z45], reds)
    # plotDensityEvol([fits_z24, fits_z31, fits_z45], reds, [[0, 1.34, 2.16, 3.2, 12.22], [0, 1.49, 2.17, 3.18, 9.53], [0, 1.74, 2.79, 4.17, 15.41]])
    # plotDensityEvol([fits_z24, fits_z31, fits_z45], reds, [[0, 1.59, 2.78, 12.22], [0, 1.70, 2.77, 9.53], [0, 2.07, 3.63, 15.41]])
    # getIntegInfo(fits_z24, llow=42.0)
    # getIntegInfo(fits_z31, llow=42.0)
    # getIntegInfo(fits_z45, llow=42.0)
    # plotLLComp(dat_z45)

if __name__ == '__main__':
    # main(alpha_fixed=-1.49)
    # main(alpha_fixed=-1.8)
    NewProc()
''' Various plots for luminosity function analysis, including comparison with literature and analysis of cosmic evolution and evolution with environment '''

import numpy as np 
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter1d
from uncertainties import unumpy, ufloat
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from astropy.table import Table 
from astropy.io import fits
import os.path as op
import re
from itertools import cycle
from glob import glob
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
markers_overall = ['o','*','s','+','v','<','>', '1', '8', 'P']
markers = cycle(tuple(markers_overall))

Lsun = 3.8e33

def plotLLComp(orig_file, ll_min=42.8, ll_max=44.0, num_files=13, sm=3, sM=30):
    ''' Issues with N673 luminosity function depending on maximum considered luminosity; an old problem no longer an issue given new contamination treatment '''
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
    ''' Integrals of number of galaxies and total luminosity in galaxies given Schechter parameters; not used '''
    ans_num, _ = quad(lambda x: TrueLumFunc(x, al, ls, phis), llow, lhigh)
    ans_lum, _ = quad(lambda x: 10**x * TrueLumFunc(x, al, ls, phis), llow, lhigh)
    return ans_num, ans_lum

def getIntegsv2(al, ls, phis, llow=42.0, lhigh=46.0):
    ''' Using linear Schechter version to be more straightforward in integral computation; used instead of log Schechter version '''
    ans_num, _ = quad(lambda x: schechter(x, al, 10**phis, 10**ls), 10**llow, 10**lhigh)
    ans_lum, _ = quad(lambda x: x * schechter(x, al, 10**phis, 10**ls), 10**llow, 10**lhigh)
    return ans_num, ans_lum

def getIntegInfo(fitpost, rndsamples=100, llow=42.0, lhigh=46.0, sa=-1.6):
    ''' Print integrals of luminosity functions given posterior samples'''
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
    return numvals, lumvals

def getIntegInfoProto(fitpostprotorig, zs=[2.4, 3.1, 4.5], rndsamples=100, llow=42.0, lhigh=46.0, sa=-1.6):
    ''' Same as getIntegInfo but considering protoclusters vs non-protoclusters '''
    protint, proteu, protel, npint, npeu, npel = np.zeros(len(zs)), np.zeros(len(zs)), np.zeros(len(zs)), np.zeros(len(zs)), np.zeros(len(zs)), np.zeros(len(zs))
    protlint, protleu, protlel, nplint, npleu, nplel = np.zeros(len(zs)), np.zeros(len(zs)), np.zeros(len(zs)), np.zeros(len(zs)), np.zeros(len(zs)), np.zeros(len(zs))
    for i, env0_path, z in zip(np.arange(len(zs)), fitpostprotorig, zs):
        fpnp, fpp = resolve_proto_fit_pair_from_env0(env0_path)
        print(f"Protoclusters at z={z}: ")
        nvi, lvi = getIntegInfo(fpp, rndsamples=rndsamples, llow=llow, lhigh=lhigh, sa=sa)
        protint[i], proteu[i], protel[i] = nvi[1], nvi[2]-nvi[1], nvi[1]-nvi[0]
        protlint[i], protleu[i], protlel[i] = lvi[1], lvi[2]-lvi[1], lvi[1]-lvi[0]
        print(f"Not protoclusters at z={z}: ")
        nvi, lvi = getIntegInfo(fpnp, rndsamples=rndsamples, llow=llow, lhigh=lhigh, sa=sa)
        npint[i], npeu[i], npel[i] = nvi[1], nvi[2]-nvi[1], nvi[1]-nvi[0]
        nplint[i], npleu[i], nplel[i] = lvi[1], lvi[2]-lvi[1], lvi[1]-lvi[0]
    return protint, proteu, protel, npint, npeu, npel, protlint, protleu, protlel, nplint, npleu, nplel

def add_LumFunc_plot(ax1, no_ylabel=False, ax2=None):
    """ Set up the plot for the luminosity function """
    ax1.set_yscale('log')
    if ax2 is None: axlab = ax1
    else: axlab = ax2
    axlab.set_xlabel(r"$\log$ L (erg s$^{-1}$)")
    if not no_ylabel: ax1.set_ylabel(r"$\phi_{\rm{true}}$ (Mpc$^{-3}$ dex$^{-1}$)")
    ax1.minorticks_on()
    if ax2 is not None: ax2.minorticks_on()

def getSamples(logL, nsamples, rndsamples=200, sa=-1.6, ret_prop=0, pers_use=[16, 50, 84], return_params=False):
    ''' Get samples of the luminosity function based on posterior samples of the MCMC Schechter fits '''
    lf = []
    if nsamples.shape[1]==4: alpha = nsamples[:,2]
    else: alpha = np.repeat(sa, nsamples.shape[0])
    if ret_prop: return np.percentile(alpha, pers_use), np.percentile(nsamples[:,0], pers_use), np.percentile(nsamples[:,1], pers_use)
    Lstars, alphas, phistars = np.zeros(rndsamples), np.zeros(rndsamples), np.zeros(rndsamples)
    for i in np.arange(rndsamples):
        ind = np.random.randint(0, nsamples.shape[0])
        Lstars[i], alphas[i], phistars[i] = nsamples[ind, 0], alpha[ind], nsamples[ind, 1]
        modlum = TrueLumFunc(logL, alphas[i], Lstars[i], phistars[i])
        lf.append(modlum)
        
    lf = np.array(lf)
    medianLF = np.median(lf, axis=0)
    if return_params: return lf, medianLF, Lstars, alphas, phistars
    return lf, medianLF

def getnsamples(samples, lnprobcut=7.5):
    ''' Remove some of the outlier starting values in walkers of MCMC chains '''
    nsamples = []
    while len(nsamples)<len(samples)//4: 
        chi2sel = (samples[:, -1] >
                (np.max(samples[:, -1], axis=0) - lnprobcut))
        nsamples = samples[chi2sel, :]
        lnprobcut *= 2.0
    return nsamples

def find_fitposterior_env0(run_dir, filter_prefix, env_mark='env0', bin_mark='bin1'):
    '''Locate fitposterior for env0 / bin1 in a run directory (same idea as getDiffFields glob).'''
    patterns = [
        op.join(run_dir, f'{filter_prefix}*_fitposterior*{env_mark}*{bin_mark}*.dat'),
        op.join(run_dir, f'{filter_prefix}*fitposterior*{env_mark}*{bin_mark}*.dat'),
        op.join(run_dir, f'{filter_prefix}*fitp*{env_mark}*{bin_mark}*.dat'),
    ]
    matches = []
    for p in patterns:
        matches.extend(glob(p))
    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError(
            f'No fitposterior in {run_dir} matching {filter_prefix} {env_mark} {bin_mark}'
        )
    return matches[-1]

def _nw_ns_token(filename):
    m = re.search(r'nw\d+_ns\d+', filename)
    return m.group(0) if m else None

def resolve_proto_fit_pair_from_env0(env0_fit_path):
    '''From env0 bin1 "all" fitposterior, find matching env2 bin1 (not PC) and bin2 (PC) chains.

    Handles varying MCMC lengths (nw200_ns5000 vs nw250_ns7000, etc.) by pairing bin1/bin2
    files that share the same nw/ns token when possible.
    Returns
    -------
    path_not_pc, path_pc : str
        env2 bin1 (field / not in protocluster), env2 bin2 (protocluster)
    '''
    run_dir = op.dirname(env0_fit_path)
    sub2 = op.join(run_dir, '2')
    base = op.basename(env0_fit_path)
    filt = base.split('_')[0]

    fn_np = base.replace('env0', 'env2').replace('_all_', '_pc_')
    path_np = op.join(sub2, fn_np)
    path_pc = path_np.replace('bin1', 'bin2')
    if op.isfile(path_np) and op.isfile(path_pc):
        return path_np, path_pc

    chain_key = _nw_ns_token(base)
    cand_bin1 = sorted(glob(op.join(sub2, f'{filt}*_fitposterior*env2*bin1*.dat')))
    pairs = []
    for p1 in cand_bin1:
        p2 = p1.replace('bin1', 'bin2')
        if op.isfile(p2):
            pairs.append((p1, p2))
    if not pairs:
        raise FileNotFoundError(
            f'No env2 PC/non-PC fitposterior pair in {sub2} for filter {filt}'
        )
    if chain_key:
        for p1, p2 in pairs:
            if chain_key in p1:
                return p1, p2
    return pairs[-1]

def getProtoFiles(fitpostprotorig):
    ''' Read protocluster posterior sample files to get samples.

    fitpostprotorig : list of str
        Paths to env0 bin1 *all* fitposterior files (one per field/redshift run).
    '''
    samples_prot, samples_notprot = [], []
    for env0_path in fitpostprotorig:
        fpnp, fp_p = resolve_proto_fit_pair_from_env0(env0_path)
        dat = Table.read(fp_p, format='ascii')
        dat2 = Table.read(fpnp, format='ascii')
        samples_prot.append(np.lib.recfunctions.structured_to_unstructured(dat.as_array()))
        samples_notprot.append(np.lib.recfunctions.structured_to_unstructured(dat2.as_array()))
        del dat, dat2
    return samples_prot, samples_notprot

def plotLsalProt(fitpostprotorig, reds, cmap_len=256, sigma=1.0):
    ''' Show full posterior distributions of L* and alpha for protoclusters vs not '''
    probs = np.array([0.997, 0.954, 0.683])
    samples_prot, samples_notprot = getProtoFiles(fitpostprotorig)
    fig, ax = plt.subplots(1, len(reds), sharex=True, sharey=True, figsize=(4*len(reds), 4))
    cmps, pclab = [], []
    for j in range(2):
        col = orig_palette_arr[j]
        alpha = np.linspace(0.2, 1, cmap_len)
        cols = np.repeat(np.asarray(col)[None], cmap_len, axis=0)
        cmps.append(ListedColormap(np.column_stack((cols, alpha))))
        if j==0: pcl = 'PC'
        else: pcl = 'Not PC'
        pclab.append(pcl)
    for i, z in enumerate(reds):
        nsamples_prot = getnsamples(samples_prot[i])
        nsamples_notprot = getnsamples(samples_notprot[i])
        all_samples = [nsamples_prot, nsamples_notprot]
        for j in range(2):
            ls, al, lik = all_samples[j][:,0], all_samples[j][:,2], all_samples[j][:,3]
            # lik = gaussian_filter1d(likorig, sigma=sigma)
            liks = np.sort(lik)[::-1]
            linliks = np.exp(liks/liks.max())
            linliks /= linliks.sum()
            linlikscs = np.cumsum(linliks)
            levels = np.zeros_like(probs)
            for k, pr in enumerate(probs):
                levels[k] = liks[np.argmin(abs(linlikscs-pr))]
            # levels = np.percentile(lik, [0.27, 4.55, 31.73])
            # ax[i].tricontour(ls, al, lik, levels=10, linewidths=0.5, colors='k')
            ax[i].tricontour(ls, al, lik, levels, cmap=cmps[j])
        # ax[i].legend(loc='best') label=rf'$z={z}$ {pclab[j]}'
        ax[i].set_title(rf'$z={z}$')
    ax[0].set_ylabel(r'$\alpha$')
    ax2 = fig.add_subplot(111, frameon=False)
    ax2.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    ax2.set_xlabel(r'log L$_*$')
    plt.tight_layout()
    fig.savefig('XMM_Proto_Lsal_comp_corrn.png', bbox_inches='tight', dpi=300)

def plotProtoEvol(fitpostprotorig, reds, Lmin=42.0, Lmax=43.5, Lnum=1001, sa=-1.6):
    ''' Plot protoclusters vs non-protocluster luminosity functions in multiple redshifts, along with ratios in a bottom panel '''
    samples_prot, samples_notprot = getProtoFiles(fitpostprotorig)
    logL = np.linspace(Lmin, Lmax, Lnum)
    fig, ax = plt.subplots(nrows=2, ncols=len(reds), sharex=True, sharey='row', figsize=(12, 5), height_ratios=[0.7, 0.3])
    col1, col2 = orig_palette_arr[:2]
    for i, z in enumerate(reds):
        if i==0: no_ylabel=False
        else: no_ylabel=True
        add_LumFunc_plot(ax[0,i], no_ylabel=no_ylabel, ax2=ax[1,i])
        nsamples_prot = getnsamples(samples_prot[i])
        nsamples_notprot = getnsamples(samples_notprot[i])
        lfp, lfpbest = getSamples(logL, nsamples_prot, sa=sa)
        lfnp, lfnpbest = getSamples(logL, nsamples_notprot, sa=sa)
        lfratbest = lfpbest / lfnpbest
        ax[0,i].plot(logL, lfpbest, linestyle='-', color=col1, label=rf'$z={z}$ PC')
        ax[0,i].plot(logL, lfnpbest, linestyle=':', color=col2, label=rf'$z={z}$ Not PC')
        ax[1,i].plot(logL, lfratbest, linestyle='-', color='purple')
        # indp, indnp = np.argsort(np.median(lfp, axis=1)), np.argsort(np.median(lfnp, axis=1))
        for jj, lfpi, lfnpi in zip(np.arange(len(lfp)), lfp, lfnp):
            ax[0,i].plot(logL, lfpi, linestyle='-', color=col1, alpha=0.05, label='')
            ax[0,i].plot(logL, lfnpi, linestyle=':', color=col2, alpha=0.05, label='')
            ax[1,i].plot(logL, lfpi/lfnpi, linestyle='-', color='purple', alpha=0.05)
            # ax[1,i].plot(logL, lfp[indp[jj]]/lfnp[indnp[jj]], linestyle='-', color='purple', alpha=0.05)
        ax[0,i].set_xlim(Lmin, Lmax)
        ax[0,i].set_ylim(1.0e-6, 3.0e-2)
        ax[0,i].legend(loc='best', frameon=False)
        ax[1,i].set_ylim(0,5)
    ax[1,0].set_ylabel(r'$\phi_{\rm prot}/\phi_{\rm field}$')
    # ax[1,0].yaxis.get_major_ticks()[-1].label1.set_visible(False)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    for i in range(1,3):
        xticks = ax[1,i].xaxis.get_major_ticks()
        xticks[0].label1.set_visible(False)
    # plt.tight_layout()
    fig.savefig("CosmicEvolXMM_PCcorrsnew_subpanel.png", bbox_inches='tight', dpi=300)

def plotProtoEvolProp(fitpostprotorig, reds, dzs, sa=-1.6, llow=42.5, only_integ=False):
    ''' Plot comparison of Schechter parameters and integral for protoclusters vs not '''
    protint, proteu, protel, npint, npeu, npel, protlint, protleu, protlel, nplint, npleu, nplel = getIntegInfoProto(fitpostprotorig, zs=reds, llow=llow, sa=sa)
    if only_integ: return
    samples_prot, samples_notprot = getProtoFiles(fitpostprotorig)
    fig, ax = plt.subplots(nrows=1, ncols=4, sharex=True, figsize=(13, 4))
    paramspa, paramsnpa = [], []
    for i, z in enumerate(reds):
        nsamples_prot = getnsamples(samples_prot[i])
        nsamples_notprot = getnsamples(samples_notprot[i])
        paramsp = getSamples(None, nsamples_prot, sa=sa, ret_prop=1)
        paramsnp = getSamples(None, nsamples_notprot, sa=sa, ret_prop=1)
        paramspa.append(paramsp); paramsnpa.append(paramsnp)
    col1, col2 = orig_palette_arr[:2]
    plotlab = [r'$\alpha$', r'$\log L_*$', r'$\log \phi_*$']
    lz = len(reds)
    for i in range(3):
        pspi = [paramspa[j][i][1] for j in range(lz)]
        pspiue = [paramspa[j][i][2]-pspi[j] for j in range(lz)]
        pspile = [pspi[j]-paramspa[j][i][0] for j in range(lz)]
        psnpi = [paramsnpa[j][i][1] for j in range(lz)]
        psnpiue = [paramsnpa[j][i][2]-psnpi[j] for j in range(lz)]
        psnpile = [psnpi[j]-paramsnpa[j][i][0] for j in range(lz)]
        if i==0: labelp, labelnp = 'Protocluster', 'Not in protocluster'
        else: labelp, labelnp = '', ''
        ax[i].errorbar(reds, pspi, yerr=np.row_stack((pspile, pspiue)), xerr=dzs, color=col1, linestyle='none', markersize=10, capsize=2, label=labelp, marker='s')
        ax[i].errorbar(reds, psnpi, yerr=np.row_stack((psnpile, psnpiue)), xerr=dzs, color=col2, linestyle='none', markersize=10, capsize=2, label=labelnp, marker='^')
        ax[i].set_ylabel(plotlab[i])
    ax[3].errorbar(reds, protint, yerr=np.row_stack((protel, proteu)), xerr=dzs, color=col1, linestyle='none', markersize=10, capsize=2, label='', marker='s')
    # ax[4].errorbar(reds, protlint, yerr=np.row_stack((protlel, protleu)), xerr=dzs, color=col1, linestyle='none', markersize=10, capsize=2, label='', marker='s')
    ax[3].errorbar(reds, npint, yerr=np.row_stack((npel, npeu)), xerr=dzs, color=col2, linestyle='none', markersize=10, capsize=2, label='', marker='^')
    # ax[4].errorbar(reds, nplint, yerr=np.row_stack((nplel, npleu)), xerr=dzs, color=col2, linestyle='none', markersize=10, capsize=2, label='', marker='^')
    ax[3].set_ylabel(rf'$\int_{{{llow:0.1f}}}^{{\infty}}\phi(\mathcal{{L}})d\mathcal{{L}}$ (Mpc$^{{-3}}$)')
    # ax[4].set_ylabel(rf'$\int_{{{llow:0.1f}}}^{{\infty}}10^{{\mathcal{{L}}}}\phi(\mathcal{{L}})d\mathcal{{L}}$ ($L_{{\odot}}$ Mpc$^{{-3}}$)')
    ax[0].legend(loc='best', frameon=True, fontsize='small')
    ax2 = fig.add_subplot(111, frameon=False)
    ax2.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    ax2.set_xlabel('Redshift')
    plt.tight_layout()
    fig.savefig("CosmicEvolXMMPropcorrsnewn4col.png", bbox_inches='tight', dpi=300)

def plotEvolution(fitpostfs, reds, Lmin=42.0, Lmax=43.5, Lnum=1001, sa=-1.6):
    ''' Plot cosmic evolution of luminosity function '''
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
    fig.savefig("CosmicEvolXMM_corrsnew.png", bbox_inches='tight', dpi=300)

def plotDensityEvol(fit_names_orig, reds, dens_vals, Lmin=42.0, Lmax=43.5, Lnum=1001, ymin=1.0e-6, ymax=3.0e-2, sa=-1.6):
    ''' Plot density evolution of luminosity function in different redshifts '''
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
    
    fig.savefig("CosmicDensEvolCOSMOS_corrsnew_3bins.png", bbox_inches='tight', dpi=300)

def calc_phi_err(phi, logphierr):
    ''' Linear error in luminosity function (given log value) '''
    return np.log(10) * phi * logphierr

def OtherLFData(sobfile, sobothers):
    ''' Literature luminosity functions '''
    herenz = {'lum':np.array([42.3, 42.5, 42.7, 42.9, 43.1, 43.3]), 'phi':np.array([5.9e-3, 3.1e-3, 1.4e-3, 4.8e-4, 1.5e-4, 2.3e-5]), 'phierr':np.array([8.6e-4, 4.1e-4, 2.3e-4, 1.2e-4, 5.9e-5, 2.3e-5])}
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
    return herenz, sobs, logLsob, logLsobe, phisob, phiseu, phisel, zavg, ref, logLso, logLsoe, phiso, phisoeu, phisoel, refuniq, colref, markref

def plotStuffNew(fitpostfs, reds, sobfile='sty378_supp/SC4K_full_LFs_Table_C1.fits', sobothers='sty378_supp/SSC4K_compilation_Table_C2.fits', Lmin=42.0, Lmax=43.8, Lnum=1001, sobkeys=['IA427 ($z=2.5$)', 'IA505 ($z=3.2$)', 'IA679 ($z=4.6$)'], sobzs=[2.5, 3.2, 4.6], maxdiff=0.21, llims=[43.1, 43.1, 43.2], ymin=5.0e-7, ymax=3.0e-2, sa=-1.6, llims_low=[42.0, 42.1, 42.2], veffdats=None):
    ''' Comparison with literature '''
    logL = np.linspace(Lmin, Lmax, Lnum)
    herenz, sobs, logLsob, logLsobe, phisob, phiseu, phisel, zavg, ref, logLso, logLsoe, phiso, phisoeu, phisoel, refuniq, colref, markref = OtherLFData(sobfile, sobothers)
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
        ax[i].plot(logL, lfbest, linestyle='-', color=coli, label=rf'Nagaraj+25 $z={z}$')
        if veffdats is not None and (i==0 or i==1):
            # if i==2: lvd, alvd = r'"" $\bf{\rm{No}}$ Contam Removal', 0.1
            lvd, alvd = '"" Normal Contam Removal', 0.25
            vlum, vlf, vlfe = veffdats[i]['Luminosity'], veffdats[i]['BinLF'], veffdats[i]['BinLFErr']
            condv = vlf>1.0e-12
            ax[i].errorbar(vlum[condv], vlf[condv], yerr=vlfe[condv], fmt='b^', label=lvd, alpha=alvd)
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
        if i==2: ax[i].errorbar(herenz['lum'], herenz['phi'], yerr=herenz['phierr'], color=next(orig_palette), marker=next(markers), label=r'Herenz+19 $3<z<6.7$', capsize=2, linestyle='none')
        # ax[i].vlines(llims[i], ymin, ymax, colors='k', label='')
        # Now using contamination algorithm so don't need a 
        condvi = logL>=llims[i]
        ax[i].fill_between(logL[condvi], ymin*np.ones_like(logL[condvi]), ymax*np.ones_like(logL[condvi]), color='k', alpha=0.1, label='')
        condvi = logL<=llims_low[i]
        ax[i].fill_between(logL[condvi], ymin*np.ones_like(logL[condvi]), ymax*np.ones_like(logL[condvi]), color='k', alpha=0.1, label='')
        ax[i].legend(loc='best', frameon=False, fontsize=8)
    ax[0].set_xlim(Lmin, Lmax)
    ax[0].set_ylim(ymin, ymax)
    plt.tight_layout()
    
    fig.savefig("FullLitCompcorrsnew_vf_xmm.png", bbox_inches='tight', dpi=300)

def plotLumFuncStd(logL, lfs_new, lfs_old, filter, numtot=25, Lmin=42.0, Lmax=43.8, rndsamples=50, ymin=5.0e-7, ymax=3.0e-2, rndfac=5, sobfile='sty378_supp/SC4K_full_LFs_Table_C1.fits', sobothers='sty378_supp/SSC4K_compilation_Table_C2.fits', sobkeys=['IA427 ($z=2.5$)', 'IA505 ($z=3.2$)', 'IA679 ($z=4.6$)'], maxdiff=0.21, stdver=0):
    '''Experiment with effects of uncertainties of completeness and contamination on luminosity function and its uncertainty '''
    assert filter.lower()=='n501'
    z = 3.1
    lfs_new_med, lfs_new_std = np.median(lfs_new, axis=(0,1)), np.std(lfs_new, axis=(0,1))
    lfs_old_med, lfs_old_std = np.median(lfs_old, axis=0), np.std(lfs_old, axis=0)
    lfs_rat = lfs_new_std / lfs_old_std
    print("Median std ratio: ", np.median(lfs_rat))

    ######### Literature area #########
    _, sobs, logLsob, logLsobe, phisob, phiseu, phisel, zavg, ref, logLso, logLsoe, phiso, phisoeu, phisoel, refuniq, colref, markref = OtherLFData(sobfile, sobothers)

    ##### Plot area #####
    fig, ax = plt.subplots()
    add_LumFunc_plot(ax)
    if not stdver:
        for i in range(numtot):
            for j in range(rndsamples):
                if i==0 and j==0: label='Varied completeness'
                else: label=''
                ax.plot(logL, lfs_new[i][j], linestyle='-', color='r', alpha=0.02, label=label)
        for j in range(rndsamples*rndfac):
            if j==0: label='Fixed completeness'
            else: label=''
            ax.plot(logL, lfs_old[j], linestyle='-', color='b', alpha=0.02, label=label)
    else:
        ax.plot(logL, lfs_new_med, linestyle='-', color='r', label='Varied completeness')
        ax.plot(logL, lfs_old_med, linestyle='-', color='b', label='Fixed completeness')
        ax.fill_between(logL, lfs_new_med - lfs_new_std, lfs_new_med + lfs_new_std, color='r', alpha=0.1, label='')
        ax.fill_between(logL, lfs_old_med - lfs_old_std, lfs_old_med + lfs_old_std, color='b', alpha=0.1, label='')

    condsob = sobs == sobkeys[1]
    ax.errorbar(logLsob[condsob], phisob[condsob], yerr=np.row_stack((phisel[condsob], phiseu[condsob])), xerr=logLsobe[condsob]/2, linestyle='none', marker=markers_overall[0], color='k', label='', capsize=2)
    condsobo = abs(zavg-z)<maxdiff
    refsj = np.unique(ref[condsobo])
    for k, rj in enumerate(refsj):
        if rj=='Konno+2016': continue
        ind = np.where(refuniq==rj)[0][0]
        condsofull = np.logical_and(condsobo, ref==rj)
        ax.errorbar(logLso[condsofull], phiso[condsofull], yerr=np.row_stack((phisoel[condsofull], phisoeu[condsofull])), xerr=logLsoe[condsofull]/2, linestyle='none', marker=markref[ind], color=colref[ind], label='', capsize=2)
    leg = ax.legend(loc='best', frameon=False)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    ax.set_xlim(Lmin, Lmax)
    ax.set_ylim(ymin, ymax)
    figname = f'LFCompCombo{filter}.png'
    if stdver: figname = figname.replace('Combo', 'ComboStd')
    fig.savefig(figname, bbox_inches='tight', dpi=300)
    plt.close('all')
    
    fig, ax = plt.subplots()
    ax.plot(logL, lfs_rat, 'b-')
    ax.set_xlim(Lmin, Lmax)
    ax.set_ylim(lfs_rat.min(), lfs_rat.max())
    ax.set_xlabel(r"$\log$ L (erg s$^{-1}$)")
    ax.set_ylabel('Ratio of standard deviation (varied/fixed)')
    fig.savefig(f'LFCompStdRat{filter}.png', bbox_inches='tight', dpi=300)
    plt.close('all')

def plotLumFuncCombo(base_dir, numtot=25, filter='N501', Lmin=42.0, Lmax=43.8, Lnum=401, rndsamples=50, ymin=5.0e-7, ymax=3.0e-2, rndfac=5):
    ''' Paired with plotlumfuncstd to show results of experiment with completeness uncertainties '''
    # fn = glob(op.join(base_dir+'_combo', "*VeffLF*.dat"))[0]
    # veff = Table.read(fn, format='ascii')
    # vlum, vlf, vlfe, vlfo, vlfeo = veff['Luminosity'], veff['BinLF'], veff['BinLFErr'], veff['BinLFOrig'], veff['BinLFErrOrig']
    # vlf += 0.04; vlfo += 0.04
    logL = np.linspace(Lmin, Lmax, Lnum)
    # fig, ax = plt.subplots()
    lfs = []
    # add_LumFunc_plot(ax)
    # ax.errorbar(vlum, vlf, yerr=vlfe, fmt='b^', linestyle='none', capsize=2, label=r'V$_{\rm eff}$ + Filter')
    # ax.errorbar(vlum, vlfo, yerr=vlfeo, fmt='cs', linestyle='none', capsize=2, label=r'V$_{\rm eff}$')
    # aln, lsn, psn = np.zeros((numtot, rndsamples)), np.zeros((numtot, rndsamples)), np.zeros((numtot, rndsamples))
    for i in range(numtot):
        fpf = glob(op.join(base_dir+f'_{i}', '*fitposterior*.dat'))[0]
        dat = Table.read(fpf,format='ascii')
        samplei = np.lib.recfunctions.structured_to_unstructured(dat.as_array())
        del dat
        nsamples = getnsamples(samplei)
        lf, _, _, _, _ = getSamples(logL, nsamples, rndsamples=rndsamples, return_params=True)
        lfs.append(lf) #; lfbests.append(lfbest)
    # lfrealbest = np.median(lfbests, axis=0)
    # for i in range(numtot):
    #     for j in range(rndsamples):
    #         if i==0 and j==0: label='MCMC solutions'
    #         else: label=''
    #         ax.plot(logL, lfs[i][j], linestyle='-', color='r', alpha=0.02, label=label)
    # ax.plot(logL, lfrealbest, 'k-')
    # leg = ax.legend(loc='best', frameon=False)
    # for lh in leg.legend_handles:
    #     lh.set_alpha(1)
    # ax.set_xlim(Lmin, Lmax)
    # ax.set_ylim(ymin, ymax)
    # fig.savefig(f'ComboLF{filter}.png', bbox_inches='tight', dpi=300)
    # plt.close('all')

    fpf = glob(op.join(base_dir, f'{filter}*fitposterior*.dat'))[0]
    dat = Table.read(fpf,format='ascii')
    samples = np.lib.recfunctions.structured_to_unstructured(dat.as_array())
    del dat
    nsamples = getnsamples(samples)
    lfs_old, _, _, _, _ = getSamples(logL, nsamples, rndsamples=rndsamples*rndfac, return_params=True)
    plotLumFuncStd(logL, np.array(lfs).astype(float)*1.1, lfs_old.astype(float)*1.1, filter, numtot=numtot, Lmin=Lmin, Lmax=Lmax, rndsamples=rndsamples, ymin=ymin, ymax=ymax, rndfac=rndfac, stdver=0)

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

def NewProc():
    ''' Primary code to run all of the pertinent code above to create plots '''
    fits_z24 = op.join('LFMCMCOdin', 'ODIN_fsa0_sa-1.49_mcf50_ll45.0_ec2_contam_0.5_cb10xmmrun', 'N419_xmm_all_fitposterior_ODIN_fsa0_sa-1.49_mcf50_ll45.0_ec2_contam_0.5_cb10xmmrun_nb50_nw200_ns5000_mcf50_ec_2_env0_bin1.dat')
    fits_z31 = op.join('LFMCMCOdin', 'ODIN_fsa0_sa-1.49_mcf50_ll45.0_ec2_contam_0.5_cb10xmmrun', 'N501_xmm_all_fitposterior_ODIN_fsa0_sa-1.49_mcf50_ll45.0_ec2_contam_0.5_cb10xmmrun_nb50_nw200_ns5000_mcf50_ec_2_env0_bin1.dat')
    fits_z45 = op.join('LFMCMCOdin', 'ODIN_fsa0_sa-1.49_mcf50_ll45.0_ec2_contam_0.68_cb4xmmrun', 'N673_xmm_all_fitposterior_ODIN_fsa0_sa-1.49_mcf50_ll45.0_ec2_contam_0.68_cb4xmmrun_nb50_nw200_ns5000_mcf50_ec_2_env0_bin1.dat')

    # veffnc_z24 = op.join('LFMCMCOdin', 'ODIN_fsa0_sa-1.49_mcf50_ll45.0_ec2_contam_0.5_cb10nocontam', 'N419_new_all_VeffLF_ODIN_fsa0_sa-1.49_mcf50_ll45.0_ec2_contam_0.5_cb10nocontam_nb50_nw200_ns5000_mcf50_ec_2_env0_bin1_c1.dat')
    # veffnc_z31 = op.join('LFMCMCOdin', 'ODIN_fsa0_sa-1.49_mcf50_ll45.0_ec2_contam_0.5_cb10nocontam', 'N501_new_all_VeffLF_ODIN_fsa0_sa-1.49_mcf50_ll45.0_ec2_contam_0.5_cb10nocontam_nb50_nw200_ns5000_mcf50_ec_2_env0_bin1_c1.dat')
    # veffnc_z45 = op.join('LFMCMCOdin', 'ODIN_fsa0_sa-1.49_mcf50_ll45.0_ec2_contam_0.68_cb4nocontam', 'N673_new_all_VeffLF_ODIN_fsa0_sa-1.49_mcf50_ll45.0_ec2_contam_0.68_cb4nocontam_nb50_nw200_ns5000_mcf50_ec_2_env0_bin1_c1.dat')
    # veffdats = [Table.read(veffnc_z24, format='ascii'), Table.read(veffnc_z31, format='ascii'), Table.read(veffnc_z45, format='ascii')]

    # fits_z24 = op.join('LFMCMCOdin', 'ODIN_fsa1_sa-1.60_mcf50_ll45.0_ec2_contam_0.5_cb10corrsn', 'N419_new_all_fitposterior_ODIN_fsa1_sa-1.60_mcf50_ll45.0_ec2_contam_0.5_cb10corrsn_nb50_nw120_ns2000_mcf50_ec_2_env0_bin1.dat')
    # fits_z31 = op.join('LFMCMCOdin', 'ODIN_fsa1_sa-1.60_mcf50_ll45.0_ec2_contam_0.5_cb10corrsn', 'N501_new_all_fitposterior_ODIN_fsa1_sa-1.60_mcf50_ll45.0_ec2_contam_0.5_cb10corrsn_nb50_nw150_ns3000_mcf50_ec_2_env0_bin1.dat')
    # fits_z45 = op.join('LFMCMCOdin', 'ODIN_fsa1_sa-1.60_mcf50_ll45.0_ec2_contam_0.5_cb10corrsn', 'N673_new_all_fitposterior_ODIN_fsa1_sa-1.60_mcf50_ll45.0_ec2_contam_0.5_cb10corrsn_nb50_nw120_ns2000_mcf50_ec_2_env0_bin1.dat')
    # dat_z45 = op.join('LFMCMCOdin', 'ODIN_fsa0_sa-1.49_mcf50_ll43.2_ec2', 'N673_ll_431_all_ODIN_fsa0_sa-1.49_mcf50_ll43.2_ec2_env0_bin1.dat')
    reds = [2.4, 3.1, 4.5]
    # plotEvolution([fits_z24, fits_z31, fits_z45], reds)
    plotProtoEvol([fits_z24, fits_z31, fits_z45], reds)
    plotProtoEvolProp([fits_z24, fits_z31, fits_z45], reds, dzs=[0.062, 0.063, 0.083], llow=42.5)
    # plotStuffNew([fits_z24, fits_z31, fits_z45], reds, llims=[43.32, 43.45, 43.67], llims_low=[42.20, 42.36, 42.50])
    # plotDensityEvol([fits_z24, fits_z31, fits_z45], reds, [[0, 1.34, 2.16, 3.2, 12.22], [0, 1.49, 2.17, 3.18, 9.53], [0, 1.74, 2.79, 4.17, 15.41]])
    # plotDensityEvol([fits_z24, fits_z31, fits_z45], reds, [[0, 1.59, 2.78, 12.22], [0, 1.70, 2.77, 9.53], [0, 2.07, 3.63, 15.41]])
    # getIntegInfo(fits_z24, llow=42.5)
    # getIntegInfo(fits_z31, llow=42.5)
    # getIntegInfo(fits_z45, llow=42.5)
    # plotLLComp(dat_z45)

    plotLsalProt([fits_z24, fits_z31, fits_z45], reds)

def plotMultVeff(*filenames):
    ''' Plot multiple V/V_max results '''
    fig, ax = plt.subplots()
    add_LumFunc_plot(ax)
    namefull = ''
    for i, fn in enumerate(filenames):
        veffdi = Table.read(fn, format='ascii')
        lumi, lfi, lfei = veffdi['Luminosity'], veffdi['BinLF'], veffdi['BinLFErr']
        fne = fn.split('/')[-1]
        basic = fne.split('_VeffLF')[0]
        filter_name = fne.split('_')[0]
        if filter_name=='N673': cb = 3
        else: cb = 10
        extra = fne.split('_nb50')[0].split('0.5')[1].split(f'cb{cb}')[1]
        namei = f'{basic}_{extra}'
        ax.errorbar(lumi, lfi, yerr=lfei, color=orig_palette_arr[i], linestyle='none', marker=markers_overall[i], label=namei)
        namefull += namei
        if i<len(filenames)-1: namefull+='_'
    ax.legend(loc='best', frameon=False)
    fig.savefig(f'VeffComp_{namefull}.png', bbox_inches='tight', dpi=300)
    plt.close('all')

def plotDiffFields(fit1=None, fit2=None, filter='N501', f1='COSMOS', f2='XMM-LSS', Lmin=42.0, Lmax=43.5, Lnum=1001, fits=None, field_names=None, out_name=None):
    ''' Compare LF posteriors for an arbitrary number of fields.

    Backward-compatible usage:
        plotDiffFields(fit1, fit2, filter='N501', f1='COSMOS', f2='XMM-LSS')

    New usage:
        plotDiffFields(fits=[fit1, fit2, fit3], field_names=['COSMOS', 'XMM-LSS', 'SHELA P12'], filter='N501')
    '''
    logL = np.linspace(Lmin, Lmax, Lnum)
    if fits is None:
        fits = [fit1]
        if fit2 is not None:
            fits.append(fit2)
    if field_names is None:
        if len(fits) == 2:
            field_names = [f1, f2]
        else:
            field_names = [f'Field{i+1}' for i in range(len(fits))]
    if len(field_names) != len(fits):
        raise ValueError("field_names length must match number of fit files")

    samples = []
    for fpf in fits:
        dat = Table.read(fpf,format='ascii')
        samples.append(np.lib.recfunctions.structured_to_unstructured(dat.as_array()))
        del dat
    fig, ax = plt.subplots()
    add_LumFunc_plot(ax)
    for i, (sampi, fi) in enumerate(zip(samples, field_names)):
        coli = orig_palette_arr[i % len(orig_palette_arr)]
        nsamples = getnsamples(sampi)
        lf, lfbest = getSamples(logL, nsamples)
        ax.plot(logL, lfbest, linestyle='-', color=coli, label=fi)
        for lfi in lf:
            ax.plot(logL, lfi, linestyle='-', color=coli, alpha=0.05, label='')
    ax.set_xlim(Lmin, Lmax)
    ax.set_ylim(1.0e-6, 3.0e-2)
    ax.legend(loc='best', frameon=False)
    if out_name is None:
        name_stub = '_'.join([fi.replace(' ', '-') for fi in field_names])
        out_name = f"LFComp_{name_stub}_{filter}.png"
    fig.savefig(out_name, bbox_inches='tight', dpi=300)
    plt.close('all')

def plotDiffFieldsProto(filter='N501', Lmin=42.0, Lmax=43.5, Lnum=1001, fits_env0=None, field_names=None, out_name=None, sa=-1.6):
    ''' Compare LF posteriors across fields, with protocluster (env2 bin2) vs non-protocluster (env2 bin1).

    fits_env0 : list of str
        One env0 bin1 *all* fitposterior path per field; proto/field chains are resolved via
        resolve_proto_fit_pair_from_env0 (handles different nw/ns between env0 and env2).
    field_names : list of str
        Legend labels per field.
    '''
    logL = np.linspace(Lmin, Lmax, Lnum)
    if fits_env0 is None or field_names is None:
        raise ValueError('fits_env0 and field_names are required')
    if len(fits_env0) != len(field_names):
        raise ValueError('fits_env0 and field_names must have the same length')
    fig, ax = plt.subplots()
    add_LumFunc_plot(ax)
    for i, (env0_path, fnlabel) in enumerate(zip(fits_env0, field_names)):
        path_field, path_pc = resolve_proto_fit_pair_from_env0(env0_path)
        col = orig_palette_arr[i % len(orig_palette_arr)]
        for lfpath, linestyle, role in (
            (path_field, ':', 'field'),
            (path_pc, '-', 'PC'),
        ):
            dat = Table.read(lfpath, format='ascii')
            samp = np.lib.recfunctions.structured_to_unstructured(dat.as_array())
            del dat
            nsamples = getnsamples(samp)
            lf, lfbest = getSamples(logL, nsamples, sa=sa)
            ax.plot(logL, lfbest, linestyle=linestyle, color=col, label=f'{fnlabel} ({role})')
            for lfi in lf:
                ax.plot(logL, lfi, linestyle=linestyle, color=col, alpha=0.05, label='')
    ax.set_xlim(Lmin, Lmax)
    ax.set_ylim(1.0e-6, 3.0e-2)
    ax.legend(loc='best', frameon=False, fontsize='small')
    if out_name is None:
        name_stub = '_'.join([fi.replace(' ', '-') for fi in field_names])
        out_name = f'LFCompProto_{name_stub}_{filter}.png'
    fig.savefig(out_name, bbox_inches='tight', dpi=300)
    plt.close('all')

def _run_dir_suffixes_for_filter(filter):
    if filter == 'N673':
        return ['om09', 'xmm2'], ['COSMOS', 'XMM-LSS']
    return (
        ['om09', 'xmm2', 'shela_p12', 'shela_p56', 'shela_p78'],
        ['COSMOS', 'XMM-LSS', 'SHELA P12', 'SHELA P56', 'SHELA P78'],
    )

def _ml_contam_cb_for_filter(filter):
    if filter == 'N673':
        return 42.50, 0.68, 4
    if filter == 'N419':
        return 42.20, 0.5, 10
    return 42.36, 0.5, 10

def getDiffFields(filter):
    ml, cl, cb = _ml_contam_cb_for_filter(filter)
    base_dir = 'LFMCMCOdin'
    next_dir_base = f'ODIN_fsa0_sa-1.49_ml{ml}_ll45.0_ec2_contam_{cl}_cb{cb}'
    bases, fields = _run_dir_suffixes_for_filter(filter)
    fits = []
    field_names = []
    for i, base in enumerate(bases):
        run_dir = op.join(base_dir, next_dir_base + base)
        fits.append(find_fitposterior_env0(run_dir, filter))
        field_names.append(fields[i])
    plotDiffFields(fits=fits, field_names=field_names, filter=filter)

def getDiffFieldsProto(filter):
    ml, cl, cb = _ml_contam_cb_for_filter(filter)
    base_dir = 'LFMCMCOdin'
    next_dir_base = f'ODIN_fsa0_sa-1.49_ml{ml}_ll45.0_ec2_contam_{cl}_cb{cb}'
    bases, fields = _run_dir_suffixes_for_filter(filter)
    fits_env0 = []
    for base in bases:
        run_dir = op.join(base_dir, next_dir_base + base)
        fits_env0.append(find_fitposterior_env0(run_dir, filter))
    plotDiffFieldsProto(filter=filter, fits_env0=fits_env0, field_names=fields)

if __name__ == '__main__':
    # NewProc()
    # plotMultVeff('LFMCMCOdin/ODIN_fsa0_sa-1.49_mcf50_ll45.0_ec2_contam_0.5_cb10newdata/N501_new_trial_VeffLF_ODIN_fsa0_sa-1.49_mcf50_ll45.0_ec2_contam_0.5_cb10newdata_nb50_nw200_ns4000_mcf50_ec_2_env0_bin1_c1.dat', 'LFMCMCOdin/ODIN_fsa0_sa-1.49_mcf50_ll45.0_ec2_contam_0.5_cb10corrsnew/N501_new_all_VeffLF_ODIN_fsa0_sa-1.49_mcf50_ll45.0_ec2_contam_0.5_cb10corrsnew_nb50_nw150_ns3000_mcf50_ec_2_env0_bin1_c1.dat')
    # plotLumFuncCombo('LFMCMCOdin/ODIN_fsa0_sa-1.49_mcf50_ll45.0_ec2_contam_0.5_cb10lumminnv')
    # getDiffFields(filter='N419')
    getDiffFieldsProto(filter='N673')
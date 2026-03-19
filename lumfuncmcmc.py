''' Backbone file for calculating luminosity functions '''

import numpy as np 
import logging
import emcee
import pickle
from uncertainties import unumpy, ufloat
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.integrate import trapezoid
from scipy.interpolate import RegularGridInterpolator as RGIScipy
from scipy.stats import binned_statistic, poisson, uniform
from math import lgamma
from astropy.table import Table
from time import time
import matplotlib.pyplot as plt
import corner
import VmaxLumFunc as V
from scipy.optimize import fsolve
from multiprocessing import Pool
import os.path as op
import seaborn as sns
sns.set_context("paper",font_scale=1.3) # options include: talk, poster, paper
sns.set_style("ticks")
sns.set_style({"xtick.direction": "in","ytick.direction": "in",
               "xtick.top":True, "ytick.right":True,
               "xtick.major.size":12, "xtick.minor.size":4,
               "ytick.major.size":12, "ytick.minor.size":4,
               })

c = 3.0e18 # Speed of light in Angstroms/s

def flin(B, x):
    ''' Linear function '''
    return B[0]*x + B[1]

def poisson_lnpmf(k, mu):
    ''' Log poisson probability mass function '''
    return k*np.log(mu) - lgamma(k+1) - mu

def consecutive(data, stepsize=1):
    ''' Find regions of consecutive values in an array (for continuity purposes)--typically used on an array of indices'''
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def makeTransFigs(filter, lam, trans, dlogL, delz, pden, lammin=4900., lammax=5125.):
    ''' To make plots of tranmission curve information (including the effects on measuring luminosities and effective volumes)'''
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].plot(lam, trans, 'b-')
    ax[1].plot(dlogL, delz, 'b-')
    ax[2].semilogy(dlogL, pden, 'b-')
    ax[0].set_xlim(lammin, lammax); ax[0].set_ylim(trans.min(), trans.max())
    ax[1].set_xlim(dlogL.min()-0.01, dlogL.max()); ax[1].set_ylim(delz.min(), delz.max())
    ax[2].set_xlim(dlogL.min()-0.01, dlogL.max()); ax[2].set_ylim(pden.min()-0.01, pden.max())
    ax[0].set_xlabel(r'$\lambda$ ($\AA$)'); ax[0].set_ylabel('Transmission')
    ax[1].set_xlabel(r'$\log L - \log L_{\rm min}$ (erg s$^{-1}$)'); ax[1].set_ylabel(r'$\Delta z$')
    ax[2].set_xlabel(r'$\log L - \log L_{\rm min}$ (erg s$^{-1}$)'); ax[2].set_ylabel(r'Probability Density')
    plt.tight_layout()
    fig.savefig(f'{filter}_TransInfo.png', bbox_inches='tight', dpi=300)

def getContamination(filter='N419', file_name_orig='N419_LAE_Contamination_Analysis_12_26_2024.csv', interp_type='linear', errtab='confidence_interval_1s.txt', binnum=5, contam_lim=0.01, test_contam_num=10001, contam_type='L_LCA', density_frac=1.0, mag_corr=0.0, nsamp=25): #cat_noagn_orig='LyaN419FluxesFinalIntRem.dat':
    ''' Determine contamination fraction as a function of narrow-band magnitude and make a nifty plot showing it'''
    file_name = file_name_orig.replace('N419', filter)
    if not op.exists(file_name):
        x = np.linspace(0, 100, 1001)
        y = np.ones_like(x)
        z = np.zeros_like(x)
        contamf = interp1d(x, y, kind=interp_type, fill_value=1.0, bounds_error=False)
        contamhf = interp1d(x, z, kind=interp_type, fill_value=0.0, bounds_error=False)
        contamlf = interp1d(x, z, kind=interp_type, fill_value=0.0, bounds_error=False)
        return contamf, contamhf, contamlf, -99.0
    dat = Table.read(file_name, format='csv')
    nb_mag, cl = dat['NARROWBAND_MAGNITUDE']+mag_corr, dat['CLASSIFICATION']
    # ids_desi, flux, z, ps, cl, comment = dat['ID'], dat['Lya Flux'], dat['z'], dat['P/S'], dat['Class'], dat['Comment']
    # condlae = np.logical_or(np.logical_and(ps=='s', cl=='LAE'), ps!='s')
    assert contam_type == 'L_LCA' #Can un-comment other contamination options if desired
    speclae = np.where(cl=='LAE')[0]
    specagn = np.where(cl=='AGN')[0]
    specctm = np.where(cl=='CONTAM')[0]
    allspec = np.where(np.logical_or.reduce((cl=='LAE', cl=='CONTAM', cl=='AGN')))[0]
    # if contam_type == 'LU_LCAU':
    #     speclae = np.where(np.logical_or(cl=='LAE', cl=='UNDET'))[0]
    #     allspec = np.where(np.logical_or.reduce((cl=='LAE', cl=='CONTAM', cl=='AGN', cl=='UNDET')))[0]
    # elif contam_type == 'L_LCA':
    #     speclae = np.where(cl=='LAE')[0]
    #     allspec = np.where(np.logical_or.reduce((cl=='LAE', cl=='CONTAM', cl=='AGN')))[0]
    # elif contam_type == 'L_LC':
    #     speclae = np.where(cl=='LAE')[0]
    #     allspec = np.where(np.logical_or.reduce((cl=='LAE', cl=='CONTAM')))[0]
    # elif contam_type == 'LA_LCA':
    #     speclae = np.where(np.logical_or(cl=='LAE', cl=='AGN'))[0]
    #     allspec = np.where(np.logical_or.reduce((cl=='LAE', cl=='CONTAM', cl=='AGN')))[0]
    # elif contam_type == 'LU_LCU':
    #     speclae = np.where(np.logical_or(cl=='LAE', cl=='UNDET'))[0]
    #     allspec = np.where(np.logical_or.reduce((cl=='LAE', cl=='CONTAM', cl=='UNDET')))[0]
    # elif contam_type == 'LAU_LCAU':
    #     speclae = np.where(np.logical_or.reduce((cl=='LAE', cl=='AGN', cl=='UNDET')))[0]
    #     allspec = np.where(np.logical_or.reduce((cl=='LAE', cl=='CONTAM', cl=='AGN', cl=='UNDET')))[0]
    # else:
    #     print("Not one of the possible options")
    #     return None, None, None, None

    pois = Table.read(errtab, format='ascii')
    num, lb, hb = pois['Num'], pois['LowBound'], pois['HighBound']
    nb_lae = nb_mag[speclae]
    nb_all = nb_mag[allspec]
    nb_agn, nb_ctm = nb_mag[specagn], nb_mag[specctm]
    # print(f"nb_lae min {nb_lae.max():0.2f}, max {nb_lae.min():0.2f}")
    # print(f"nb_all min {nb_all.max():0.2f}, max {nb_all.min():0.2f}")
    # print("Total LAE sample size:", flux_lae.size)
    # print("Total sample size:", flux_all.size)
    # pers = np.linspace(0, 100, binnum+1)
    # bin_edges = np.percentile(nb_all, pers)
    bin_edges = np.linspace(nb_all.min(), nb_all.max()+1.0e-6, binnum+1)
    # bin_edges[-1] += 1.0e-6 # Want to make sure the last flux is included
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.0
    contam, contaml, contamh = np.ones(binnum), np.zeros(binnum), np.zeros(binnum)
    flss, fass, fassn = np.ones(binnum, dtype=int), np.ones(binnum, dtype=int), np.ones(binnum)
    for i in range(binnum):
        cond = np.logical_and(nb_all>=bin_edges[i], nb_all<bin_edges[i+1])
        cond_lae = np.logical_and(nb_lae>=bin_edges[i], nb_lae<bin_edges[i+1])
        cond_agn = np.logical_and(nb_agn>=bin_edges[i], nb_agn<bin_edges[i+1])
        cond_ctm = np.logical_and(nb_ctm>=bin_edges[i], nb_ctm<bin_edges[i+1])
        fls, fas = nb_lae[cond_lae].size, nb_all[cond].size
        fagns, fctms = nb_agn[cond_agn].size, nb_ctm[cond_ctm].size
        fas_new = fls + fagns + fctms*density_frac
        contam[i] = fls/fas_new
        flss[i], fass[i], fassn[i] = fls, fas, fas_new
        if fls>num.max(): contaml[i], contamh[i] = np.sqrt(fls)/fas_new, np.sqrt(fls)/fas_new
        else:
            cond_pois = np.where(fls==num)[0][0]
            contaml[i], contamh[i] = (fls-lb[cond_pois])/fas_new, (hb[cond_pois]-fls)/fas_new
        if contamh[i]<0 or np.isnan(contamh[i]): contamh[i] = 0.0
        if contaml[i]<0 or np.isnan(contaml[i]): contaml[i] = 0.0
        if fls==0: contamh[i], contaml[i] = 0.0, 0.0
    contamf = interp1d(bin_centers, contam, kind=interp_type, fill_value=(contam[0], 1.0), bounds_error=False)
    contamhf = interp1d(bin_centers, contamh, kind=interp_type, fill_value=(contamh[0], 0.0), bounds_error=False)
    contamlf = interp1d(bin_centers, contaml, kind=interp_type, fill_value=(contaml[0], 0.0), bounds_error=False)

    test_contam = np.linspace(bin_edges[0], bin_edges[-1], test_contam_num)
    ctc = contamf(test_contam)
    if ctc.min() > contam_lim: 
        nbcontam = -99.0 #Super bright magnitude in case we never hit contam lim
    else:
        indcontam = np.argmin(np.abs(ctc - contam_lim))
        nbcontam = test_contam[indcontam]

    # # pois = poisson.rvs(flss, size=(nsamp, binnum))
    # maxs = poisson.cdf(fassn, flss)
    # u = uniform.rvs(scale=maxs, size=(nsamp, binnum))
    # pois = poisson.ppf(u, flss)
    # contamsamp = pois / fassn
    # nbcsamp = -99.0 * np.ones(nsamp)
    # for i in range(nsamp):
    #     contamsf = interp1d(bin_centers, contamsamp[i], kind=interp_type, fill_value=(contam[0], 1.0), bounds_error=False)
    #     ctc = contamsf(test_contam)
    #     if ctc.min()<=contam_lim:
    #         indcs = np.argmin(np.abs(ctc - contam_lim))
    #         nbcsamp[i] = test_contam[indcs]

    # obj = {}
    # obj['mags'], obj['contams'], obj['nbcontams'] = bin_centers, contamsamp, nbcsamp
    # pickle.dump(obj, open(f'{filter}_contamination_samp.pickle', 'wb'))

    # print(f"nbcontam: {nbcontam:0.2f}")

    plt.figure(figsize=(6,6))
    plt.errorbar(bin_centers, contam, yerr=np.row_stack((contaml, contamh)), xerr=np.row_stack((bin_centers-bin_edges[:-1], bin_edges[1:]-bin_centers)), fmt='bs')
    for i, bc in enumerate(bin_centers):
        if contam[i] > 0.5: locy = contam[i] - contaml[i]-0.05
        else: locy = contam[i] + contamh[i] + 0.05
        plt.text(bc, locy, fr'$\frac{{{flss[i]}}}{{{fass[i]}}}$', color='k', horizontalalignment='center')
    bin_check = np.linspace(bin_edges.min(), bin_edges.max(), 1001)
    plt.plot(bin_check, contamf(bin_check), 'r')
    plt.fill_between(bin_check, contamf(bin_check)-contamlf(bin_check), contamf(bin_check)+contamhf(bin_check), color='r', alpha=0.1)
    if filter=='N673': plt.gca().set_xticks(plt.gca().get_xticks()[:-2])
    plt.xlim(bin_check.max(), bin_check.min())
    plt.ylim(-0.05, 1.05)
    plt.xlabel('NB Magnitude (AB)', fontsize='large')
    plt.ylabel('Fraction of true LAEs', fontsize='large') # in {filter}')
    plt.savefig(op.join('Contamination', f'{filter}_Contam_{binnum}_{contam_type}_final_v2.png'), bbox_inches='tight', dpi=300)
    # breakpoint()
    return contamf, contamhf, contamlf, nbcontam

def getRealLumRed(file_name='N501_Nicole.txt', interp_type='cubic', wav_rest=1215.67, delznum=51):
    ''' Determine the true luminosity of a source given a transmission curve and a redshift (this routine creates interpolation functions that can be used to evaluate the result at (a) desired redshift(s))'''
    trans_dat = Table.read(file_name, format='ascii')
    lam, tra = trans_dat['lambda'], trans_dat['transmission']
    cond = tra>0.0
    lam, tra = lam[cond], tra[cond]
    zs = (lam-wav_rest) / wav_rest
    if 'perfect' in file_name.lower():
        zmin, zmax = zs.min(), zs.max()
        return lambda x: np.piecewise(x, [x<=zmin, x<zmax, x>=zmax], [np.inf, 0.0, np.inf])
    del_logL = np.log10(tra.max()) - np.log10(tra)
    del_logL_lin = np.linspace(del_logL.min(), del_logL.max(), delznum)
    delz = np.zeros_like(del_logL_lin)
    for i, tv in enumerate(del_logL_lin):
        inds = np.where(del_logL<=tv)[0]
        inds_consec = consecutive(inds)
        zs_consec = [zs[indsi] for indsi in inds_consec]
        delz[i] = sum([zsi[-1] - zsi[0] for zsi in zs_consec])
    delzf = interp1d(del_logL_lin, delz, kind=interp_type, bounds_error=False, fill_value = (0, delz.max()))
    return interp1d(zs, del_logL, kind=interp_type, bounds_error=False, fill_value = (del_logL[0], del_logL[-1])), delzf, zs[np.argmax(tra)]

def getTransPDF(lam, tra, pdflen=10000, num_discrete=51, interp_type='cubic', wav_rest=1215.67):
    ''' Given a transmission curve, determine the probability distribution function of the true luminosity of a source being higher (assuming that sources are distributed uniformly along the redshift axis)'''
    del_logL = np.log10(tra.max()) - np.log10(tra)
    il, ir = 0, len(del_logL)-1
    while del_logL[il+1]-del_logL[il]<0.0 or del_logL[il+1]>(del_logL.max()-del_logL.min())/2.0: il+=1
    while del_logL[ir-1]-del_logL[ir]<0.0 or del_logL[ir-1]>(del_logL.max()-del_logL.min())/2.0: ir-=1

    flat_frac = (lam[ir]-lam[il])/(lam[-1]-lam[0])

    fl = interp1d(del_logL[:il],lam[:il],kind=interp_type)
    fr = interp1d(del_logL[ir:],lam[ir:],kind=interp_type)

    del_logL_arr_orig = np.linspace(max(del_logL[:il].min(),del_logL[ir:].min()), min(del_logL[:il].max(),del_logL[ir:].max()),pdflen)
    del_z_arr = (fr(del_logL_arr_orig) - fl(del_logL_arr_orig))/wav_rest
    del_logL_arr = 1.0 * del_logL_arr_orig

    del_logL_arr_diff = np.diff(del_logL_arr)
    pdf_arr = np.diff(fr(del_logL_arr))/del_logL_arr_diff - np.diff(fl(del_logL_arr))/del_logL_arr_diff
    pdf_arr = np.append(pdf_arr, pdf_arr[-1])
    # del_logL_arr = np.append(del_logL_arr, tra.max())
    # pdf_arr = np.append(pdf_arr, pdf_arr[-1]) # Assume the PDF stays constant for the small sliver constituting the mostly flat top
    if del_logL_arr[0]-del_logL.min()>1.0e-12:
        del_logL_arr = np.insert(del_logL_arr, 0, del_logL.min())
        pdf_arr = np.insert(pdf_arr, 0, pdf_arr[0])
    integ = trapezoid(pdf_arr[1:], del_logL_arr[1:])
    pdf_arr[1:] *= (1.0-flat_frac) / integ # Normalize
    # pdf_arr[0] = flat_frac/(1.0-flat_frac) * integ / (del_logL_arr[1]-del_logL_arr[0])
    pdf_arr[0] = flat_frac / (del_logL_arr[1]-del_logL_arr[0])
    pdf_arr /= trapezoid(pdf_arr, del_logL_arr) # Just normalize again since the trapezoid rule is not a perfect integrator by any means

    # If in fact there is no flat top part, we will run into issues
    if pdf_arr[0]<1.0e-10: 
        pdf_arr = np.delete(pdf_arr, 0)
        del_logL_arr = np.delete(del_logL_arr, 0)
    log_pdf = np.log10(pdf_arr)
    diff_log = np.hstack([abs(np.diff(log_pdf)),0.0])
    diff_cumsum = np.cumsum(diff_log)/diff_log.sum() #Normalized cumulative sum
    logL_discrete = np.zeros(num_discrete)
    logL_discrete[-1] = del_logL_arr.max()
    indi_arr = np.zeros(del_logL_arr.size,dtype=int)
    for i in range(1,num_discrete-1):
        indi = np.argmin(abs(diff_cumsum-i/num_discrete))
        if indi<=indi_arr[i-1]:
            while indi<=indi_arr[i-1]: indi+=1
        indi_arr[i] = indi
        logL_discrete[i] = del_logL_arr[indi]

    # pdf_arr_sort, indsort = np.unique(pdf_arr, return_index=True)
    # f_reverse = interp1d(pdf_arr_sort,del_logL_arr[indsort],kind='cubic',fill_value=0.0,bounds_error=False)
    # pdf_even_space = np.linspace(pdf_arr.min(), pdf_arr.max(), num_discrete)
    # logL_discrete = f_reverse(pdf_even_space)

    return interp1d(del_logL_arr, pdf_arr, kind=interp_type, fill_value=(pdf_arr[0], 0.0), bounds_error=False), logL_discrete, interp1d(del_logL_arr_orig, del_z_arr, kind=interp_type, fill_value=(del_z_arr[0],del_z_arr[-1]), bounds_error=False)

def getBoundsTransPDF(logL_width=2.0, file_name='N501_Nicole.txt', pdflen=100000, fulllen=10000, wav_rest=1215.67, maglen=101, num_discrete=51):
    ''' Partner routine with getTransPDF that determines the transmission curve based on input files (and only includes the desired extent in the wings) '''
    trans_dat = Table.read(file_name, format='ascii')
    lam, trans = trans_dat['lambda'], trans_dat['transmission']
    cond = trans>0.0
    lam, trans = lam[cond], trans[cond]
    transf = interp1d(lam, trans, kind='cubic', bounds_error=False, fill_value=0.0)
    lam_full = np.linspace(lam[0],lam[-1],fulllen)
    trans_max = trans.max()
    trans_min = 10**(-1.0*logL_width) * trans_max
    trans_full = transf(lam_full)
    tfam = np.argmax(trans_full)
    left_ind = np.argmin(abs(trans_full[:tfam+1]-trans_min))
    right_ind = np.argmin(abs(trans_full[tfam:]-trans_min)) + tfam

    # logLs = np.linspace(0.0,logL_width,maglen)
    # delz = np.zeros_like(logLs)
    # for i, logL in enumerate(logLs):
    #     trans_mini = 10**(-1.0*logL) * trans_max
    #     left_indi = np.argmin(abs(trans_full[:tfam+1]-trans_mini))
    #     right_indi = np.argmin(abs(trans_full[tfam:]-trans_mini)) + tfam
    #     delz[i] = (lam[right_indi]-lam[left_indi])/wav_rest
    interp_type = 'linear'
    transpdf, logL_discrete, delzf = getTransPDF(lam_full[left_ind:right_ind], trans_full[left_ind:right_ind], pdflen=pdflen, num_discrete=num_discrete, interp_type=interp_type, wav_rest=wav_rest)

    # filt = file_name.split('_')[0]
    # transfigs = {'filter': filt, 'lam': lam, 'trans': trans, 'logL_discrete': logL_discrete, 'dz': delzf(logL_discrete), 'pdf': transpdf(logL_discrete)}
    # pickle.dump(transfigs, open(f'FilterFig{filt}.pickle', 'wb'))
    # makeTransFigs(filter, lam, trans, logL_discrete, delzf(logL_discrete), transpdf(logL_discrete))

    return transpdf, logL_discrete, delzf # (lam_full[right_ind]-lam_full[left_ind])/wav_rest #, interp1d(logLs, delz, bounds_error=False, fill_value=(delz[0],delz[-1]))

class RGINNExt:
    ''' Multi-dimensional linear interpolation within convex hull of points and nearest point within hull for points outside; function created with help of stackoverflow '''
    def __init__( self, points, values, method='cubic' ):
        self.interp = RGIScipy(points, values, method=method,
                                              bounds_error=False, fill_value=np.nan)
        self.nearest = RGIScipy(points, values, method='nearest',
                                           bounds_error=False, fill_value=None)
        
    def __call__( self, xi ):
        vals = self.interp( xi )
        idxs = np.isnan( vals )
        if type(xi)==tuple: vals[idxs] = self.nearest((xi[0][idxs], xi[1][idxs]))
        else: vals[idxs] = self.nearest( xi[idxs] )
        return vals

def makeCompFuncSamp(num, DL, file_name='cosmos_completeness_n501_grid_extrap_samp.pickle', filter='N501', wave=1215.67, dwave=73.0, distnum=21, magnum=1001, contam_lim=0.01, mag_min=28., mag_max=21., use_contam=True, aper_corr=0.0, interp_type='linear'):
    ''' In an experiment to understand the effects of uncertainties on completeness and contamination, get the effective completeness curve for a particular realization of completeness'''
    with open(file_name,'rb') as f:
        dat = pickle.load(f)
    mag, dist, comp = dat['Mags']+aper_corr, dat['Dist'], dat['CompSamps'][num]
    if use_contam:
        with open(f'{filter}_contamination_samp.pickle', 'rb') as f:
            obj = pickle.load(f)
        bin_centers, contam, nbcontam = obj['mags'], obj['contams'][num], obj['nbcontams'][num]
        cf = interp1d(bin_centers, contam, kind=interp_type, fill_value=1.0, bounds_error=False)
    interp_comp = RGINNExt((dist, mag), comp)
    interp_comp_simp_orig = RectBivariateSpline(dist, mag, comp, kx=1, ky=1)
    distcontam = np.linspace(dist.min(), dist.max(), distnum)
    magcontam = np.linspace(mag.min(), mag.max(), magnum)
    if use_contam:
        dc, mc = np.meshgrid(distcontam, magcontam, indexing='ij')
        # cgs17 = magAB2cgs(mc, wave, dwave)*1.0e17
        contampart = 1.0/cf(mc)
        contampart[mc<nbcontam] = 1.0/contam_lim

        vals = interp_comp_simp_orig.ev(dc, mc) * contampart
        interp_comp_simp = RectBivariateSpline(distcontam, magcontam, vals, kx=1, ky=1)
    else: interp_comp_simp = interp_comp_simp_orig

    # plot_Comp(interp_comp_simp, mag, comp, dist, DL, f'{filter}_{num}', wave=wave, dwave=dwave, mag_min=mag_min, mag_max=mag_max)
    return interp_comp, interp_comp_simp_orig, interp_comp_simp, nbcontam, cf

def makeCompFunc(DL, file_name='cosmos_completeness_grid_extrap.pickle', binnum=5, filter='N501', wave=1215.67, dwave=73.0, distnum=21, magnum=1001, contam_lim=0.01, contam_type='L_LCA', mag_min=28., mag_max=21., density_frac=1.0, use_contam=True, aper_corr=0.0):
    ''' Determine the effective completeness curve (or just completeness if use_contam is False); plot this curve'''
    with open(file_name,'rb') as f:
        dat = pickle.load(f)
    mag, dist, comp = dat['Mags']+aper_corr, dat['Dist'], dat['Comp']
    # fig = plt.figure()
    # sc = plt.contourf(mag, dist, np.log10(comp), levels=10)
    # plt.colorbar(sc, label='Modified completeness')
    # plt.xlabel('Magnitude')
    # plt.ylabel('Distance from center of field')
    if use_contam: cf, chf, clf, nbcontam = getContamination(filter=filter, binnum=binnum, contam_lim=contam_lim, contam_type=contam_type, density_frac=density_frac, mag_corr=0.0)
    else: nbcontam, cf = -99.0, None
    interp_comp = RGINNExt((dist, mag), comp)
    interp_comp_simp_orig = RectBivariateSpline(dist, mag, comp, kx=1, ky=1)
    distcontam = np.linspace(dist.min(), dist.max(), distnum)
    magcontam = np.linspace(mag.min(), mag.max(), magnum)
    if use_contam:
        dc, mc = np.meshgrid(distcontam, magcontam, indexing='ij')
        # cgs17 = magAB2cgs(mc, wave, dwave)*1.0e17
        contampart = 1.0/cf(mc)
        contampart[mc<nbcontam] = 1.0/contam_lim

        vals = interp_comp_simp_orig.ev(dc, mc) * contampart
        interp_comp_simp = RectBivariateSpline(distcontam, magcontam, vals, kx=1, ky=1)
    else: interp_comp_simp = interp_comp_simp_orig
    # fig2 = plt.figure()
    # sc = plt.contourf(magcontam, distcontam, np.log10(vals), levels=10)
    # plt.colorbar(sc, label='Modified completeness')
    # plt.xlabel('Magnitude')
    # plt.ylabel('Distance from center of field')
    # plt.show()
    # plt.close('all')
    plot_Comp(interp_comp_simp, mag, comp, dist, DL, filter, wave=wave, dwave=dwave, mag_min=mag_min, mag_max=mag_max)
    return interp_comp, interp_comp_simp_orig, interp_comp_simp, nbcontam, cf

def makeCompFuncMag(DL, file_name='shela_completeness_n501_region1.pickle', binnum=5, filter='N501', contam_lim=0.01, contam_type='L_LCA', density_frac=1.0, use_contam=True, aper_corr=0.0, interp_type='linear', label='', mag_min=28., mag_max=20., wave=1215.67, dwave=73.0):
    ''' Determine a magnitude-only effective completeness curve.

    The pickle file should contain two 1D arrays:
      - magnitude: one of [Mags, mag, mags, Magnitude]
      - completeness: one of [Comp, comp, Completeness, completeness]
    '''
    with open(file_name,'rb') as f:
        dat = pickle.load(f)

    mag_key, comp_key = None, None
    for key in ['Mags', 'mag', 'mags', 'Magnitude']:
        if key in dat:
            mag_key = key
            break
    for key in ['Comp', 'comp', 'Completeness', 'completeness']:
        if key in dat:
            comp_key = key
            break
    if mag_key is None or comp_key is None:
        raise KeyError(f'Could not find magnitude/completeness keys in {file_name}')

    mag = np.array(dat[mag_key][:-1], dtype=float) + aper_corr
    comp = np.array(dat[comp_key][:-1], dtype=float)
    inds = np.argsort(mag)
    mag, comp = mag[inds], comp[inds]

    comp_orig = interp1d(mag, comp, kind=interp_type, bounds_error=False, fill_value=(comp[0], comp[-1]))
    if use_contam:
        cf, chf, clf, nbcontam = getContamination(filter=filter, binnum=binnum, contam_lim=contam_lim, contam_type=contam_type, density_frac=density_frac, mag_corr=0.0)
        comp_use_arr = comp / cf(mag)
        comp_use_arr[mag<nbcontam] = comp[mag<nbcontam] / contam_lim
    else:
        nbcontam, cf = -99.0, None
        comp_use_arr = comp
    comp_use_arr = np.clip(comp_use_arr, 0.0, 1.0e3)
    comp_use = interp1d(mag, comp_use_arr, kind=interp_type, bounds_error=False, fill_value=(comp_use_arr[0], comp_use_arr[-1]))
    # comp_use = interp1d(mag, comp_use_arr, kind=interp_type, bounds_error=False, fill_value=(comp_use_arr[0], 0.0))
    plot_Comp(comp_use, mag, comp, None, DL, filter, wave=wave, dwave=dwave, mag_min=mag_min, mag_max=mag_max, label=label)
    return comp_use, comp_orig, comp_use, nbcontam, cf

def cgs2magAB(cgs, wave, dwave):
    ''' cgs flux to AB magnitude conversion '''
    Flam = cgs/dwave
    Fnu = Flam*wave**2/c
    return -2.5*np.log10(Fnu)-48.6

def magAB2cgs(mag, wave, dwave):
    ''' AB magnitude to cgs flux conversion '''
    Fnu = 10**(-0.4*(mag+48.6))
    Flam = Fnu*c/wave**2
    return Flam * dwave

def lum2cgs(lum, DL):
    ''' Luminosity to cgs flux conversion given a luminosity distance '''
    return 10**lum / (4.0*np.pi*(3.086e24*DL)**2)

def cgs2lum(cgs, DL):
    ''' cgs flux to luminosity conversion given luminosity distance'''
    return np.log10(cgs * 4.0*np.pi*(3.086e24*DL)**2)

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
    return np.log(10.0) * 10**logphistar * 10**((logL-logLstar)*(alpha+1))*np.exp(-10**(logL-logLstar))

def TrueLumFuncNoPhi(logL,alpha,logLstar):
    ''' Same as truelumfunc but with log(phi*)=0 '''
    return np.log(10.0) * 10**((logL-logLstar)*(alpha+1))*np.exp(-10**(logL-logLstar))

def Omega(logL,dLz,compfunc,Omega_0,wave,dwave):
    ''' Calculate fractional area of the sky in which galaxies have fluxes large enough so that they can be detected

    Input
    -----
    logL : float or numpy 1-D array
        Value or array of log luminosities in erg/s
    dLz : float
        Luminosity distance (for a given z=z0) in Mpc
    compfunc: interp1d instance
        1-D interpolation function for average completeness vs magnitude
    Omega_0: float
        Effective survey area in square arcseconds

    Returns
    -------
    Omega(logL,z0) : Float or 1-D array (same size as logL)
    '''
    if callable(compfunc): 
        flux_cgs = lum2cgs(logL, dLz)
        mags = cgs2magAB(flux_cgs, wave, dwave)
        comp = compfunc(mags)
    else: 
        comp = compfunc
    return Omega_0/V.sqarcsec * comp

def normalFunc(x,mu,sig):
    ''' Normal probability distribution function '''
    return 1.0/(np.sqrt(2.0*np.pi)*sig) * np.exp(-(x-mu)**2/(2.0*sig**2))

def plot_Comp(compf, mag, comp, dist, DL, fn, mag_min=28., mag_max=20., wave=1215.67, dwave=73.0, label=''):
    ''' Plot (effective) completeness curve '''
    magarr = np.linspace(mag_min, mag_max, 31)
    cgs = magAB2cgs(magarr, wave=wave, dwave=dwave)
    lumarr = cgs2lum(cgs, DL)
    lumvals = cgs2lum(magAB2cgs(mag, wave=wave, dwave=dwave), DL)
    fig, ax = plt.subplots(figsize=(6,6))
    if dist is not None:
        cmap = plt.cm.plasma
        norm = plt.Normalize(vmin=dist.min(), vmax=dist.max())
        colors = cmap(norm(dist))
        for i, d in enumerate(dist):
            # ax.scatter(lumvals, comp[i], c=colors[i], s=10)
            ax.plot(lumarr, compf.ev(d, magarr), color=colors[i])
    else:
        ax.plot(lumarr, compf(magarr), 'b-', label=label)
        ax.legend(loc='best')
    ax.set_yscale('log')
    ax.set_xlim(lumarr.min(), lumarr.max())
    ax.set_ylim(1.0e-3, 2.2)
    if dist is not None:
        cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
        cb.set_label('Distance from center (arcmin)', fontsize='large')
    ax.set_xlabel(r'Log Luminosity (erg s$^{-1}$)', fontsize='large')
    ax.set_ylabel('Effective Completeness', fontsize='large')
    fig.savefig(f'{fn}_EffComp.png',bbox_inches='tight',dpi=300)
    plt.close(fig)
    # breakpoint()

class LumFuncMCMC:
    ''' A class to facilitate the calculation of the luminosity function given input fluxes and other information '''
    def __init__(self, z, del_red=None, flux=None, flux_e=None, nb=None, nb_e=None, line_name="OIII", line_plot_name=r'[OIII] $\lambda 5007$', lum=None, lum_e=None, Omega_0=43200., nbins=50, nboot=100, sch_al=-1.6, sch_al_lims=[-3.0,1.0], Lstar=42.5, Lstar_lims=[40.0,45.0], phistar=-3.0, phistar_lims=[-8.0,5.0], Lc=40.0, Lh=46.0, nwalkers=100, nsteps=1000, fix_sch_al=False, min_comp_frac=0.5, diff_rand=True, field_name='COSMOS', interp_comp=None, interp_comp_simp=None, interp_comp_simp_orig=None, dist_orig=None, dist=None, maglow=26.0, maghigh=19.0, magnum=25, distnum=100, comps=None, size_ln=1001, wav_filt=5015.0, filt_width=73.0, binned_stat_num=50, err_corr=False, wav_rest=1215.67, size_ln_conv=41, size_lprime=51, logL_width=2.0, trans_only=False, norm_only=False, trans_file='N501_Nicole.txt', maxlum=None, minlum=None, transsim=False, corrf=None, corref=None, flux_lim=15.0, T_EL=1.0, alls_file_name=None, vgal_file_name=None, weight=None, contam_lim=0.01, contambin=5, cgscontam=1.0, cf=None, contam_type='L_LCA', varying=False, density_frac=1.0, aper_corr=0.0, beta=[1.0, 0.0], extra_text='', frac_use=1.0, minlum2=None, frac1=1.0, frac2=None, comps2=None, interp_comp_simp2=None, comp_region=None):
        ''' Initialize LumFuncMCMC class

        Init
        ----
        z : Float
            Redshift of objects in field
        del_red: Float
            Width of redshift bin
        flux : 1-D Numpy array or None Object
            Array of fluxes in 10^-17 erg/cm^2/s
        flux_e : 1-D Numpy array or None Object
            Array of flux errors in 10^-17 erg/cm^2/s
        line_name: string
            Name of line or monochromatic luminosity element
        line_plot_name: (raw) string
            Fancier name of line or luminosity element for putting in plot labels
        lum: numpy array (1 dim) or None Object
            Array of log luminosities in erg/s
        lum_e: numpy array (1 dim) or None Object
            Array of log luminosity errors in erg/s
        Omega_0: Float
            Effective survey area in square arcseconds
        nbins: int
            Number of bins for plotting luminosity function and conducting V_eff method
        nboot: int
            Number of iterations for bootstrap method for determining errors for V_eff method
        sch_al: float
            Schechther alpha parameter
        sch_al_lims: two-element list
            Minimum and maximum values allowed in Schechter alpha prior
        Lstar: float
            Schechther log(Lstar) parameter
        Lstar_lims: two-element list
            Minimum and maximum values allowed in Schechter log(Lstar) prior
        phistar: float
            Schechther log(phistar) parameter
        phistar_lims: two-element list
            Minimum and maximum values allowed in Schechter log(phistar) prior
        Lc, Lh: floats
            Minimum and maximum log luminosity, respectively, for likelihood integral
        nwalkers : int
            The number of walkers for emcee when fitting a model
        nsteps : int
            The number of steps each walker will make when fitting a model
        fix_sch_al: Bool
            Whether or not to fix the alpha parameter of true luminosity function
        min_comp_frac: Float
            Minimum completeness fraction considered
        field_name: String
            Name of field
        interp_comp: Interpolation function
            Interpolation function for completeness
        comp1d: Interpolation function
            1-D interpolation function for completeness averaged over distance from field center
        dist: 1-D Numpy Array
            Array of distances from center of field
        maglow, maghigh: Floats
            Min and max magnitudes for distance-averaging
        comps: 1-D Numpy Array
            Array of completeness values for all objects
        wav_filt: Float
            Central wavelength of filter
        '''
        self.z, self.del_red = z, del_red
        self.min_comp_frac = min_comp_frac
        self.line_name = line_name
        self.line_plot_name = line_plot_name
        self.Lc, self.Lh = Lc, Lh
        self.Omega_0, self.frac_use = Omega_0, frac_use
        self.Omega_0_sr = Omega_0/V.sqarcsec
        self.nbins, self.nboot = nbins, nboot
        self.sch_al, self.sch_al_lims = sch_al, sch_al_lims
        self.Lstar, self.Lstar_lims = Lstar, Lstar_lims
        self.phistar, self.phistar_lims = phistar, phistar_lims
        self.nwalkers, self.nsteps = nwalkers, nsteps
        self.fix_sch_al = fix_sch_al
        self.all_param_names = ['Lstar','phistar','sch_al']
        self.diff_rand = diff_rand
        self.field_name = field_name
        self.dist, self.dist_orig, self.distnum = dist, dist_orig, distnum
        self.maglow, self.maghigh, self.magnum = maglow, maghigh, magnum
        self.comps, self.size_ln, self.wav_filt, self.comps2 = comps, size_ln, wav_filt, comps2
        self.interp_comp_simp2 = interp_comp_simp2
        self.comp2df = None
        self.comp_region = comp_region
        self.size_ln_conv, self.size_lprime = size_ln_conv, size_lprime
        self.filt_width, self.binned_stat_num = filt_width, binned_stat_num
        self.err_corr, self.wav_rest, self.logL_width = err_corr, wav_rest, logL_width
        self.trans_only, self.norm_only = trans_only, norm_only
        self.transf, self.logL_discrete, self.delzf = getBoundsTransPDF(logL_width=self.logL_width,wav_rest=self.wav_rest,num_discrete=self.size_lprime,file_name=trans_file)
        # self.filt_width_eff = self.del_red_eff * self.wav_rest
        self.maxlum, self.minlum, self.transsim, self.minlum2 = maxlum, minlum, transsim, minlum2
        self.frac1, self.frac2 = frac1, frac2
        self.corrf, self.corref = corrf, corref
        self.flux_lim = flux_lim*1.0e-17 #In cgs
        self.logLfuncz, self.delzfv2, self.ztmax = getRealLumRed(file_name=trans_file, wav_rest=self.wav_rest, delznum=self.size_lprime)
        self.filt_name = trans_file.split('_')[0]
        self.delz_use = self.delzf(self.logL_width)
        self.T_EL, self.weight = T_EL, weight
        self.varying, self.extra_text = varying, extra_text
        self.aper_corr, self.beta = aper_corr, beta
        
        self.setDLdVdz()
        print("Finished DL, dVdz")

        if interp_comp is None: 
            self.interp_comp, self.interp_comp_simp_orig, self.interp_comp_simp, self.nbcontam, self.cf = makeCompFunc(binnum=contambin, filter=self.filt_name, wave=wav_rest, dwave=filt_width, contam_lim=contam_lim, contam_type=contam_type, density_frac=density_frac, aper_corr=self.aper_corr)
            cgscontam = magAB2cgs(self.nbcontam, self.wav_filt, self.filt_width)
            lumcontam = cgs2lum(cgscontam, self.DL)
            if flux is not None: condcontam = flux <= cgscontam*1.0e17
            else: condcontam = lum <= np.log10(lumcontam)
        else: 
            self.interp_comp, self.interp_comp_simp_orig, self.interp_comp_simp, self.nbcontam, self.cf = interp_comp, interp_comp_simp_orig, interp_comp_simp, cgscontam, cf #Giant max flux in case of not calculating value
            # Have already taken care of the condition in this case
            if flux is not None: condcontam = flux < np.inf
            else: condcontam = lum < np.inf
        print("Got completeness")
        if flux is not None: 
            self.flux, self.nb = 1.0e-17*flux[condcontam], 1.0e-17*nb[condcontam]
            if flux_e is not None:
                self.flux_e, self.nb_e = 1.0e-17*flux_e[condcontam], 1.0e-17*nb_e[condcontam]
        else:
            self.lum = lum[condcontam]
            if lum_e is not None: self.lum_e = lum_e[condcontam]
            else: self.lum_e = None
            self.getFluxes()
            self.nb, self.nb_e = None, None
        if lum is None: 
            self.getLumin()
        self.N = self.lum.size
        print("Finished getting fluxes and luminosities")
        if self.nb is None: self.mags = cgs2magAB(self.flux, self.wav_filt, self.filt_width) # For the completeness
        else: self.mags = cgs2magAB(self.nb, self.wav_filt, self.filt_width)
        if self.comps is None: self.comps = self.interp_comp_simp.ev(self.dist, self.mags)
        
        # Modify fluxes based on contamination limit
        self.alls_file_name, self.vgal_file_name = alls_file_name, vgal_file_name
        self.getalls()
        
        if not self.transsim:
            self.get1DComp()
        else:
            if self.minlum is None: self.getCompInfo()
            else: self.Omega_arr = self.weight * Omega(self.lum,self.DL,self.comps,self.Omega_0,self.wav_filt,self.filt_width)
            print("Finished getting Omega array")
        self.setup_logging()

    def getalls(self):
        # alls_file_name = f'Likes_alls_field{self.field_name}_z{self.z}_mcf{self.min_comp_frac}_fl{self.flux_lim}_tel{self.T_EL}_vgal.pickle'
        try:
            with open(self.alls_file_name, 'rb') as f:
                alls_output = pickle.load(f)
            with open(self.vgal_file_name, 'rb') as f:
                alls_output2 = pickle.load(f)
        except:
            return
        als, lss, likes = alls_output['Alphas'], alls_output['Lstars'], alls_output['likelihoods']
        vgal = alls_output2['Vgal']
        self.likeallsf = RectBivariateSpline(als, lss, likes)
        self.vgalf = RectBivariateSpline(als, lss, vgal)
        del alls_output, alls_output2
        self.plotLike(lss, als, likes, vgal, nameext=self.extra_text)

    def getCompInfo(self, compcut=0.03):
        ''' Create several arrays that will be used in efficient calculations of luminosity functions (numpy shenanigans)'''
        self.maggrid = np.linspace(self.maghigh, self.maglow, self.magnum)
        # distgrid = np.sort(np.random.choice(self.dist_orig, size=self.distnum))
        self.distgrid = np.linspace(self.dist_orig.min(), self.dist_orig.max(), num=self.distnum)
        self.distg, self.magg = np.meshgrid(self.distgrid, self.maggrid, indexing='ij')
        if self.interp_comp_simp2 is None:
            comps = self.interp_comp_simp.ev(self.distg, self.magg)
            cond = self.comps<=self.min_comp_frac + compcut
            minlums = cgs2lum(self.flux[cond], self.DL)
            if self.minlum is None:
                self.minlum = np.median(minlums)
                inds = np.argsort(self.dist[cond])
                distuse = self.dist[cond][inds]
                self.minlumf = interp1d(distuse, minlums, fill_value=(minlums[0], minlums[-1]), bounds_error=False)
            else: self.minlumf = lambda x: self.minlum*np.ones_like(x)
            comp_avg_dist = np.average(comps,axis=0)
            self.comp1df = interp1d(self.maggrid, comp_avg_dist, bounds_error=False, fill_value=(comp_avg_dist[0], comp_avg_dist[-1]))
        else:
            self.comp1df = self.interp_comp_simp
            self.comp2df = self.interp_comp_simp2
        self.comps1d = self.comp1df(self.mags)
        self.Omega_arr = self.weight * Omega(self.lum,self.DL,self.comps,self.Omega_0,self.wav_filt,self.filt_width)
        self.logL = np.linspace(self.minlum,self.Lh,self.size_ln)
        self.Omega_gen = Omega(self.logL,self.DL,self.comp1df,self.Omega_0,self.wav_filt,self.filt_width)

    def getminlum_z_func(self): 
        ''' This functionality of treating the minimum luminosity as a function of redshift is not used given the computational expenses '''
        comps = self.interp_comp_simp.ev(self.distg, self.magg)
        roots = np.zeros((self.size_lprime, self.distnum))
        minlums = np.zeros((self.size_lprime, self.distnum))
        for i in range(self.size_lprime):
            if i%10==0: print(f"Gotten to outer loop number {i} in minlum 2d calculation")
            for j in range(self.distnum):
                comps_use = self.trans_vals[i] * comps[j]
                if comps_use.max() > self.min_comp_frac:
                    func = interp1d(self.maggrid, comps_use, bounds_error=False, fill_value=(comps_use[0], comps_use[-1]))
                    roots[i,j] = fsolve(lambda x: func(x)-self.min_comp_frac, [25.0])[0]
            fluxes = magAB2cgs(roots[i], self.wav_filt, self.filt_width)
            minlumsi = cgs2lum(fluxes, self.DLs[i])
            minlums[i] = np.clip(minlumsi, self.Lc, self.Lh)
        self.minlum2df = RectBivariateSpline(self.zarr, self.distgrid, minlums)

    def getmaxlum_z_func(self):
        ''' Ditto as minimum--not used for computational reasons'''
        lums = cgs2lum(self.flux_lim, self.DLs)
        self.maxlumf = interp1d(self.zarr, lums, kind='linear', bounds_error=False, fill_value=(lums[0], lums[-1]))

    def get1DComp(self):
        ''' Get LAE-point-averaged estimates of the 1-D completeness function (of magnitude) '''
        print("Setting the computational arrays")
        self.getCompInfo()
        ########### Things for new version of transmission convolution ###########
        cgs = lum2cgs(self.logL, self.DL)
        mags = cgs2magAB(cgs, self.wav_filt, self.filt_width)
        if hasattr(self.interp_comp_simp, 'ev'):
            self.Omega_full = self.Omega_0_sr * np.average(self.interp_comp_simp.ev(self.dist[:,None], mags), axis=0)
        else:
            self.Omega_full = self.Omega_0_sr * self.comp1df(mags)
        self.trans_vals = 10**(-self.logLfuncz(self.zarr)) / self.T_EL
        self.trans_mult = self.trans_vals * self.dVdzs
        # self.ptransmult = self.Omega_0_sr * self.comps_full[:,None] * self.trans_mult
        # self.ptransmult = self.Omega_0_sr * self.comps_full * np.average(self.dVdzs)

        ########### For convolution part ###########
        # self.logL_conv = np.linspace(self.minlum_conv,self.Lh,self.size_ln_conv)
        # self.logL_conv_all = np.zeros((self.size_ln_conv,self.size_lprime))
        # for i in range(self.size_ln_conv):
        #     self.logL_conv_all[i] = np.linspace(self.logL_conv[i],self.logL_conv[i]+self.logL_width,self.size_lprime)
        # L_all = 10**self.logL_conv_all
        # flux_cgs = L_all/(4.0*np.pi*(3.086e24*self.DL)**2)
        # mags = cgs2magAB(flux_cgs, self.wav_filt, self.filt_width)
        # self.comp_conv = self.comp1df(mags)
        # self.trans_conv = self.transf(self.logL_discrete)
        # self.norm_vals = normalFunc(self.logL_conv[None],self.lum[:,None],self.lum_e[:,None])

        ####### Just transmission convolution #######
        self.trans_conv = self.transf(self.logL_discrete)
        self.logL_trans_lnpart = self.lum[:,None] + self.logL_discrete
        self.logL_trans_integ = self.logL[:,None] + self.logL_discrete
        
        flux_cgs = lum2cgs(self.logL_trans_lnpart, self.DL)
        mags = cgs2magAB(flux_cgs, self.wav_filt, self.filt_width)
        self.comps_trans_lnpart = self.comp1df(mags)

        flux_cgs = lum2cgs(self.logL_trans_integ, self.DL)
        mags = cgs2magAB(flux_cgs, self.wav_filt, self.filt_width)
        self.comps_trans_integ = self.comp1df(mags)

        self.delz_trans = self.delzf(self.logL_trans_integ-self.minlum)
        self.not_tlf = self.comps_trans_integ * self.delz_trans * self.trans_conv

        ###### Just normal convolution #####
        self.logL_norm = np.zeros((self.lum.size,self.size_ln_conv))
        for i in range(self.lum.size):
            self.logL_norm[i] = self.lum[i] + np.linspace(-3.0*self.lum_e[i],3.0*self.lum_e[i],self.size_ln_conv)
        self.norm_vals_norm = normalFunc(self.logL_norm,self.lum[:,None],self.lum_e[:,None])
        flux_cgs = lum2cgs(self.logL_norm, self.DL)
        mags = cgs2magAB(flux_cgs, self.wav_filt, self.filt_width)
        self.comps_norm = self.comp1df(mags)

        ###### New combination of everything that is centered for normal distribution around the mean #####
        self.logL_conv = self.logL_norm[:,:,None] + self.logL_discrete
        flux_cgs = lum2cgs(self.logL_conv, self.DL)
        mags = cgs2magAB(flux_cgs, self.wav_filt, self.filt_width)
        self.comps_conv = self.comp1df(mags)
        
    def setDLdVdz(self):
        ''' Create 1-D interpolated functions for luminosity distance (cm) and comoving volume differential (Mpc^3); also get function for minimum luminosity considered '''
        print("Setting DL and dVdz")
        self.DL = V.cosmo.luminosity_distance(self.z).value
        self.dVdz = V.cosmo.differential_comoving_volume(self.z).value
        # if self.err_corr or self.trans_only: self.volume = self.dVdz * self.del_red_eff
        self.zarr = np.linspace(self.ztmax-0.55*self.delz_use, self.ztmax+0.55*self.delz_use, self.size_lprime)
        self.DLs = V.cosmo.luminosity_distance(self.zarr).value
        self.lum_lim = cgs2lum(self.flux_lim, self.DL)
        self.dVdzs = V.cosmo.differential_comoving_volume(self.zarr).value
        self.volume = self.dVdz * self.del_red # Actual total volume per steradian of the survey (redshift integral separate from luminosity function integral)

    def getLumin(self):
        ''' Set the sample log luminosities (and error if flux errors available)
            based on given flux values and luminosity distance values
        '''
        if self.flux_e is not None: 
            ulum = unumpy.log10(4.0*np.pi*(self.DL*3.086e24)**2 * unumpy.uarray(self.flux,self.flux_e))
            ulumnb = unumpy.log10(4.0*np.pi*(self.DL*3.086e24)**2 * unumpy.uarray(self.nb,self.nb_e))
            self.lum, self.lum_e = unumpy.nominal_values(ulum), unumpy.std_devs(ulum)
            self.lumnb, self.lumnb_e = unumpy.nominal_values(ulumnb), unumpy.std_devs(ulumnb)
            self.lum_bin_edges = np.percentile(self.lum,np.linspace(0.,100.,self.binned_stat_num+1))
            self.lum_err_bins, _, _ = binned_statistic(self.lum, self.lum_e, statistic='median',bins=self.lum_bin_edges)
            self.lum_bin_mid = np.array([(self.lum_bin_edges[i]+self.lum_bin_edges[i+1])/2.0 for i in range(self.binned_stat_num)])
            self.lum_err_func = interp1d(self.lum_bin_mid, self.lum_err_bins, bounds_error=False, fill_value=(self.lum_err_bins[0],self.lum_err_bins[-1]))
        else:
            self.lum = cgs2lum(self.flux, self.DL)
            self.lumnb = cgs2lum(self.nb, self.DL)
            self.lum_e, self.lumnb_e = None, None
            self.lum_bin_edges, self.lum_err_bins, self.lum_bin_mid, self.lum_err_func = None, None, None, None

    def getFluxes(self):
        ''' Set sample fluxes based on luminosities if not available '''
        if self.lum_e is not None:
            ulum = unumpy.uarray(self.lum,self.lum_e)
            uflux = lum2cgs(ulum, self.DL)
            self.flux, self.flux_e = unumpy.nominal_values(uflux), unumpy.std_devs(uflux)
        else:
            self.flux = lum2cgs(self.lum, self.DL)
            self.flux_e = None

    def calclikeLsal(self, alnum=50, lsnum=50):
        ''' Get log likelihood values for the alpha and L* parameters (shape part of the luminosity function) on a grid'''
        if self.comp_region is not None or (hasattr(self, 'comp2df') and self.comp2df is not None):
            return self.calclikeLsalShela(p=self.comp_region, alnum=alnum, lsnum=lsnum)
        self.normhist, bin_edges = np.histogram(self.lum, bins=self.nbins, density=True)
        self.Lmed = (bin_edges[:-1] + bin_edges[1:])/2.0
        als = np.linspace(self.sch_al_lims[0], self.sch_al_lims[1], alnum)
        lss = np.linspace(self.Lstar_lims[0], self.Lstar_lims[1], lsnum)
        # compgrid = np.zeros((len(self.dist), *self.logL_trans_integ.shape))
        compgrid = np.zeros((self.dist.size, self.logL.size))
        # L_all = 10**self.logL_trans_integ.ravel()
        flux_cgs_orig = lum2cgs(self.logL, self.DL)
        flux_cgs = flin(self.beta, flux_cgs_orig)
        cond_bad = flux_cgs < flux_cgs_orig
        flux_cgs[cond_bad] = flux_cgs_orig[cond_bad]
        mags = cgs2magAB(flux_cgs, self.wav_filt, self.filt_width)
        for i, dist in enumerate(self.dist):
            if i%400==0: print(f"Got to i={i} for calculating comps grid")
            compgrid[i] = self.interp_comp_simp.ev(dist, mags)
            # compgrid[i] = comps.reshape(*self.logL_trans_integ.shape)
        # compG = compgrid * self.trans_conv[None,None]

        ldo = len(self.dist)
        likes = np.zeros((alnum, lsnum))
        for i in range(alnum):
            print(f"Got to i={i} in main al ls loop")
            for j in range(lsnum):
                # time1 = time()
                tlf = TrueLumFuncNoPhi(self.logL_trans_integ, als[i], lss[j])
                # integ = tlf[None] * compG
                # phiobs = trapezoid(integ, self.logL_trans_integ[None], axis=2)
                phimed = trapezoid(tlf*self.trans_conv[None], self.logL_trans_integ, axis=1)
                phiobs = compgrid * phimed
                phiobsnorm = phiobs / trapezoid(phiobs, self.logL, axis=1)[:,None]
                likeij = np.zeros(ldo)
                for k in range(ldo):
                    likeij[k] = np.interp(self.lum[k], self.logL, phiobsnorm[k])
                likes[i,j] = np.log(likeij).sum()
                # time2 = time()
                # print("Time taken for one iteration:", time2-time1)
                # if i%10==0: 
                #     truenorm = tlf[:,0] / trapezoid(tlf[:,0], self.logL)
                #     phimednorm = phimed / trapezoid(phimed, self.logL)
                #     phiobsuse = np.median(phiobsnorm, axis=0)
                #     self.plotPracLumFunc(truenorm, phimednorm, phiobsuse, als[i], lss[j], likes[i,j])
        return als, lss, likes

    def calclikeLsalShela(self, p=None, alnum=50, lsnum=50):
        ''' Get log likelihood values for alpha and L* for the non-circular (SHELA) completeness model.

        Parameters
        ----------
        p : array-like of bool/int
            Region membership per source. False/0 uses completeness function 1;
            True/1 uses completeness function 2.
        '''
        self.normhist, bin_edges = np.histogram(self.lum, bins=self.nbins, density=True)
        self.Lmed = (bin_edges[:-1] + bin_edges[1:])/2.0
        als = np.linspace(self.sch_al_lims[0], self.sch_al_lims[1], alnum)
        lss = np.linspace(self.Lstar_lims[0], self.Lstar_lims[1], lsnum)

        if p is None:
            p = self.comp_region
        if p is None:
            p = np.zeros(self.lum.size, dtype=bool)
        else:
            p = np.asarray(p).astype(bool)
            if p.size != self.lum.size:
                raise ValueError("Length of region array p must match number of sources.")

        flux_cgs_orig = lum2cgs(self.logL, self.DL)
        flux_cgs = flin(self.beta, flux_cgs_orig)
        cond_bad = flux_cgs < flux_cgs_orig
        flux_cgs[cond_bad] = flux_cgs_orig[cond_bad]
        mags = cgs2magAB(flux_cgs, self.wav_filt, self.filt_width)

        # Region-1 completeness curve (required)
        if self.comp1df is None:
            raise ValueError("self.comp1df is not set. Build region-1 completeness before calling calclikeLsalShela.")
        compgrid1 = self.comp1df(mags)

        # Region-2 completeness curve (fallback to region-1 if not supplied)
        if hasattr(self, 'comp2df') and self.comp2df is not None:
            compgrid2 = self.comp2df(mags)
        else:
            compgrid2 = compgrid1

        compgrid = np.where(p[:,None], compgrid2[None,:], compgrid1[None,:])

        ldo = self.lum.size
        likes = np.zeros((alnum, lsnum))
        for i in range(alnum):
            print(f"Got to i={i} in main al ls loop")
            for j in range(lsnum):
                tlf = TrueLumFuncNoPhi(self.logL_trans_integ, als[i], lss[j])
                phimed = trapezoid(tlf*self.trans_conv[None], self.logL_trans_integ, axis=1)
                phiobs = compgrid * phimed
                phiobsnorm = phiobs / trapezoid(phiobs, self.logL, axis=1)[:,None]
                likeij = np.zeros(ldo)
                for k in range(ldo):
                    likeij[k] = np.interp(self.lum[k], self.logL, phiobsnorm[k])
                likes[i,j] = np.log(likeij).sum()
        return als, lss, likes

    def calclikeLsalTH(self, alnum=50, lsnum=50):
        ''' Calculate alpha, L* likelihood with a perfect top-hat filter '''
        self.normhist, bin_edges = np.histogram(self.lum, bins=self.nbins, density=True)
        self.Lmed = (bin_edges[:-1] + bin_edges[1:])/2.0
        als = np.linspace(self.sch_al_lims[0], self.sch_al_lims[1], alnum)
        lss = np.linspace(self.Lstar_lims[0], self.Lstar_lims[1], lsnum)
        # compgrid = np.zeros((len(self.dist), *self.logL_trans_integ.shape))
        compgrid = np.zeros((self.dist.size, self.logL.size))
        # L_all = 10**self.logL_trans_integ.ravel()
        flux_cgs_orig = lum2cgs(self.logL, self.DL)
        flux_cgs = flin(self.beta, flux_cgs_orig)
        cond_bad = flux_cgs < flux_cgs_orig
        flux_cgs[cond_bad] = flux_cgs_orig[cond_bad]
        mags = cgs2magAB(flux_cgs, self.wav_filt, self.filt_width)
        for i, dist in enumerate(self.dist):
            if i%400==0: print(f"Got to i={i} for calculating comps grid")
            compgrid[i] = self.interp_comp_simp.ev(dist, mags)

        likes = np.zeros((alnum, lsnum))
        for i in range(alnum):
            print(f"Got to i={i} in main al ls loop")
            for j in range(lsnum):
                # tic = time()
                tlf = TrueLumFuncNoPhi(self.lum, als[i], lss[j])
                tlfll = TrueLumFuncNoPhi(self.logL, als[i], lss[j])
                phiobs = self.comps * tlf
                fornorm = compgrid * tlfll
                phiobsnorm = phiobs / trapezoid(fornorm, self.logL, axis=1)
                likes[i,j] = np.log(phiobsnorm).sum()
                # toc = time()
                # print("Time per iteration: ", toc-tic)
                # breakpoint()
        return als, lss, likes
    
    def calcVgalPhistar(self, alnum=50, lsnum=50, rnum=100, exceed=1.5):
        ''' Calculate number of observed galaxies predicted by Schechter parameters if phi* = 1 (log phi* = 0)'''
        if self.frac2 is not None or (hasattr(self, 'comp2df') and self.comp2df is not None):
            return self.calcVgalPhistarShela(alnum=alnum, lsnum=lsnum, rnum=rnum, exceed=exceed)
        fac_sr_to_arcmin = np.pi / 180. / 60.
        integ_mult = 2 * np.pi * fac_sr_to_arcmin**2
        # self.getminlum_z_func()
        # self.getmaxlum_z_func()
        als = np.linspace(self.sch_al_lims[0], self.sch_al_lims[1], alnum)
        lss = np.linspace(self.Lstar_lims[0], self.Lstar_lims[1], lsnum)
        R = np.sqrt(self.Omega_0_sr/np.pi) # Angular radius of circular field in radians
        rs = np.linspace(0, R, rnum) / fac_sr_to_arcmin # Get radial position in arcmin
        vgal = np.zeros((alnum, lsnum))
        # logLr = np.zeros((rnum, self.size_ln))
        mlh = cgs2lum(self.flux_lim, self.DL)
        # ml = self.minlum2df.ev(self.z, rs)
        ml = self.minlumf(rs)
        
        for j in range(lsnum):
            print(f"Got to j={j} in main al ls loop")
            logLr = np.zeros((rnum, self.size_ln))
            for kk in range(rnum):
            #     ml = self.minlumf(rs[kk])
            #     ml = self.minlum2df.ev(self.zarr, rs[kk])
                
                logLr[kk] = np.linspace(ml[kk], max(ml[kk], min(mlh, lss[j] + exceed)), num=self.size_ln)
            flux_cgs_orig = lum2cgs(logLr, self.DL)
            flux_cgs = flin(self.beta, flux_cgs_orig)
            cond_bad = flux_cgs < flux_cgs_orig
            flux_cgs[cond_bad] = flux_cgs_orig[cond_bad]
            fcn = self.trans_vals[:,None,None] * flux_cgs[None]
            mags = cgs2magAB(fcn, self.wav_filt, self.filt_width)
            comps = self.interp_comp_simp.ev(rs[None,:,None], mags)
            # comps[comps<self.min_comp_frac] = 0.0
            
            for i in range(alnum):
                # time1 = time()
                tlf = TrueLumFuncNoPhi(logLr, als[i], lss[j])
                integ = self.dVdzs[:,None,None] * comps * rs[None,:,None] * tlf[None]
                vgal[i,j] = integ_mult * trapezoid(trapezoid(trapezoid(integ, logLr[None], axis=2), rs), self.zarr)
                # time2 = time()
                # print(f"Time to go through one vgal calculation: {time2-time1}")
                # breakpoint()
        return als, lss, vgal

    def calcVgalPhistarTH(self, alnum=50, lsnum=50, rnum=100, exceed=1.5):
        ''' Calculate number of observed galaxies given Schechter parameters (with phi* = 1) if we had a perfect top-hat filter'''
        fac_sr_to_arcmin = np.pi / 180. / 60.
        integ_mult = 2 * np.pi * fac_sr_to_arcmin**2 * self.volume
        als = np.linspace(self.sch_al_lims[0], self.sch_al_lims[1], alnum)
        lss = np.linspace(self.Lstar_lims[0], self.Lstar_lims[1], lsnum)
        R = np.sqrt(self.Omega_0_sr/np.pi) # Angular radius of circular field in radians
        rs = np.linspace(0, R, rnum) / fac_sr_to_arcmin # Get radial position in arcmin
        vgal = np.zeros((alnum, lsnum))
        # logLr = np.zeros((rnum, self.size_ln))
        mlh = cgs2lum(self.flux_lim, self.DL)
        # ml = self.minlum2df.ev(self.z, rs)
        ml = self.minlumf(rs)
        
        for j in range(lsnum):
            print(f"Got to j={j} in main al ls loop")
            logLr = np.zeros((rnum, self.size_ln))
            for kk in range(rnum):
            #     ml = self.minlumf(rs[kk])
            #     ml = self.minlum2df.ev(self.zarr, rs[kk])
                
                logLr[kk] = np.linspace(ml[kk], max(ml[kk], min(mlh, lss[j] + exceed)), num=self.size_ln)
            flux_cgs_orig = lum2cgs(logLr, self.DL)
            flux_cgs = flin(self.beta, flux_cgs_orig)
            cond_bad = flux_cgs < flux_cgs_orig
            flux_cgs[cond_bad] = flux_cgs_orig[cond_bad]
            mags = cgs2magAB(flux_cgs, self.wav_filt, self.filt_width)
            comps = self.interp_comp_simp.ev(rs[:,None], mags)
            # comps[comps<self.min_comp_frac] = 0.0
            
            for i in range(alnum):
                # time1 = time()
                tlf = TrueLumFuncNoPhi(logLr, als[i], lss[j])
                integ = comps * rs[:,None] * tlf
                vgal[i,j] = integ_mult * trapezoid(trapezoid(integ, logLr, axis=1), rs)
                # time2 = time()
                # print(f"Time to go through one vgal calculation: {time2-time1}")
                # breakpoint()
        return als, lss, vgal

    def calcVgalPhistarShela(self, alnum=50, lsnum=50, rnum=100, exceed=1.5):
        ''' Calculate expected observed counts (phi*=1) for non-circular SHELA geometry.

        The angular integral is replaced by area-fraction weighting:
            int dOmega -> Omega_0_sr * (frac1 * C1 + frac2 * C2)
        where C1 and C2 are completeness models for the two regions.
        '''
        als = np.linspace(self.sch_al_lims[0], self.sch_al_lims[1], alnum)
        lss = np.linspace(self.Lstar_lims[0], self.Lstar_lims[1], lsnum)
        vgal = np.zeros((alnum, lsnum))

        if self.comp1df is None:
            raise ValueError("self.comp1df is not set. Build region-1 completeness before calling calcVgalPhistarShela.")
        compf1 = self.comp1df
        compf2 = self.comp2df if hasattr(self, 'comp2df') and self.comp2df is not None else compf1

        frac1 = 1.0 if self.frac1 is None else float(self.frac1)
        frac2 = 0.0 if self.frac2 is None else float(self.frac2)
        frac_norm = frac1 + frac2
        if frac_norm <= 0:
            frac1, frac2 = 1.0, 0.0
        else:
            frac1, frac2 = frac1/frac_norm, frac2/frac_norm

        mlh = cgs2lum(self.flux_lim, self.DL)
        ml1 = self.minlum if self.minlum is not None else self.Lc
        ml2 = self.minlum2 if self.minlum2 is not None else ml1

        for j in range(lsnum):
            print(f"Got to j={j} in main al ls loop")
            logL1 = np.linspace(ml1, max(ml1, min(mlh, lss[j] + exceed)), num=self.size_ln)
            logL2 = np.linspace(ml2, max(ml2, min(mlh, lss[j] + exceed)), num=self.size_ln)

            flux1_orig = lum2cgs(logL1, self.DL)
            flux1 = flin(self.beta, flux1_orig)
            cond_bad1 = flux1 < flux1_orig
            flux1[cond_bad1] = flux1_orig[cond_bad1]

            flux2_orig = lum2cgs(logL2, self.DL)
            flux2 = flin(self.beta, flux2_orig)
            cond_bad2 = flux2 < flux2_orig
            flux2[cond_bad2] = flux2_orig[cond_bad2]

            mags1 = cgs2magAB(self.trans_vals[:,None] * flux1[None], self.wav_filt, self.filt_width)
            mags2 = cgs2magAB(self.trans_vals[:,None] * flux2[None], self.wav_filt, self.filt_width)
            comps1 = compf1(mags1)
            comps2 = compf2(mags2)

            for i in range(alnum):
                tlf1 = TrueLumFuncNoPhi(logL1, als[i], lss[j])
                tlf2 = TrueLumFuncNoPhi(logL2, als[i], lss[j])
                integ1 = self.dVdzs[:,None] * comps1 * tlf1[None]
                integ2 = self.dVdzs[:,None] * comps2 * tlf2[None]
                v1 = trapezoid(trapezoid(integ1, logL1, axis=1), self.zarr)
                v2 = trapezoid(trapezoid(integ2, logL2, axis=1), self.zarr)
                vgal[i,j] = self.Omega_0_sr * (frac1 * v1 + frac2 * v2)
        return als, lss, vgal

    def setup_logging(self):
        '''Setup Logging for MCSED

        Builds
        -------
        self.log : class
            self.log.info() is for general print and self.log.error() is
            for raise cases
        '''
        self.log = logging.getLogger('lumfuncmcmc')
        if not len(self.log.handlers):
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
            self.log = logging.getLogger('lumfuncmcmc')
            self.log.setLevel(logging.DEBUG)
            self.log.addHandler(handler)

    def set_parameters_from_list(self,input_list):
        ''' For a given set of model parameters, set the needed class variables.

        Input
        -----
        theta : list
            list of input parameters for Schechter Fit'''
        self.Lstar = input_list[0]
        self.phistar = input_list[1]
        if self.fix_sch_al: pass
        else: self.sch_al = input_list[2]

    def lnprior(self):
        ''' Simple, uniform prior for input variables

        Returns
        -------
        0.0 if all parameters are in bounds, -np.inf if any are out of bounds
        '''
        flag = 1.0
        for param in self.all_param_names:
            flag *= ((getattr(self,param) >= getattr(self,param+'_lims')[0]) *
                     (getattr(self,param) <= getattr(self,param+'_lims')[1]))
        if not flag: 
            return -np.inf
        else: 
            return 0.0

    def lnlike(self):
        ''' Calculate the log likelihood and return the value and stellar mass
        of the model as well as other derived parameters (an old version -- need care to see if it works properly)

        Returns
        -------
        log likelihood (float)
            The log likelihood includes a ln term and an integral term (based on Poisson statistics). '''
        lnpart = np.log(TrueLumFunc(self.lum,self.sch_al,self.Lstar,self.phistar)*self.comps).sum()
        integ = TrueLumFunc(self.logL,self.sch_al,self.Lstar,self.phistar) * self.Omega_gen
        fullint = self.volume * trapezoid(integ,self.logL)
        return lnpart - fullint
    
    def lnlike_conv(self):
        ''' Likelihood with convolution of both measurement errors and transmission effects; a bit out of date: use with care '''
        tlf = 10**self.phistar * TrueLumFuncNoPhi(self.logL_conv,self.sch_al,self.Lstar)
        not_norm = tlf*self.comps_conv*self.trans_conv
        trapezoid_inner = trapezoid(not_norm,self.logL_conv)
        numer = trapezoid(trapezoid_inner*self.norm_vals_norm, self.logL_norm)
        # denom = trapezoid(trapezoid_inner, self.logL_conv)
        lnpart = np.log(numer).sum()
        # fullint = self.Omega_0_sr * self.volume * denom
        integ = 10**self.phistar * TrueLumFuncNoPhi(self.logL_trans_integ,self.sch_al,self.Lstar) * self.not_tlf
        fullint = self.Omega_0_sr * self.dVdz * trapezoid(trapezoid(integ,self.logL_trans_integ),self.logL)
        return lnpart - fullint

    def lnlike_trans(self):
        ''' Likelihood with transmission effects; a bit out of date; to use with care '''
        tlf = 10**self.phistar * TrueLumFuncNoPhi(self.logL_trans_lnpart,self.sch_al,self.Lstar)
        lnpart = np.log(trapezoid(tlf*self.comps_trans_lnpart*self.trans_conv,self.logL_trans_lnpart)).sum()
        integ = 10**self.phistar * TrueLumFuncNoPhi(self.logL_trans_integ,self.sch_al,self.Lstar) * self.not_tlf
        fullint = self.Omega_0_sr * self.dVdz * trapezoid(trapezoid(integ,self.logL_trans_integ),self.logL)
        lnold = lnpart - fullint
        return lnold
    
    def lnlike_trans_v2(self):
        ''' Version of ln likelihood used for our analysis '''
        like_alls = self.likeallsf.ev(self.sch_al, self.Lstar)
        vgals = self.vgalf.ev(self.sch_al, self.Lstar)
        num = 10**self.phistar * self.frac_use * vgals * self.weight
        like_phi = poisson_lnpmf(self.N, int(num))
        return like_alls + like_phi

    def lnlike_norm(self):
        ''' Likelihood with just measurement errors included; use with care '''
        tlf = 10**self.phistar * TrueLumFuncNoPhi(self.logL_norm,self.sch_al,self.Lstar)
        lnpart = np.log(trapezoid(tlf*self.comps_norm*self.norm_vals_norm,self.logL_norm)).sum()
        integ = 10**self.phistar * TrueLumFuncNoPhi(self.logL,self.sch_al,self.Lstar) * self.Omega_gen
        fullint = self.volume * trapezoid(integ,self.logL)
        return lnpart - fullint

    def lnprob(self, theta):
        ''' Calculate the log probability (old version)

        Returns
        -------
        log prior + log likelihood, (float)
            The log probability is just the sum of the logs of the prior and likelihood. '''
        self.set_parameters_from_list(theta)
        lp = self.lnprior()
        if np.isfinite(lp):
            lnl = self.lnlike()
            return lnl+lp
        else:
            return -np.inf
        
    def lnprob_conv(self, theta):
        ''' lnprob in case of full convolution (errors and transmission): use with care '''
        self.set_parameters_from_list(theta)
        lp = self.lnprior()
        if np.isfinite(lp):
            lnl = self.lnlike_conv()
            return lnl+lp
        else:
            return -np.inf
        
    def lnprob_trans(self, theta):
        ''' lnprob used in our work '''
        self.set_parameters_from_list(theta)
        lp = self.lnprior()
        if np.isfinite(lp):
            lnl = self.lnlike_trans_v2()
            return lnl+lp
        else:
            return -np.inf
        
    def lnprob_norm(self, theta):
        ''' lnprob in case of just measurement errors and no transmission effects; use with care '''
        self.set_parameters_from_list(theta)
        lp = self.lnprior()
        if np.isfinite(lp):
            lnl = self.lnlike_norm()
            return lnl+lp
        else:
            return -np.inf

    def get_init_walker_values(self, num=None):
        ''' Before running emcee, this function generates starting points
        for each walker in the MCMC process.

        Returns
        -------
        pos : np.array (2 dim)
            Two dimensional array with Nwalker x Ndim values
        '''
        # theta = [self.sch_al, self.Lstar, self.phistar]
        theta_lims = np.vstack((self.Lstar_lims,self.phistar_lims))
        if not self.fix_sch_al: theta_lims = np.vstack((theta_lims,self.sch_al_lims))
        if num is None:
            num = self.nwalkers
        if self.diff_rand: pos_part1 = np.random.rand(num,len(theta_lims))
        else: pos_part1 = np.random.rand(num)[:,np.newaxis]
        pos = (pos_part1 * (theta_lims[:, 1]-theta_lims[:, 0]) + theta_lims[:, 0])
        return pos

    def get_param_names(self):
        ''' Grab the names of the parameters for plotting

        Returns
        -------
        names : list
            list of all parameter names
        '''
        names = [r'$\log L_*$',r'$\log \phi_*$']
        if not self.fix_sch_al: names += [r'$\alpha$']
        return names

    def get_params(self):
        ''' Grab the the parameters in each class

        Returns
        -------
        vals : list
            list of all parameter values
        '''
        vals = [self.Lstar,self.phistar]
        if not self.fix_sch_al: vals += [self.sch_al]
        self.nfreeparams = len(vals)
        return vals
    
    def fit_model(self):
        ''' Using emcee to find parameter estimations for given set of
        data measurements and errors
        '''
        self.log.info('Fitting Schechter model to true luminosity function using emcee')
        pos = self.get_init_walker_values()
        ndim = pos.shape[1]
        if self.err_corr: func = 'lnprob_conv'
        else: func = 'lnprob'
        if self.trans_only: func = 'lnprob_trans'
        if self.norm_only: func = 'lnprob_norm'
        sampler = emcee.EnsembleSampler(self.nwalkers, ndim, getattr(self,func))
        # Do real run
        start = time()
        sampler.run_mcmc(pos, self.nsteps, rstate0=np.random.get_state())
        end = time()
        elapsed = end - start
        self.log.info("Total time taken: %0.2f s" % elapsed)
        self.log.info("Time taken per step per walker: %0.2f ms" %
                        (elapsed / (self.nsteps) * 1000. /
                       self.nwalkers))
        # Calculate how long the run should last
        tau = np.max(sampler.acor)
        burnin_step = int(tau*3)
        if burnin_step>self.nsteps//2: burnin_step = self.nsteps//2
        self.log.info("Mean acceptance fraction: %0.2f" %
                      (np.mean(sampler.acceptance_fraction)))
        self.log.info("AutoCorrelation Steps: %i, Number of Burn-in Steps: %i"
                      % (np.round(tau), burnin_step))
        new_chain = np.zeros((self.nwalkers, self.nsteps, ndim+1))
        new_chain[:, :, :-1] = sampler.chain
        self.chain = sampler.chain
        new_chain[:, :, -1] = sampler.lnprobability
        self.samples = new_chain[:, burnin_step:, :].reshape((-1, ndim+1))
        self.log.info("Shape of self.samples")
        self.log.info(self.samples.shape)
        self.log.info("Median lnprob: %.5f; Max lnprob: %.5f"%(np.median(sampler.lnprobability), np.amax(sampler.lnprobability)))

    def VeffLF(self, varying=False, combo=False, phifunc=None, lum=None):
        ''' Use V_Eff method to calculate properly weighted measured luminosity function '''
        print("Ready to calculate V effective method")
        if phifunc is not None: self.phifunc, self.lum = phifunc, lum
        else:
            if varying: self.phifunc = 1.0/(self.dVdz * self.delzf(self.lum - self.minlum) * self.Omega_arr * self.frac_use)
            else: self.phifunc = 1.0/(self.volume * self.Omega_arr * self.frac_use)
        if combo: return
        self.Lavg, self.lfbinorig, self.var = V.getBootErrLog(self.lum,self.phifunc,self.nboot,self.nbins,Lmin=self.minlum, Lmax=self.maxlum)
        if self.corrf is not None:
            ucorr_orig = unumpy.uarray(self.corrf(self.Lavg), self.corref(self.Lavg))
            ulf = unumpy.uarray(self.lfbinorig, np.sqrt(self.var))
            cond = self.lfbinorig>0
            ulf_new = unumpy.uarray(np.zeros_like(self.lfbinorig), np.zeros_like(self.lfbinorig))
            ulf_new[cond] = 10 ** (unumpy.log10(ulf[cond]) + ucorr_orig[cond])
            self.lfbinorig_orig, self.var_orig = self.lfbinorig*1.0, self.var*1.0 #Want to show original values
            self.lfbinorig = unumpy.nominal_values(ulf_new)
            self.var = unumpy.std_devs(ulf_new) ** 2
            self.var[~cond] = self.var_orig[~cond] + self.corref(self.Lavg[~cond])**2

    def set_median_fit(self,rndsamples=200,lnprobcut=7.5):
        '''
        set attributes
        median modeled ("observed") luminosity function for rndsamples random samples
        This function is applied only when a triangle plot is not desired

        Input
        -----
        rndsamples : int
            number of random samples over which to compute medians
        lnprobcut : float
            Some of the emcee chains include outliers.  This value serves as
            a cut in log probability space with respect to the maximum
            probability.  For reference, a Gaussian 1-sigma is 2.5 in log prob
            space.

        Creates
        -------
        self.medianLF : list (1d)
            median fitted ("observed") luminosity function
        '''
        nsamples = []
        while len(nsamples)<len(self.samples)//4: 
            chi2sel = (self.samples[:, -1] >
                    (np.max(self.samples[:, -1], axis=0) - lnprobcut))
            nsamples = self.samples[chi2sel, :]
            lnprobcut *= 2.0
        # nsamples = self.samples
        self.log.info("Shape of nsamples (with a lnprobcut applied)")
        self.log.info(nsamples.shape)
        lf = []
        for i in np.arange(rndsamples):
            ind = np.random.randint(0, nsamples.shape[0])
            self.set_parameters_from_list(nsamples[ind, :])
            modlum = TrueLumFunc(self.lum,self.sch_al,self.Lstar,self.phistar)
            lf.append(modlum)
        self.medianLF = np.median(np.array(lf), axis=0)
        self.VeffLF(varying=self.varying)

    def plotLike(self, lss, als, likes, vgal, nameext='', levels=15):
        ''' Plot alpha, L* likelihoods from the grid and the number o galaxies given alpha, L*, with phi* = 1'''
        fig1, ax1 = plt.subplots()
        sc = ax1.contourf(lss, als, likes, levels=levels)
        ax1.set_xlabel(r'$\mathcal{L}_*$')
        ax1.set_ylabel(r'$\alpha$')
        fig1.colorbar(sc, label='Log likelihood')
        file_name = f'AllsLike{nameext}.png'
        fig1.savefig(file_name, bbox_inches='tight', dpi=300)
        fig2, ax2 = plt.subplots()
        sc = ax2.contourf(lss, als, np.log10(vgal), levels=levels)
        ax2.set_xlabel(r'$\mathcal{L}_*$')
        ax2.set_ylabel(r'$\alpha$')
        fig2.colorbar(sc, label='Log # Obs Galaxies')
        fig2.savefig(f'AllsVgal{nameext}.png', bbox_inches='tight', dpi=300)
        plt.close('all')

    def plotPracLumFunc(self, tlft, phimed, phiobs, al, ls, likesij):
        ''' For testing purposes '''
        fig, ax = plt.subplots()
        self.add_LumFunc_plot(ax)
        ax.plot(self.logL, tlft, 'b-', label='Norm True LF')
        ax.plot(self.logL, phimed, 'k-', label='Norm TC LF')
        ax.plot(self.logL, phiobs, 'r-', label='Norm Obs LF')
        condhist = self.normhist>0
        ax.scatter(self.Lmed[condhist], self.normhist[condhist], c='k', s=8, label='Norm Lum Hist')
        ax.text(0, 0, f'Alpha: {al:0.2f}; Lstar: {ls:0.2f}; Ln Like {likesij:0.0f}', transform=ax.transAxes)
        ax.legend(loc='best', frameon=False)
        # miny = 1.0e-8
        # ax.set_ylim(miny, max(tlft.max(), phiobs.max()))
        xmin, xmax = self.Lmed.min()-0.2, self.Lmed.max()+0.2
        ax.set_xlim(xmin, xmax)
        cond = np.logical_and(self.logL>=xmin, self.logL<=xmax)
        ymin = min(tlft[cond].min(), phimed[cond].min(), phiobs[cond].min(), self.normhist[condhist].min())
        ymax = max(tlft[cond].max(), phimed[cond].max(), phiobs[cond].max(), self.normhist[condhist].max())
        ax.set_ylim(ymin, ymax)
        # cond = np.logical_or(tlft>ymin, phiobs>miny)
        # ax.set_xlim(self.logL.min(), self.logL[cond].max())
        plt.show()

    def add_LumFunc_plot(self,ax1):
        """ Set up the plot for the luminosity function """
        ax1.set_yscale('log')
        ax1.set_xlabel(r"$\log$ L (erg s$^{-1}$)")
        ax1.set_ylabel(r"$\phi_{\rm{true}}$ (Mpc$^{-3}$ dex$^{-1}$)")
        ax1.minorticks_on()

    def VeffPlotCommands(self, ax):
        ''' Part of V/V_max method plotting '''
        markersize = self.nfreeparams * 1
        cond_veff = np.logical_and(self.Lavg >= self.minlum, self.lfbinorig>1.0e-12)
        if self.corrf is not None: label=r'$V_{\rm eff}$ + Filter'
        else: label=r'$V_{\rm eff}$'
        ax.errorbar(self.Lavg[cond_veff],self.lfbinorig[cond_veff],yerr=np.sqrt(self.var[cond_veff]),fmt='b^', label=label, markersize=markersize)
        # ax.errorbar(self.Lavg[~cond_veff],self.lfbinorig[~cond_veff],yerr=np.sqrt(self.var[~cond_veff]),fmt='b^',alpha=0.2, label='', markersize=markersize)
        if self.corrf is not None:
            ax.errorbar(self.Lavg[cond_veff],self.lfbinorig_orig[cond_veff],yerr=np.sqrt(self.var_orig[cond_veff]),fmt='cs', label=r'$V_{\rm eff}$', markersize=markersize)
            # ax.errorbar(self.Lavg[~cond_veff],self.lfbinorig_orig[~cond_veff],yerr=np.sqrt(self.var_orig[~cond_veff]),fmt='cs',alpha=0.2, label='', markersize=markersize)
        leg = ax.legend(loc='best', frameon=False, fontsize='x-small')
        for lh in leg.legend_handles:
            lh.set_alpha(1)

    def plotVeff(self, outname, imgtype='png', varying=False):
        ''' Plot V/V_max method results'''
        self.VeffLF(varying=varying)
        fig, ax = plt.subplots()
        self.add_LumFunc_plot(ax)
        self.VeffPlotCommands(ax)
        fig.savefig(outname+'.'+imgtype, bbox_inches='tight', dpi=300)

    def plotVeffEnv(self, Lavgs, lfbinorigs, vars, minlums, labels, outname, imgtype='png', fmt_seq=['b^', 'r*', 'ko', 'mx', 'cs', 'gh', 'y+'], lflums=None, lfs=None, linestyle_seq=['-', '--', '-.', ':', '-', '--', '-.', ':']):
        ''' Plot V/V_max method in multiple environments'''
        fig, ax = plt.subplots()
        self.add_LumFunc_plot(ax)
        ilist = np.arange(len(Lavgs))
        for i, Lavg, lfbinorig, var, minlum in zip(ilist, Lavgs, lfbinorigs, vars, minlums):
            col = fmt_seq[i][0]
            cond_veff = Lavg >= minlum
            ax.plot(Lavg[cond_veff],lfbinorig[cond_veff],fmt_seq[i],linestyle='none',label=labels[i])
            ax.errorbar(Lavg[cond_veff],lfbinorig[cond_veff],yerr=np.sqrt(var[cond_veff]),fmt='none',ecolor=col,label='',alpha=0.1)
            ax.errorbar(Lavg[~cond_veff],lfbinorig[~cond_veff],yerr=np.sqrt(var[~cond_veff]),fmt=fmt_seq[i],alpha=0.1,label='')
            if lfs is not None: 
                lfli, lfi = lflums[i], lfs[i]
                indsort = np.argsort(lfli)
                ax.plot(lfli[indsort], lfi[indsort], col+linestyle_seq[i], label='')
        ax.legend(loc='best', frameon=False, fontsize='small')
        fig.savefig(outname+'.'+imgtype, bbox_inches='tight', dpi=300)

    def add_subplots(self,ax1,nsamples,rndsamples=200):
        ''' Add Subplots to Triangle plot below '''
        lf = []
        indsort = np.argsort(self.lum)
        lstars = np.zeros(rndsamples)
        for i in np.arange(rndsamples):
            if i==0: labeli = 'MCMC solutions'
            else: labeli = ''
            ind = np.random.randint(0, nsamples.shape[0])
            self.set_parameters_from_list(nsamples[ind, :])
            lstars[i] = self.Lstar
            modlum = TrueLumFunc(self.lum,self.sch_al,self.Lstar,self.phistar)
            lf.append(modlum)
            ax1.plot(self.lum[indsort],modlum[indsort],color='r',linestyle='solid',alpha=0.1, label=labeli)
        self.medianLF = np.median(np.array(lf), axis=0)
        self.VeffLF(varying=self.varying)
        # label = 'MCMC Best-fit'
        ax1.plot(self.lum[indsort],self.medianLF[indsort],color='dimgray',linestyle='solid',label='')
        self.VeffPlotCommands(ax1)
        xmin = self.minlum
        xmax = min(max(self.lum),np.median(lstars)+1.0)
        ax1.set_xlim(left=xmin,right=xmax)
        cond = np.logical_and(self.lum<=xmax,self.lum>=xmin)
        ax1.set_ylim(bottom=np.percentile(self.medianLF[cond],0),top=np.percentile(self.medianLF[cond],100))
        
    def triangle_plot(self, outname, lnprobcut=7.5, imgtype='png'):
        ''' Make a triangle corner plot for samples from fit

        Input
        -----
        outname : string
            The triangle plot will be saved as "triangle_{outname}.png"
        lnprobcut : float
            Some of the emcee chains include outliers.  This value serves as
            a cut in log probability space with respect to the maximum
            probability.  For reference, a Gaussian 1-sigma is 2.5 in log prob
            space.
        imgtype : string
            The file extension of the output plot
        '''
        # Make selection for three sigma sample
        nsamples = []
        while len(nsamples)<len(self.samples)//4: 
            chi2sel = (self.samples[:, -1] >
                    (np.max(self.samples[:, -1], axis=0) - lnprobcut))
            nsamples = self.samples[chi2sel, :]
            lnprobcut *= 2.0
        # nsamples = self.samples
        self.log.info("Shape of nsamples (with a lnprobcut applied)")
        self.log.info(nsamples.shape)
        names = self.get_param_names()
        indarr = np.arange(len(nsamples[0]))
        fsgrad = 11+int(round(0.75*len(indarr)))
        percentilerange = [.95] * len(names)
        fig = corner.corner(nsamples[:, :-1], labels=names,
                            range=percentilerange,
                            label_kwargs={"fontsize": fsgrad}, show_titles=True,
                            title_kwargs={"fontsize": fsgrad-2},
                            quantiles=[0.16, 0.5, 0.84], bins=30)
        w = fig.get_figwidth()
        if len(indarr)>=4: 
            figw = w-(len(indarr)-13)*0.025*w
            poss = [0.50-0.008*(len(indarr)-4), 0.78-0.001*(len(indarr)-4), 0.48+0.008*(len(indarr)-4), 0.19+0.001*(len(indarr)-4)]
        else: 
            figw = w
            poss = [0.67,0.75,0.32,0.23]
        fig.set_figwidth(figw)
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.set_position(poss)
        self.add_LumFunc_plot(ax1)
        self.add_subplots(ax1,nsamples)
        fig.savefig("%s.%s" % (outname,imgtype), dpi=200)
        plt.close(fig)

    def add_fitinfo_to_table(self, percentiles, start_value=1, lnprobcut=7.5):
        ''' Put the Schechter parameter basic fitting results into a table. This assumes that "Ln Prob" is the last column in self.samples'''
        nsamples = []
        while len(nsamples)<len(self.samples)//4: 
            chi2sel = (self.samples[:, -1] >
                    (np.max(self.samples[:, -1], axis=0) - lnprobcut))
            nsamples = self.samples[chi2sel, :-1]
            lnprobcut *= 2.0
        # nsamples = self.samples[:,:-1]
        self.log.info("Number of table entries: %d"%(len(self.table[0])))
        self.log.info("Len(percentiles): %d; len(other axis): %d"%(len(percentiles), len(np.percentile(nsamples,percentiles[0],axis=0))))
        n = len(percentiles)
        for i, per in enumerate(percentiles):
            for j, v in enumerate(np.percentile(nsamples, per, axis=0)):
                self.table[-1][(i + start_value + j*n)] = v
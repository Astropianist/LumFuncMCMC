import numpy as np
import os.path as op
from astropy.table import Table

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

def calcChiSq(logL, veff, veffe, alpha, logLstar, logphistar, numpar=2):
    lumfcalc = TrueLumFunc(logL, alpha, logLstar, logphistar)
    dof = len(logL) - numpar
    rcs = (veff-lumfcalc)**2/veffe**2
    return rcs.sum()/dof

def main(filter='N419'):
    dir_main = 'LFMCMCOdin'
    alphas = np.linspace(-2.0, -1.1, 10)
    al_used = []
    chisq = []
    veff_pre = f'fixed_sa_{filter}_VeffLF_ODIN_fsa1_sa'
    veff_str = 'mcf50_ll43.1_ec0_nb50_nw120_ns1500_mcf50_ec_0_env0_bin1_c0.dat'
    for i, al in enumerate(alphas):
        diri_str = f'ODIN_fsa1_sa{al:0.2f}_mcf50_ll43.1_ec0'
        diri = op.join(dir_main, diri_str)
        veff_file = op.join(diri, f'{veff_pre}{al:0.2f}_{veff_str}')
        lumf_file = op.join(diri, f'fixed_sa_{filter}_{diri_str}_env0_bin1.dat')
        try:
            lumf = Table.read(lumf_file, format='ascii')
            logLstar, logphistar = lumf[r'$\log L_*$_50'], lumf[r'$\log \phi_*$_50']
            veff = Table.read(veff_file, format='ascii')
            logL, v, ve = veff['Luminosity'], veff['BinLF'], veff['BinLFErr']
            chisq.append(calcChiSq(logL, v, ve, al, logLstar, logphistar))
            al_used.append(al)
        except: pass
    print(al_used, chisq)

if __name__=='__main__':
    main('N419')
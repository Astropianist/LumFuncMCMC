import numpy as np
from astropy.io import fits, ascii
from astropy.table import Table
from uncertainties import unumpy
from scipy.integrate import trapezoid
from astropy.coordinates import SkyCoord
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import CloughTocher2DInterpolator as CTI
from scipy.interpolate import NearestNDInterpolator as NNI

c = 3.0e18 # Speed of light in Angstroms/s
cosmos_center = SkyCoord('10h00m24s', '2d10m55s')
ccra, ccdec = 10*15 + 24*15/3600, 2 + 10/60 + 55/3600

class GriddataExt:
    def __init__( self, points, values):
        self.interp = CTI(points, values, fill_value=np.nan)
        self.nearest = NNI(points, values)
        
    def __call__( self, xi ):
        vals = self.interp( xi )
        idxs = np.isnan( vals )
        if type(xi)==tuple: vals[idxs] = self.nearest((xi[0][idxs], xi[1][idxs]))
        else: vals[idxs] = self.nearest( xi[idxs] )
        return vals

def getDist(ra1, dec1, ra2, dec2):
    dra = ((ra1 - ra2)
                * np.cos(np.pi/180.*dec2) * 60.)
    ddec = (dec1 - dec2)*60.
    d = np.sqrt(dra**2 + ddec**2)
    return d

def getTrans(fn='N501_with_atm.txt'):
    dat = ascii.read(fn)
    lam, trans = dat['lambda'], dat['transmission']
    del dat
    return lam, trans

def getContSubtFlux(aper_corr=0.23, fn='COSMOS_N501_LAEs_starmasked_08_23.fits', center=cosmos_center, zp=29.736, ABzp=-48.6):
    fac = 10**(0.4*aper_corr)
    fac_cont = 10**(0.4*zp)
    cosmos = fits.getdata(fn)
    # breakpoint()
    name = cosmos['number']
    ra, dec = cosmos['x_world'], cosmos['y_world']
    coords = SkyCoord(ra, dec, unit='degree')
    sep = coords.separation(center).arcmin
    fg, fr, f501 = cosmos['flux_aper_g'][:,3]*fac*fac_cont, cosmos['flux_aper_r'][:,3]*fac*fac_cont, cosmos['flux_aper'][:,3]*fac
    fge, fre, f501e = cosmos['fluxerr_aper_g'][:,3]*fac*fac_cont, cosmos['fluxerr_aper_r'][:,3]*fac*fac_cont, cosmos['fluxerr_aper'][:,3]*fac
    ufg, ufr, uf501 = unumpy.uarray(fg, fge), unumpy.uarray(fr, fre), unumpy.uarray(f501, f501e)
    ufHa = uf501 - (0.83*ufg + 0.17*ufr)
    Ha, Haerr = unumpy.nominal_values(ufHa), unumpy.std_devs(ufHa)
    fac_cgs = 10**(-0.4*(zp - ABzp))
    # breakpoint()
    return Ha*fac_cgs, Haerr*fac_cgs, sep, name, ra, dec, cosmos['lae_surface_density'].ravel(), cosmos['cell_area']

def getLineFlux(fn='LAE_catalog_COSMOS_gr-n501_SE_2024_03_06_expanded.csv', tfn='N501_Nicole.txt', wav_filt=5014.0, center=cosmos_center):
    dat = Table.read(fn, format='ascii')
    name, ra, dec = dat['index'], dat['RA'], dat['DEC']
    coords = SkyCoord(ra, dec, unit='degree')
    sep = coords.separation(center).arcmin
    col = fn.split('-')[0].split('_')[-1]
    filt = fn.split('-')[1].split('_')[0]
    filtf, filtfe = dat[f'{filt} flux (ujy)'], dat[f'{filt} flux err (ujy)']
    grf, grfe = dat[f'{col} flux (ujy)'], dat[f'{col} flux err (ujy)']
    ufiltf, ugrf = unumpy.uarray(filtf, filtfe), unumpy.uarray(grf, grfe)
    lam, trans = getTrans(tfn)
    Tc = trans.max()
    Tint = trapezoid(trans, lam)
    breakpoint()
    fac_flux = 1.0e-29 * c/wav_filt**2 * Tint/Tc * 1.0e17
    ulf = fac_flux * (ufiltf - ugrf)
    dlaef, pcf = getLAEDensity(band=filt.upper())
    try:
        dlaes = dlaef.ev(ra, dec)
    except:
        dlaes = dlaef(np.column_stack((ra, dec)))
    dlaes[dlaes<0] = 0.0
    pcs = pcf(np.column_stack((ra, dec)))
    return name, ra, dec, unumpy.nominal_values(ulf), unumpy.std_devs(ulf), dlaes, pcs, sep

def getLAEDensity(band='N501'):
    fn = f'COSMOS_{band}_sd_and_pcs.txt'
    dat = Table.read(fn, format='ascii')
    ra, dec, dlae, pc = dat['RA'], dat['DEC'], dat['delta_LAE']+1, dat['tag_pc']
    ra_use, dec_use = np.unique(ra), np.unique(dec)
    rl, dl = ra_use.size, dec_use.size
    try:
        dlae_use, pc_use = dlae.reshape(rl, dl), pc.reshape(rl, dl)
        dlaef = RBS(ra_use, dec_use, dlae_use)
        # pcf = RBS(ra_use, dec_use, pc_use, kx=0, ky=0)
        pcf = RGI((ra_use, dec_use), pc_use, method='nearest', bounds_error=False, fill_value=None)
    except:
        dlaef = GriddataExt(np.column_stack((ra, dec)), dlae)
        pcf = NNI(np.column_stack((ra, dec)), pc)
    return dlaef, pcf

def main_old(wav_filt=5014.0):
    lam, trans = getTrans('N501_Nicole.txt')
    Tc = trans.max()
    Tint = trapezoid(trans, lam)
    Ha, Hae, sep, name, ra, dec, surfden, cellarea = getContSubtFlux()
    dist = getDist(ra, dec, ccra, ccdec)
    fac_flux = c/wav_filt**2 * Tint/Tc * 1.0e17 # Want units of 1.0e-17 cgs
    Haf, Hafe = Ha * fac_flux, Hae * fac_flux
    breakpoint()
    dat = Table()
    dat['Galaxy_name'] = name
    dat['Lya_flux'] = Haf
    dat['Lya_flux_e'] = Hafe
    dat['dist'] = sep
    dat['distv2'] = dist
    dat['Surface_density'] = surfden
    dat['Cell_area'] = cellarea
    dat.write('LyaN501Fluxes.dat', format='ascii', overwrite=True)

def main(filter='N501'):
    if filter=='N501': col, wav = 'gr', 5014.0
    elif filter=='N419': col, wav ='rg', 4193.0
    else: col, wav = 'gi', 6750.0
    fn = f'LAE_catalog_COSMOS_{col}-{filter.lower()}_SE_2024_03_06_expanded.csv'
    tfn = f'{filter}_Nicole.txt'
    names, ras, decs, lyf, lyfe, dlaes, pcs, seps = getLineFlux(fn=fn, tfn=tfn, wav_filt=wav)
    dat = Table()
    dat['Galaxy_name'] = names
    dat['RA'] = ras
    dat['Dec'] = decs
    dat['Lya_flux'] = lyf
    dat['Lya_flux_e'] = lyfe
    dat['dist'] = seps
    dat['Density'] = dlaes
    dat['Protocluster'] = pcs
    dat.write(f'Lya{filter}FluxesFinal.dat', format='ascii', overwrite=True)

if __name__ == '__main__':
    main('N501')
import numpy as np
from astropy.io import fits, ascii
from astropy.table import Table
from uncertainties import unumpy
from scipy.integrate import trapezoid
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import astropy.units as u
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import CloughTocher2DInterpolator as CTI
from scipy.interpolate import NearestNDInterpolator as NNI
import os.path as op
import glob
import re
import matplotlib.pyplot as plt

c = 3.0e18 # Speed of light in Angstroms/s
cosmos_center = SkyCoord('10h00m24s', '2d10m55s')
xmmlss_center = SkyCoord(35.71, -4.75, unit=u.deg)
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

def make_voronoi_interpolators(fits_filename):
    '''
    Read a multi-extension Voronoi / protocluster FITS file and return
    interpolation functions for surface density and protocluster number
    as functions of RA and Dec.

    Parameters
    ----------
    fits_filename : str
        Path to FITS file of form
        "XMM_N???_voronoi_sd_maglim_25.4_01_2026.fits".

    Returns
    -------
    surfden_func : callable
        Function f(ra, dec) returning the surface density at the given
        sky coordinates.
    protocluster_func : callable
        Function g(ra, dec) returning the (nearest-neighbour) protocluster
        number at the given sky coordinates.
    area_zero_mask_deg2 : float
        Total area, in deg^2, of pixels in the mask (extension 2)
        with value 0.
    '''
    with fits.open(fits_filename) as hdul:
        # 1st extension: surface density map
        surfden = np.array(hdul[0].data, copy=True)
        # 2nd extension: protocluster map
        proto = np.array(hdul[1].data, copy=True)
        # 3rd extension: mask used to compute zero fraction
        mask = np.array(hdul[2].data, copy=True)
        # Use WCS from the surface density / protocluster extension (assumed same)
        wcs = WCS(hdul[1].header)

    # In the protocluster map, cap values at 1
    proto = proto.astype(float, copy=False)
    proto[proto > 1] = 1.0

    # Total area (deg^2) of pixels in the mask (extension 2) with value 0
    pix_scales_deg = proj_plane_pixel_scales(wcs)  # degrees per pixel
    area_per_pix_deg2 = pix_scales_deg[0] * pix_scales_deg[1]
    n_zero = np.count_nonzero(mask == 0)
    area_zero_mask_deg2 = n_zero * area_per_pix_deg2

    ny, nx = surfden.shape
    y = np.arange(ny, dtype=float)
    x = np.arange(nx, dtype=float)

    # Interpolator in pixel space for surface density
    surfden_spline_pix = RBS(y, x, surfden, kx=1, ky=1)

    # Interpolator in pixel space for protocluster map
    proto_interp_pix = RGI(
        (y, x),
        proto,
        method='nearest',
        bounds_error=False,
        fill_value=None
    )

    def surfden_func(ra, dec):
        '''Surface density as a function of RA, Dec (in degrees).'''
        ra_arr = np.asarray(ra)
        dec_arr = np.asarray(dec)
        # Convert sky coordinates to pixel coordinates; origin=0 for numpy
        x_pix, y_pix = wcs.wcs_world2pix(ra_arr, dec_arr, 0)
        return surfden_spline_pix.ev(y_pix, x_pix)

    def protocluster_func(ra, dec):
        '''Protocluster number (nearest) as a function of RA, Dec (in degrees).'''
        ra_arr = np.asarray(ra)
        dec_arr = np.asarray(dec)
        x_pix, y_pix = wcs.wcs_world2pix(ra_arr, dec_arr, 0)
        points = np.stack([y_pix, x_pix], axis=-1)
        return proto_interp_pix(points)

    return surfden_func, protocluster_func, area_zero_mask_deg2

def get_exptime_areas(fits_filename):
    '''
    Compute XMM/SHELA map areas from EXPTIME and MASK extensions.

    Parameters
    ----------
    fits_filename : str
        Multi-extension FITS file (e.g., XMM_N???_voronoi_sd_maglim_25.4_01_2026.fits)
        where extension 2 is MASK and extension 3 is EXPTIME.

    Returns
    -------
    area_exptime_gt0_deg2 : float
        Total area (deg^2) with EXPTIME > 0, including masked regions.
    area_exptime_gt0_unmasked_deg2 : float
        Total area (deg^2) with EXPTIME > 0 and MASK == 0.
    '''
    with fits.open(fits_filename) as hdul:
        mask = np.asarray(hdul[2].data)
        exptime = np.asarray(hdul[3].data)
        wcs = WCS(hdul[3].header)

    pix_scales_deg = proj_plane_pixel_scales(wcs)  # deg / pixel
    area_per_pix_deg2 = abs(pix_scales_deg[0] * pix_scales_deg[1])

    exptime_gt0 = exptime > 0
    unmasked = mask == 0

    n_exptime_gt0 = np.count_nonzero(exptime_gt0)
    n_exptime_gt0_unmasked = np.count_nonzero(exptime_gt0 & unmasked)

    area_exptime_gt0_deg2 = n_exptime_gt0 * area_per_pix_deg2
    area_exptime_gt0_unmasked_deg2 = n_exptime_gt0_unmasked * area_per_pix_deg2
    return area_exptime_gt0_deg2, area_exptime_gt0_unmasked_deg2

def _infer_shela_field_token(filename):
    match = re.search(r'(SHELA_P\d+)', op.basename(filename))
    return match.group(1) if match is not None else None

def _plot_line_flux_comparison(ulf_vals, catalog_line_flux_cgs, out_png):
    '''Plot measured line flux vs catalog-estimated line flux (both in 1e-17 cgs).'''
    x = np.asarray(unumpy.nominal_values(ulf_vals), dtype=float)
    y = np.asarray(catalog_line_flux_cgs, dtype=float) * 1.0e17
    cond = np.isfinite(x) & np.isfinite(y)
    x, y = x[cond], y[cond]
    if x.size == 0:
        return
    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    if hi <= lo:
        hi = lo + 1.0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=8, alpha=0.7, color='tab:blue')
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.4, label='1-1')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r'Measured line flux ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$)')
    ax.set_ylabel(r'Catalog estimated line flux ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$)')
    ax.legend(loc='best', frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

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
    field = fn.split('_')[2]
    dat = Table.read(fn, format='ascii')
    name, ra, dec = dat['index'], dat['RA'], dat['DEC']
    is_shela = _infer_shela_field_token(fn) is not None
    if is_shela:
        sep = np.zeros_like(ra, dtype=float)
    else:
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
    fac_flux = 1.0e-29 * c/wav_filt**2 * Tint/Tc * 1.0e17
    ulf = fac_flux * (ufiltf - ugrf)
    unb = fac_flux * ufiltf
    if field.lower()=='cosmos':
        dlaef, pcf = getLAEDensity(band=filt.upper(), field=field)
    else:
        shela_token = _infer_shela_field_token(fn)
        if shela_token is not None:
            fits_pattern = f'{shela_token}_{filt.upper()}_voronoi_sd_maglim_25.*.fits'
            fits_matches = sorted(glob.glob(fits_pattern))
            if len(fits_matches)==0:
                raise FileNotFoundError(f'Could not find FITS file matching {fits_pattern}')
            fits_name = fits_matches[0]
        else:
            fits_name = f'XMM_{filt.upper()}_voronoi_sd_maglim_25.4_01_2026.fits'
        dlaef, pcf, area_not_mask = make_voronoi_interpolators(fits_name)
    if field.lower()!='cosmos':
        dlaes, pcs = dlaef(ra, dec), pcf(ra, dec)
        print("Area not covered by mask", area_not_mask)
        if not is_shela:
            area_tot = np.pi*2.4**2
            print("Area of circle with radius 2.4 deg", area_tot)
            print("Fraction of the circular area occupied by sources", area_not_mask/area_tot)
    else:
        try: dlaes = dlaef.ev(ra, dec)
        except: dlaes = dlaef(np.column_stack((ra, dec)))
        dlaes[dlaes<0] = 0.0
        pcs = pcf(np.column_stack((ra, dec)))
    if 'estimated line flux (cgs)' in dat.colnames:
        out_plot = op.basename(fn).replace('.csv', '_lineflux_comparison.png')
        _plot_line_flux_comparison(ulf, dat['estimated line flux (cgs)'], out_plot)
    else:
        print('Column "estimated line flux (cgs)" not found; skipping line-flux comparison plot.')
    return name, ra, dec, unumpy.nominal_values(ulf), unumpy.std_devs(ulf), unumpy.nominal_values(unb), unumpy.std_devs(unb), dlaes, pcs, sep

def getLAEDensity(band='N501', field='COSMOS'):
    fn = f'{field}_{band}_sd_and_pcs.txt'
    if not op.exists(fn): return None, None
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
        print("On the exception for density stuff")
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

def getIntRem(filter='N501', distmax=1.0):
    fnew, fold, fir = f'Lya{filter}FluxesFinal.dat', f'Lya{filter}FluxesFinalOld.dat', f'Lya{filter}FluxesFinalIntRem.dat'
    new, old, ir = Table.read(fnew, format='ascii'), Table.read(fold, format='ascii'), Table.read(fir, format='ascii')
    gno, gni = old['Galaxy_name'], ir['Galaxy_name']
    inds_int = []
    for i, go in enumerate(gno):
        if go not in gni: inds_int.append(i)
    inds_int = np.array(inds_int)
    ran, decn, rai, deci = new['RA'], new['Dec'], old['RA'][inds_int], old['Dec'][inds_int]
    coordsn, coordsi = SkyCoord(ran, decn, unit='degree'), SkyCoord(rai, deci, unit='degree')
    indsrem, minseps = [], []
    for i, ci in enumerate(coordsi):
        sep = coordsn.separation(ci).arcsec
        minseps.append(sep.min())
        if sep.min()<distmax: indsrem.append(np.argmin(sep))
    indsrem = np.array(indsrem)
    inds_all = np.arange(len(ran))
    inds_keep = np.setdiff1d(inds_all, indsrem)
    new_ir = new[inds_keep]
    new_ir.write(f'Lya{filter}FluxesIntRemNew.dat', format='ascii', overwrite=True)

def main(filter='N501', field='Cosmos'):
    if filter=='N501': col, wav = 'gr', 5014.0
    elif filter=='N419': col, wav ='rg', 4193.0
    else: col, wav = 'gi', 6750.0
    if field=='Cosmos': date, hs = '2024_08_01', 'half_stacks_'
    elif 'SHELA' in field.upper():
        date, hs = '2024_09_19', ''
    else:
        date, hs = '2024_09_19', ''
    fn = f'LAE_catalog_{field}_{col}-{filter.lower()}_SE_{hs}{date}_expanded.csv'
    tfn = f'{filter}_Nicole.txt'
    if field=='Cosmos':
        center_use = cosmos_center
    else:
        center_use = xmmlss_center
    names, ras, decs, lyf, lyfe, nbf, nbfe, dlaes, pcs, seps = getLineFlux(fn=fn, tfn=tfn, wav_filt=wav, center=center_use)
    dat = Table()
    dat['Galaxy_name'] = names
    dat['RA'] = ras
    dat['DEC'] = decs
    dat['Dec'] = decs
    dat['Lya_flux'] = lyf
    dat['Lya_flux_e'] = lyfe
    dat['NB_flux'], dat['NB_flux_e'] = nbf, nbfe
    dat['dist'] = seps
    dat['Density'] = dlaes
    dat['Protocluster'] = pcs
    dat.write(f'Lya{filter}{field}Fluxes.dat', format='ascii', overwrite=True)

if __name__ == '__main__':
    # main('N501', field='XMMLSS')
    # getIntRem('N419')
   area1, area2 = get_exptime_areas('XMM_N673_voronoi_sd_maglim_25.4_01_2026.fits')
   print(area1, area2)
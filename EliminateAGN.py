import numpy as np
from astropy.io import fits
import pickle
from astropy.table import Table

max_sep = 1.0

def get_dist(ra1,dec1,ra2,dec2):
    costh = np.sin(np.pi/180.*dec1)*np.sin(np.pi/180.*dec2) + np.cos(np.pi/180.*dec1)*np.cos(np.pi/180.*dec2)*np.cos(np.pi/180.*(ra2-ra1))
    th = np.arccos(costh)
    return th*3600. * 180./np.pi # Distance in arcsec

def convert_bright(num_col=3):
    types = [int, float, float, float, float, float]
    delimiters = ['  ', '\t']
    with open('brightest_onlyAGN.txt', 'r') as f:
        hsp = f.readline().split('  ')
        hsp = [h for h in hsp if h!='']
        hsp = [h.strip() for h in hsp]
        lh = len(hsp)
        full_dat = []
        for line in f:
            lsp = [line]
            for d in delimiters:
                temp = []
                for item in lsp:
                    temp.extend(item.split(d))
                lsp = temp
            lsp = [l for l in lsp if l!='']
            lsp = [l.strip() for l in lsp]
            full_dat.append(lsp[:num_col])
    breakpoint()
    full_dat = np.array(full_dat)
    new_dict = {}
    for i in range(num_col):
        new_dict[hsp[i]] = full_dat[:,i].astype(types[i])
    breakpoint()
    pickle.dump(new_dict, open('brightest_notNormalLAE.pickle', 'wb'), protocol=5)

def main(filter='N501'):
    # convert_bright()
    dat = Table.read(f'Lya{filter}FluxesFinal.dat', format='ascii')
    ra, dec = dat['RA'], dat['Dec']

    # DESI QSO findings
    desi = fits.getdata('ODIN_DESI_qso.fits')
    desi_ra, desi_dec = desi['ra'], desi['dec']

    # HETDEX AGN findings
    hd_ra, hd_dec = np.array([150.139160, 150.139297, 150.192719, 150.799179, 150.385727, 150.823578, 150.444580, 150.351395, 149.697174, 150.264114]), np.array([2.235123, 2.235053, 2.220048, 2.758963, 2.846420, 1.829407, 2.540313, 2.322358, 2.344097, 2.328962])

    # Combine the bad RAs and DECs to do a combined search to get the LAE index
    ra_bad, dec_bad = np.concatenate((desi_ra, hd_ra)), np.concatenate((desi_dec, hd_dec))
    
    inds_bad = []
    for i in range(len(ra_bad)):
        dists = get_dist(ra, dec, ra_bad[i], dec_bad[i])
        ind = np.argmin(dists)
        if dists[ind] > max_sep: 
            print(f"For index {ind}, closest match in LAE catalog to DESI is {dists[ind]}")
            continue
        inds_bad.append(ind)

    # X-ray Chandra catalog
    xray_file = pickle.load(open('X_Ray_Matching_08_23.pickle', 'rb'))
    xray = xray_file['xray_flux']
    inds_xray = np.where(xray>0)[0]

    if filter=='N501':
        # Main LAE file from Vandana
        cosmos = fits.getdata('COSMOS_N501_LAEs_starmasked_08_23.fits')
        ra_old, dec_old = cosmos['x_world'], cosmos['y_world']
        # NED + Old DESI weird object findings
        bright = pickle.load(open('brightest_notNormalLAE.pickle', 'rb'))
        ids_not = bright['Line']-1
        for id in ids_not: 
            dists = get_dist(ra, dec, ra_old[id], dec_old[id])
            ind = np.argmin(dists)
            if dists[ind] > max_sep:
                print(f"For index {ind}, closest match in LAE catalog to Robin's bad-source catalog is {dists[ind]}")
                continue
            inds_bad.append(ind)
        inds_bad.append(1047); inds_bad.append(2072)

    # Combine all the datasets
    all_ids_not = np.concatenate((inds_bad, inds_xray))
    all_ids_not = np.unique(all_ids_not) #Don't want repeats

    # Remove these indices from the flux file
    new_dat = Table()
    all_inds = np.arange(len(ra))
    assert len(dat['Galaxy_name']) == len(ra)
    new_inds = np.setdiff1d(all_inds, all_ids_not)
    for col in dat.columns:
        col_vals = dat[col]
        new_dat[col] = col_vals[new_inds]
    # Write a new file
    new_dat.write(f'Lya{filter}FluxesFinalIntRem.dat', format='ascii', overwrite=True)

if __name__ == '__main__':
    main('N673')
import subprocess
import os
import sys
import re
from astropy.stats import poisson_conf_interval
import requests
from io import BytesIO
from astropy.io import votable
from astropy.coordinates import SkyCoord
from astropy.table import Table
import math as math
import pandas as pd

from astropy import wcs
from astropy.io import fits
import numpy as np
import glob


def get_wcs_event(fname: str) -> wcs.WCS:
    """
    ## A function to get a WCS object from a Chandra event2 file.


    ### Args:
        fname `str`: File name of the event2 file.

    ### Returns:
        `wcs.WCS`: WCS object.
    """
    # Read the header of HDU 1 that contains wcs info.
    header = fits.open(fname)[1].header

    # Create an empty WCS
    wcs_evt2 = wcs.WCS(naxis=2)

    # Get ra, dec col number
    for key, val in header.items():
        if val == 'RA---TAN':
            ra_col = key[5:]
        if val == 'DEC--TAN':
            dec_col = key[5:]

    # fill in the wcs info.
    wcs_evt2.wcs.crpix = np.array(
        [header['TCRPX'+ra_col], header['TCRPX'+dec_col]])
    wcs_evt2.wcs.cdelt = np.array(
        [header['TCDLT'+ra_col], header['TCDLT'+dec_col]])
    wcs_evt2.wcs.cunit = np.array(
        [header['TCUNI'+ra_col], header['TCUNI'+dec_col]])
    wcs_evt2.wcs.crval = np.array(
        [header['TCRVL'+ra_col], header['TCRVL'+dec_col]])
    wcs_evt2.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    return wcs_evt2


def get_chandra_eef(thetas: list[float], R0: float = 1.32, R10: float = 10.1, alpha: float = 2.42) -> list:
    """
    ## A function to calculate Chandra EEF (encircled energy fraction) radius from Vito+16


    ### Args:
        thetas `list[float]`: A list of off-axis angle (') in arcmin.
        R0 `float` (optional): EEF radius (") for off-axis angle = 0. Defaults to `1.32`. 1.07 (90% EEF, from Table A1 of Vito+16)
        R10 `float` (optional): EEF radius (") for off-axis angle = 0. Defaults to `10.1`. 9.65 (90% EEF, from Table A1 of Vito+16)
        alpha `float` (optional): ?. Defaults to `2.42`.

    ### Returns:
        `list`: EEF radia from thetas (") in arcmin.
    """
    # Creat the EEF array
    Rs = np.zeros(len(thetas)) - 99.

    # Get sources with positive off-axis angle
    use_idxs = np.where(thetas >= 0)[0]

    # Derive EEF
    Rs[use_idxs] = R0 + R10 * (thetas[use_idxs] / 10.)**alpha

    # Check if all sources have positive off-aixs angle
    num_bad = len(thetas) - len(use_idxs)
    if num_bad > 0:
        print("warning: %d sources are not calculated due to negative off-axis angle" % num_bad)

    return Rs


def get_counts_from_event(
    evt_data: dict,
    src_x: float,
    src_y: float,
    Rsrc: int,
    Rbkg: int,
    sigma: float = 1.,
    output: str = 'net'
) -> tuple:
    """
    ## A functon to do photometry based on Chandra event2 table.


    ### Args:
        evt_data `dict`: The raw event 2 table.
        src_x `float`: The physical x coordinate of the source in the observation.
        src_y `float`: The physical y coordinate of the source in the observation.
        Rsrc `int`: Source is extracted within a circle Rin=Rsrc (px).
        Rbkg `int`: Background is an annulus, Rout=Rbkg (px).
        sigma `float` (optional): The confidence level for count errors. Defaults to `1.`.
        output `str` (optional): What values to output. Defaults to `'net'`.

    ### Returns:
        `tuple`: 
            if output = 'net': 
                Ncnt_net (net counts),
                Ncnt_net_loerr (lower error),
                Ncnt_net_uperr (upper error). 
            if output = 'tot_bkg': 
                Ncnt_tot.
                Ncnt_bkg*scl_fac.
    """
    # Get source area and background area
    area_src = np.pi * Rsrc**2
    area_bkg = np.pi * (Rbkg**2 - Rsrc**2)

    # Calculate background scale factor
    scl_fac = area_src/area_bkg

    # Calculate distance square
    r_sqs = (evt_data['x'] - src_x)**2 + (evt_data['y'] - src_y)**2

    # Select events within the aperture
    evt_in_Rsrc_idxs = np.where(r_sqs <= Rsrc**2)[0]
    evt_in_Rbkg_idxs = np.where(r_sqs <= Rbkg**2)[0]

    # Get the counts in source extraction aperture
    Ncnt_tot = len(evt_in_Rsrc_idxs)

    # Get background counts
    Ncnt_bkg = len(evt_in_Rbkg_idxs) - Ncnt_tot

    if output == 'net':
        # Get net counts and uncertainties
        # QUESTION: why is get_net_cnt not defined?
        Ncnt_net, Ncnt_net_loerr, Ncnt_net_uperr = get_net_cnt(Ncnt_tot, Ncnt_bkg, scl_fac,
                                                               sigma=sigma)
        return Ncnt_net, Ncnt_net_loerr, Ncnt_net_uperr
    elif output == 'tot_bkg':
        return Ncnt_tot, Ncnt_bkg*scl_fac
    else:
        sys.exit('output not recognized; photometry.ipynb')


def get_before_after_counts(event_data_raw: dict, src_xs: float, src_ys: float, R_apers: int, E_low: float = 5e2, E_up: float = 7e3) -> tuple[int, int, int, int]:
    """
    ## A function to calculate N1, N2, and N1', N2' described in the draft.


    ### Args:
        event_data_raw `dict`: The raw event 2 table.
        src_xs `float`: The physical x coordinate of sources in the observation.
        src_ys `float`: The physical y coordinate of sources in the observation.
        R_apers `int`: The aperture size for each source (px).
        E_low `float` (optional): The lower limit of the Chandra energy band used. Defaults to `5e2`.
        E_up `float` (optional): The upper limit of the Chandra energy band used. Defaults to `7e3`.

    ### Returns:
        `tuple[int, int, int, int]`: A tuple of N1 (counts before med_exp), N2 (counts after med_exp), N1' (counts at the edge), N2' (counts at the center).
    """
    # Only use 0.5-7 keV range
    event_data = event_data_raw[np.where((event_data_raw['energy'] >= E_low) &
                                         (event_data_raw['energy'] <= E_up))]

    # Get the time bins
    t_q0, t_q4 = np.min(event_data['time']), np.max(event_data['time'])
    t_qdet = (t_q4-t_q0)/4
    t_q1, t_q2, t_q3 = t_q0+t_qdet, t_q0+2*t_qdet, t_q0+3*t_qdet

    # Event data before/after med_exp
    evt_idxs_bf = np.where(event_data['time'] < t_q2)[0]
    evt_idxs_aft = np.where(event_data['time'] >= t_q2)[0]
    evt_idxs_edg = np.where(
        (event_data['time'] < t_q1) | (event_data['time'] >= t_q3))[0]
    evt_idxs_cent = np.where(
        (event_data['time'] >= t_q1) & (event_data['time'] < t_q3))[0]

    # Extract event x, y
    evt_xs, evt_ys = event_data['x'], event_data['y']

    # Count number of sources
    Nsrc = len(src_xs)

    # Calculate events before and after
    Ncnt_bfs,  Ncnt_afts = [], []
    Ncnt_edgs, Ncnt_cents = [], []

    # Iterate over each source
    for src_idx in range(Nsrc):
        # Extract source and background region
        R_src, R_bkg = R_apers[src_idx], R_apers[src_idx]+22.

        # Do a photometry to get total and background counts
        Ncnt_tot, Ncnt_bkg_scl = get_counts_from_event(event_data, src_xs[src_idx], src_ys[src_idx],
                                                       R_src, R_bkg, output='tot_bkg')

        # Check if background is too strong
        if Ncnt_tot < 5*Ncnt_bkg_scl:
            Ncnt_bfs.append(-99)
            Ncnt_afts.append(-99)
            Ncnt_edgs.append(-99)
            Ncnt_cents.append(-99)
        else:
            # Calculate distance square
            r_sqs = (evt_xs-src_xs[src_idx])**2 + (evt_ys-src_ys[src_idx])**2

            # Select events within the aperture
            evt_in_aper_idxs = np.where(r_sqs <= R_src**2)[0]

            # Count events before and after med_exp
            Ncnt_bfs.append(len(np.intersect1d(evt_in_aper_idxs, evt_idxs_bf)))
            Ncnt_afts.append(
                len(np.intersect1d(evt_in_aper_idxs, evt_idxs_aft)))
            Ncnt_edgs.append(
                len(np.intersect1d(evt_in_aper_idxs, evt_idxs_edg)))
            Ncnt_cents.append(
                len(np.intersect1d(evt_in_aper_idxs, evt_idxs_cent)))

    # Convert lists to arrays
    Ncnt_bfs = np.array(Ncnt_bfs)
    Ncnt_afts = np.array(Ncnt_afts)
    Ncnt_edgs = np.array(Ncnt_edgs)
    Ncnt_cents = np.array(Ncnt_cents)

    return Ncnt_bfs, Ncnt_afts, Ncnt_edgs, Ncnt_cents


def get_transient_candidates(Ncnt_bfs, Ncnt_afts):
    # A function to select candidats based on N1 and N2 (N1' and N2')
    '''
    Input:
        Ncnt_bfs, Ncnt_afts: counts before and after med_exp
    Output:
        xt_flags, if True, transient candidate
                  if False, not XT candidate
    '''
    # Create the flags
    xt_flags = np.array([False] * len(Ncnt_bfs))

    # Only calculate sources with S/N not too low
    use_idxs = np.where(Ncnt_bfs >= 0)[0]

    # Calculate counts upper and lower Poisson limit
    Ncnt_bf_lolims,  Ncnt_bf_uplims = poisson_conf_interval(Ncnt_bfs[use_idxs],
                                                            interval='frequentist-confidence', sigma=5)
    Ncnt_aft_lolims, Ncnt_aft_uplims = poisson_conf_interval(Ncnt_afts[use_idxs],
                                                             interval='frequentist-confidence', sigma=5)

    # Select XT candidates
    xt_flags[use_idxs] = \
        ((Ncnt_afts[use_idxs] > Ncnt_bf_uplims) | (Ncnt_afts[use_idxs] < Ncnt_bf_lolims)) & \
        ((Ncnt_bfs[use_idxs] > Ncnt_aft_uplims) | (Ncnt_bfs[use_idxs] < Ncnt_aft_lolims)) & \
        ((Ncnt_bfs[use_idxs] > 5*Ncnt_afts[use_idxs]) |
         (Ncnt_afts[use_idxs] > 5*Ncnt_bfs[use_idxs]))

    return xt_flags


def transient_selection(evt_data_raw, src_xs, src_ys, R_apers):
    Ncnt_bfs, Ncnt_afts, Ncnt_edgs, Ncnt_cents = get_before_after_counts(
        evt_data_raw,
        src_xs,
        src_ys,
        R_apers,
        E_low=5e2,
        E_up=7e3
    )
    # Select candidate
    # By N1 and N2
    flags_1 = get_transient_candidates(Ncnt_bfs, Ncnt_afts)

    # By N1' and N2'
    flags_2 = get_transient_candidates(Ncnt_edgs, Ncnt_cents)

    # Combine the results
    xt_idxs = np.where(flags_1 | flags_2)[0]

    return xt_idxs


def chopp(evt_data_raw, src_xs, src_ys, R_apers):
    xt_idxs = transient_selection(evt_data_raw, src_xs, src_ys, R_apers)
    return xt_idxs

# Below are an example of usage
# Note: a helpful Chandra observation list can be found at
# https://heasarc.gsfc.nasa.gov/db-perl/W3Browse/w3table.pl?tablehead=name%3Dchanmaster&Action=More+Options


def Yang_search(filename, ra, dec, theta, err_pos, significance):
    with fits.open(filename) as hdul:
        information = hdul[1].header
        obs = information['OBS_ID']
        t_start = information['TSTART']
        t_stop = information['TSTOP']

    # Set obsID here
    obs_id = int(obs)
    print(obs_id)

    # Set event2 file path
    evt_fname = filename
    print(evt_fname)

    # Set pixel scale (units: arcsec per pix)
    acis_pix_size = 0.492

    # Query CSC for sources in the observation
    # src_cat = queryCSC(obs_id)

    # Read the event file
    evt_data_raw = Table.read(evt_fname, hdu=1)

    # Read the wcs of the event file
    evt_wcs = get_wcs_event(evt_fname)

    # Extract ra, dec of sources in the observation
    src_ras, src_decs, src_thetas = \
        np.array(ra), np.array(dec), np.array(theta)
    src_pos, src_sig = np.array(err_pos), np.array(significance)
    # src_r0s, src_r1s= np.array(src_cat['err_ellipse_r0']), np.array(src_cat['err_ellipse_r1'])

    # Convert ra,dec to x,y
    src_xs, src_ys = evt_wcs.all_world2pix(src_ras, src_decs, 1)

    # Get R90 size
    r90_s = get_chandra_eef(src_thetas, R0=1.07, R10=9.65, alpha=2.22)

    # Convert to pixel scale
    r90_s /= acis_pix_size

    # Get the aperture size
    R_apers = r90_s*1.5

    # Get counts
    # split the observation
    time = (t_stop - t_start) / 1000.0
    candidates = []
    if time <= 20.0:
        xt_idxs = chopp(evt_data_raw, src_xs, src_ys, R_apers)
        for j in range(0, len(xt_idxs)):
            candidates.append(xt_idxs[j])
    else:
        xt_idxs = chopp(evt_data_raw, src_xs, src_ys, R_apers)
        for j in range(0, len(xt_idxs)):
            candidates.append(xt_idxs[j])

        w = 20.0  # window size
        N_regions = int(time/w)
        res = time - (w * N_regions)
        array = []

        # Forward
        for i in range(0, N_regions):
            low_lim, high_lim = (w * 1000 * i) + \
                t_start, (w*1000*(i+1))+t_start
            print(low_lim-t_start, high_lim-t_start, res, N_regions)
            evt_data = evt_data_raw[np.where(
                (evt_data_raw['time'] >= low_lim) &
                (evt_data_raw['time'] < high_lim)
            )]
            array.append(high_lim-low_lim)
            if (len(evt_data) != 0):
                xt_idxs = chopp(evt_data, src_xs, src_ys, R_apers)
                for j in range(0, len(xt_idxs)):
                    candidates.append(xt_idxs[j])
        low_lim, high_lim = (w * 1000 * (i + 1)) + t_start, (w *
                                                             1000 * (i + 1)) + t_start + res * 1000
        print(low_lim-t_start, high_lim-t_start, res, N_regions)
        array.append(high_lim-low_lim)
        evt_data = evt_data_raw[np.where(
            (evt_data_raw['time'] >= low_lim) &
            (evt_data_raw['time'] < high_lim)
        )]
        if (len(evt_data) != 0):
            xt_idxs = chopp(evt_data, src_xs, src_ys, R_apers)
            for j in range(0, len(xt_idxs)):
                candidates.append(xt_idxs[j])
        print(array)

        # Reverse
        for i in range(0, len(array)):
            if (i == 0):
                low_lim, high_lim = 0, array[-1]
                print(low_lim, high_lim, res, N_regions)
                evt_data = evt_data_raw[np.where(
                    (evt_data_raw['time'] >= low_lim) &
                    (evt_data_raw['time'] < high_lim)
                )]
                if (len(evt_data) != 0):
                    xt_idxs = chopp(evt_data, src_xs, src_ys, R_apers)
                    for j in range(0, len(xt_idxs)):
                        candidates.append(xt_idxs[j])
            else:
                low_lim, high_lim = array[-1]+w*(i-1)*1000, array[-1]+w*i*1000
                print(low_lim, high_lim, res, N_regions)
                evt_data = evt_data_raw[np.where(
                    (evt_data_raw['time'] >= low_lim) &
                    (evt_data_raw['time'] < high_lim)
                )]
                if (len(evt_data) != 0):
                    xt_idxs = chopp(evt_data, src_xs, src_ys, R_apers)
                    for j in range(0, len(xt_idxs)):
                        candidates.append(xt_idxs[j])

        # Shift
        a = 0.0
        for i in range(0, len(array)):
            a = a+array[i]
            low_lim, high_lim = a-w/2.*1000, a+w/2.*1000
            print(low_lim, high_lim)
            evt_data = evt_data_raw[np.where((evt_data_raw['time'] >= low_lim) &
                                             (evt_data_raw['time'] < high_lim))]
            if (len(evt_data) != 0):
                xt_idxs = chopp(evt_data, src_xs, src_ys, R_apers)
                for j in range(0, len(xt_idxs)):
                    candidates.append(xt_idxs[j])

    myFinalList = pd.unique(candidates).tolist()
    g = open("../detections_CSC20_w20.txt", "a")
    for i in range(0, len(myFinalList)):
        print(obs_id)
        g.write(str(obs_id) + ' ' + str(src_ras[myFinalList[i]]) + ' ' + str(src_decs[myFinalList[i]]) + ' ' + str(
            src_thetas[myFinalList[i]]) + ' ' + str(src_pos[myFinalList[i]]) + ' ' + str(src_sig[myFinalList[i]]) + '\n')
    g.close()

    f = open("../analyzed.txt", "a")
    f.write(str(obs_id)+"\n")
    f.close()


def off_axis(event_file, ra, dec):
    command = 'dmcoords ' + event_file + ' op=cel ra=' + \
        str(ra) + ' dec=' + str(dec) + ' verbose=1'
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    # out=np.array(out)
    output = out.split()
    for i in range(0, len(output)):
        if (output[i] == b'THETA,PHI'):
            z = output[i + 1]
            break
    return z


def search_candidates(src_file, event_file):
    with fits.open(src_file) as hdul:
        RA = hdul[1].data['RA']
        RA_err = hdul[1].data['RA_err']
        DEC = hdul[1].data['DEC']
        DEC_err = hdul[1].data['DEC_err']
        X = hdul[1].data['X']
        Y = hdul[1].data['Y']
        X_err = hdul[1].data['X_err']
        Y_err = hdul[1].data['Y_err']
        significance = hdul[1].data['SRC_SIGNIFICANCE']
    THETA = []
    err_pos = []
    print(X)
    for i in range(0, len(X)):
        a = off_axis(event_file, RA[i], DEC[i])
        b = a.decode("utf-8")
        print(b)
        if (b[-1] == '"'):
            THETA.append(float(b[:-1])/60.0)
        else:
            THETA.append(float(b[:-1]))
        err_pos.append(np.sqrt(X_err[i]**2+Y_err[i]**2)*0.492)
    Yang_search(event_file, RA, DEC, THETA, err_pos, significance)


files = glob.glob('s3_expmap_src.fits', recursive=True)
print(files)
src_file = files[0]
files = glob.glob('*evt2.fits', recursive=True)
event_file = files[0]
print(src_file, event_file)
search_candidates(src_file, event_file)

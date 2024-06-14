import time
import pandas as pd
from astropy.table import Table
from auxiliary.search_algorithm import get_wcs_event, off_axis, get_chandra_eef, get_counts_from_event
import glob
import numpy as np
from astropy.stats import poisson_conf_interval
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from results import gen_light_curve
from auxiliary.search_algorithm import *


def get_random_light_curves(n: int = 10, from_date: str = '', to_date: str = '') -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
    """
    Get random light curves for a given date range.

    Args:
        n (int, optional): Number of random light curves. Defaults to 10.
        from_date (str, optional): Start date of range. Defaults to ''. Format: 'YYYY-MM-DD'. If empty, no lower bound. Inclusive.
        to_date (str, optional): End date of range. Defaults to ''. Format: 'YYYY-MM-DD'. If empty, no upper bound. Exclusive.

    Returns:
        Dict[int, pd.DataFrame]: Dictionary of random light curves.
    """
    filtered = pd.read_csv('output/filtered_w20.csv', header=0, dtype=str)

    obsids_1 = pd.read_csv('obsid_lists/obsids_b+10_220401+.csv',
                           header=0, dtype=str, sep=',', usecols=['Obs ID', 'Public Release Date'])
    obsids_2 = pd.read_csv('obsid_lists/obsids_b-10_220401+.csv',
                           header=0, dtype=str, sep=',', usecols=['Obs ID', 'Public Release Date'])
    obsids_3 = pd.read_csv('obsid_lists/obsids_b+10_220401-.csv',
                           header=0, dtype=str, sep=',', usecols=['Obs ID', 'Public Release Date'])
    obsids_4 = pd.read_csv('obsid_lists/obsids_b-10_220401-.csv',
                           header=0, dtype=str, sep=',', usecols=['Obs ID', 'Public Release Date'])
    obsids = pd.concat([obsids_1, obsids_2, obsids_3,
                       obsids_4], ignore_index=True)
    obsids['Public Release Date'] = pd.to_datetime(
        obsids['Public Release Date'])

    if from_date:
        from_date = pd.to_datetime(from_date)
        obsids = obsids[obsids['Public Release Date'] >= from_date]

    if to_date:
        to_date = pd.to_datetime(to_date)
        obsids = obsids[obsids['Public Release Date'] < to_date]

    filtered = filtered[filtered['ObsId'].isin(obsids['Obs ID'])]

    random_detections = filtered.sample(n=n)

    random_light_curves = {}
    for i, row in random_detections.iterrows():
        obsid = row['ObsId']
        fxt_ra = float(row['RA'])
        fxt_dec = float(row['DEC'])
        fxt_theta = float(row['THETA'])
        fxt_pos_err = float(row['POS_ERR'])
        try:
            random_light_curves[i] = (gen_light_curve(
                obsid, fxt_ra, fxt_dec, fxt_theta, fxt_pos_err))
        except Exception as e:
            print(
                f'Error generating light curve for {obsid} - RA: {fxt_ra:.3f}, DEC: {fxt_dec:.3f}: {e}')

    return random_detections, random_light_curves


def plot_random_light_curves(random_detections: pd.DataFrame, random_light_curves: Dict[int, pd.DataFrame], save_location: str = 'plots/light_curves'):
    """
    Plot random light curves.

    Args:
        random_detections (pd.DataFrame): Random detections.
        random_light_curves (Dict[int, pd.DataFrame]): Random light curves.
        save_location (str, optional): Location to save the plots. Defaults to 'plots/light_curves'.
    """
    for i, row in random_detections.iterrows():
        obsid = row['ObsId']
        fxt_ra = float(row['RA'])
        fxt_dec = float(row['DEC'])

        try:
            light_curve = random_light_curves[i]
        except KeyError:
            print(
                f'No light curve found for {obsid} - RA: {fxt_ra:.3f}, DEC: {fxt_dec:.3f}')
            continue

        plt.figure()
        plt.errorbar((light_curve['time'] - light_curve['time'][0]) / 1000,
                     light_curve['counts'],
                     yerr=[
                     light_curve['error_low'], light_curve['error_high']], fmt='o')
        plt.title(f'{obsid} - RA: {fxt_ra:.3f}, DEC: {fxt_dec:.3f}')

        # xticks begin at 0 and should be in kiloseconds
        plt.xticks(
            np.arange(0, (light_curve['time'].max() - light_curve['time'][0]) / 1000, 10))

        plt.xlabel('Time (ks)')
        plt.ylabel('Counts')
        plt.savefig(f'{save_location}/{obsid}.png')
        plt.close()


def get_min_max_dates():
    obsids_1 = pd.read_csv('obsid_lists/obsids_b+10_220401+.csv',
                           header=0, dtype=str, sep=',', usecols=['Public Release Date'])
    obsids_2 = pd.read_csv('obsid_lists/obsids_b-10_220401+.csv',
                           header=0, dtype=str, sep=',', usecols=['Public Release Date'])
    obsids_3 = pd.read_csv('obsid_lists/obsids_b+10_220401-.csv',
                           header=0, dtype=str, sep=',', usecols=['Public Release Date'])
    obsids_4 = pd.read_csv('obsid_lists/obsids_b-10_220401-.csv',
                           header=0, dtype=str, sep=',', usecols=['Public Release Date'])
    obsids = pd.concat([obsids_1, obsids_2, obsids_3,
                        obsids_4], ignore_index=True)
    obsids['Public Release Date'] = pd.to_datetime(
        obsids['Public Release Date'])
    print(obsids['Public Release Date'].min())
    print(obsids['Public Release Date'].max())


def transient_selection_test(
    event_data_raw: Table,
    source_xs: list[float],
    source_ys: list[float],
    aperture_radii: list[int],
    t_begin: float,
    t_end: float
) -> np.ndarray[bool]:
    """
    Select transient candidates based on the counts before and after the event.

    Args:
        event_data_raw (Table): The raw event 2 table.
        source_xs (list[float]): list of x coordinates of sources.
        source_ys (list[float]): list of y coordinates of sources.
        aperture_radii (list[int]): list of aperture sizes, in pixels.

    Returns:
        np.ndarray[bool]: A boolean array indicating whether the source is a transient candidate.
    """
    before_counts, after_counts, edge_counts, center_counts = \
        get_before_after_counts(
            event_data_raw,
            source_xs,
            source_ys,
            aperture_radii,
            t_begin,
            t_end,
            lower_energy=5e2,
            upper_energy=7e3
        )

    print(
        'from function:',
        f"\tq1 + q2: {before_counts[0]}",
        f"\tq3 + q4: {after_counts[0]}",
        f"\tq1 + q4: {edge_counts[0]}",
        f"\tq2 + q3: {center_counts[0]}",
        sep='\n'
    )

    # Select candidate
    # By N1 and N2
    candidates_1 = get_transient_candidates(before_counts, after_counts)

    # By N1' and N2'
    candidates_2 = get_transient_candidates(edge_counts, center_counts)

    # Combine the results
    transient_candidates = np.where(candidates_1 | candidates_2)[0]

    return transient_candidates


def test_search_algorithm(detection: pd.Series, window: int = 20):
    obsid = detection['ObsId']
    fxt_ra = float(detection['RA'])
    fxt_dec = float(detection['DEC'])
    fxt_theta = float(detection['THETA'])
    fxt_pos_err = float(detection['POS_ERR'])

    current_dir = os.getcwd()

    os.chdir(f'/data/jcrans/fxrt-data/obsids/{obsid}')

    try:
        files = glob.glob('s3_expmap_src.fits', recursive=True)
        src_file = files[0]

        files = glob.glob('*evt2.fits', recursive=True)
        event_file = files[0]
    except IndexError:
        print(f'Error: No files found for {obsid}')
        os.chdir(current_dir)
        return

    with fits.open(event_file) as hdul:
        information = hdul[1].header
        obs = information['OBS_ID']
        t_start = information['TSTART']
        t_stop = information['TSTOP']

    event_data_raw = Table.read(event_file, hdu=1)

    event_wcs = get_wcs_event(event_file)
    fxt_x, fxt_y = \
        event_wcs.all_world2pix(fxt_ra, fxt_dec, 1)

    # Get R90 size
    r90_size = get_chandra_eef(
        np.array([fxt_theta]), R0=1.07, R10=9.65, alpha=2.22)[0]

    # Convert to pixel scale
    acis_pix_size = 0.492
    r90_size /= acis_pix_size

    # Get the aperture size
    aperture_radius = r90_size * 1.5

    # full observation
    is_candidate = transient_selection_test(
        event_data_raw,
        [fxt_x],
        [fxt_y],
        [aperture_radius],
        t_start,
        t_stop
    )

    if len(is_candidate) > 0:
        print(f'{obsid} - RA: {fxt_ra:.3f}, DEC: {fxt_dec:.3f} - full')

    # window is larger than observation so rest is unnecessary
    if (t_stop - t_start) / 1000.0 < window:
        os.chdir(current_dir)
        return

    # split the observation
    for t_begin, t_end in get_start_end_times((t_stop - t_start) / 1000.0, window):
        t_begin, t_end = t_begin * 1000.0 + t_start, t_end * 1000.0 + t_start

        event_data = event_data_raw[np.where(
            (event_data_raw['time'] >= t_begin) &
            (event_data_raw['time'] < t_end)
        )]

        if (len(event_data) == 0):
            continue

        is_candidate = transient_selection_test(
            event_data,
            [fxt_x],
            [fxt_y],
            [aperture_radius]
        )

        if len(is_candidate) > 0:
            print(
                f'{obsid} - RA: {fxt_ra:.3f}, DEC: {fxt_dec:.3f} - [{(t_begin - t_start) / 1000}, {(t_end - t_start) / 1000}]')

    os.chdir(current_dir)

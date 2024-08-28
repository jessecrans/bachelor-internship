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
from search import download_data, process_data, DATA_PATH
from results import find_obsids_matching_detection, get_new_detections, get_no_match_fxts, get_candidates
import os
import subprocess
import re


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
    filtered = pd.read_csv('output/detections_w20.txt',
                           header=0, dtype=str, sep=' ')

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
    """
    ## Prints the minimum and maximum public release dates of the observations.
    """
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

    # print(obsids['Public Release Date'].min())
    # print(obsids['Public Release Date'].max())

    return obsids['Public Release Date'].min(), obsids['Public Release Date'].max()


def get_start_end_times(exposure_time: float, window: float) -> list[tuple[float, float]]:
    """
    ## Get every start and end time for the given exposure time and window size.

    Calculated by splitting the exposure according to three passes.
    1. Split into windows of the given size plus a residual window.
    2. Backward split into windows of the given size plus a residual window.
    3. A window of half size, then split into windows of the given size plus a residual window.

    ### Args:
        exposure_time `float`: Exposure time, in kiloseconds.
        window `float`: Window size, in kiloseconds.

    ### Returns:
        `list[tuple[float, float]]`: List of start and end times for the given exposure time and window size.
    """
    residual_limit = 8.0
    start_end_times = []

    current_start = 0.0
    current_end = window

    if exposure_time < window:
        return [(0, exposure_time)]

    # forward
    while current_end < exposure_time:
        start_end_times.append((current_start, current_end))
        current_start += window
        current_end += window
    else:  # residual window
        if exposure_time - current_start > residual_limit:
            start_end_times.append((current_start, exposure_time))

    # backward
    current_start = exposure_time - window
    current_end = exposure_time
    while current_start > 0:
        start_end_times.append((current_start, current_end))
        current_start -= window
        current_end -= window
    else:  # residual window
        if current_end > residual_limit:
            start_end_times.append((0, current_end))

    # shift
    shift = window / 2
    start_end_times.append((0, shift))

    current_start = shift
    current_end = shift + window
    while current_end < exposure_time:
        start_end_times.append((current_start, current_end))
        current_start += window
        current_end += window
    else:  # residual window
        if exposure_time - current_start > residual_limit:
            start_end_times.append((current_start, exposure_time))

    return start_end_times


def transient_selection_test(
    event_data_raw: Table,
    source_xs: list[float],
    source_ys: list[float],
    aperture_radii: list[int],
    t_begin: float,
    t_end: float
) -> np.ndarray[bool]:
    """
    ## Transient selection test function to manually check what the algorithm is doing.

    The function is adapted from the `transient_selection` function in `search_algorithm.py`. So, the code is suboptimally written.

    ### Args:
        event_data_raw `Table`: Raw event data.
        source_xs `list[float]`: X coordinate of the source.
        source_ys `list[float]`: y coordinate of the source.
        aperture_radii `list[int]`: aperture radius of the source.
        t_begin `float`: Start time of the observation.
        t_end `float`: End time of the observation.

    ### Returns:
        `np.ndarray[bool]`: Array of boolean value indicating if the source is a candidate.
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

    # Select candidate
    # By N1 and N2
    candidates_1 = get_transient_candidates(before_counts, after_counts)

    # By N1' and N2'
    candidates_2 = get_transient_candidates(edge_counts, center_counts)

    # Combine the results
    transient_candidates = np.where(candidates_1 | candidates_2)[0]

    if len(transient_candidates) > 0:
        # if True:
        print(
            f"\tq1 + q2: {before_counts[0]}",
            f"\tq3 + q4: {after_counts[0]}",
            f"\tq1 + q4: {edge_counts[0]}",
            f"\tq2 + q3: {center_counts[0]}",
            sep='\n'
        )

    return transient_candidates


def test_search_algorithm(obsid: str, fxt_ra: float, fxt_dec: float, fxt_theta: float, window: int = 20):
    """
    ## Test the search algorithm for a given source.

    ### Args:
        obsid `str`: Obsid where the source is located.
        fxt_ra `float`: Right ascension of the source.
        fxt_dec `float`: Declination of the source.
        fxt_theta `float`: Off-axis angle of the source.
        window `int` (optional): Defaults to `20`. Window size to use for the search.
    """
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

        # print(
        #     f'Checking window: [{(t_begin - t_start) / 1000}, {(t_end - t_start) / 1000}]')

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
            [aperture_radius],
            t_begin,
            t_end
        )

        if len(is_candidate) > 0:
            print(
                f'{obsid} - RA: {fxt_ra:.3f}, DEC: {fxt_dec:.3f} - [{(t_begin - t_start) / 1000}, {(t_end - t_start) / 1000}]')

    os.chdir(current_dir)


def download_missing_data():
    date_range = [
        '',
        '2015-01-01',
        # '2022-04-01',
        # ''
    ]

    detections = get_no_match_fxts(
        20, from_date=date_range[0], to_date=date_range[1])

    for i, detection in detections.iterrows():
        print(f'Checking obsid: {detection["ObsId"]}')
        matching_obsids = find_obsids_matching_detection(detection)

        for obsid in matching_obsids:
            print(f'\tChecking matching obsid: {obsid}')
            downloaded_obsids = os.listdir(DATA_PATH)
            if obsid not in downloaded_obsids:
                download_data(obsid, data_path=DATA_PATH)
                if not process_data(obsid, data_path=DATA_PATH):
                    # message printed in process_data
                    continue
            else:
                print(f'\tObsid data already downloaded')


def filter_variable_check(candidates: pd.DataFrame, window: int = 20, exclusions: list = [], from_date: str = '', to_date: str = '') -> None:
    for i, candidate in candidates.iterrows():
        print(f'Filtering: {candidate["ObsId"]}')
        # print(f'Candidate: {candidate}')
        event_file_path = glob.glob(
            f'{DATA_PATH}/{candidate["ObsId"]}/*evt2.fits', recursive=True)[0]
        src_file_path = glob.glob(
            f'{DATA_PATH}/{candidate["ObsId"]}/s3_expmap_src.fits', recursive=True)[0]
        reg_file_path = glob.glob(
            f'{DATA_PATH}/{candidate["ObsId"]}/*src.reg', recursive=True)[0]
        asol_file_path = glob.glob(
            f'{DATA_PATH}/{candidate["ObsId"]}/*asol1.fits', recursive=True)[0]

        # getting x, y coordinates of the source
        with fits.open(src_file_path, mode='readonly') as src_file:
            src_data = src_file[1].data

        src_ras = src_data['RA']
        src_decs = src_data['DEC']
        src_xs = src_data['X']
        src_ys = src_data['Y']

        src_ra = np.array(candidate['RA'], dtype=float)
        src_dec = np.array(candidate['DEC'], dtype=float)

        src_idx = np.where((src_ras == src_ra) & (src_decs == src_dec))[0][0]

        src_x = src_xs[src_idx]
        src_y = src_ys[src_idx]

        # getting region parameters
        with open(reg_file_path) as reg_file:
            reg_file = reg_file.read()

        reg_file = reg_file.split('\n')
        regex = re.compile(r'(\w+)\(([^)]+)\)')

        reg_file = [[match[0]] + list(map(float, match[1].split(',')))
                    for item in reg_file for match in regex.findall(item)]
        reg_file = np.array(reg_file)

        reg_xs = np.array(reg_file[:, 1], dtype=float)
        reg_ys = np.array(reg_file[:, 2], dtype=float)

        reg_dists = np.sqrt((reg_xs - src_x) ** 2 + (reg_ys - src_y) ** 2)
        reg_idx = np.argmin(reg_dists)

        reg_params = reg_file[reg_idx]

        reg_string = f'{reg_params[0]}({",".join(map(str, reg_params[1:]))})'
        print(f'\tregion: {reg_string}')

        src_reg_file_path = f'{DATA_PATH}/{candidate["ObsId"]}/src.reg'
        command = f'dmmakereg \"{reg_string}\" {src_reg_file_path} clobber=yes'
        proc = subprocess.run(command, shell=True)

        # getting ccd_id
        command = f'dmcoords {event_file_path} op=cel ra={src_ra} dec={src_dec}'
        proc = subprocess.run(command, shell=True)

        command = f'pget dmcoords chip_id'
        proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
        chip_id = int(proc.stdout)
        print(f'\tchip_id: {chip_id}')

        # efficiency file
        eff_file_path = f'{DATA_PATH}/{candidate["ObsId"]}/dither_region.fits'
        command = f'dither_region infile={asol_file_path} outfile={eff_file_path} region=\"region({src_reg_file_path})\" wcsfile={event_file_path} clobber=yes'
        proc = subprocess.run(command, shell=True)

        # running glvary
        vary_file_path = f'{DATA_PATH}/{candidate["ObsId"]}/gl_prob.fits'
        lc_file_path = f'{DATA_PATH}/{candidate["ObsId"]}/lc_prob.fits'
        command = f'glvary infile=\"{event_file_path}[sky=region({src_reg_file_path}),ccd_id={chip_id}]\" outfile={vary_file_path} lcfile={lc_file_path} effile=\"{eff_file_path}[cols time,dtf=fracarea]\" clobber=yes'
        proc = subprocess.run(command, shell=True, stdout=subprocess.PIPE)

        # outfile
        try:
            with fits.open(vary_file_path, mode='readonly') as vary_file:
                vary_data = vary_file[1].data
                vary_header = vary_file[1].header
        except:
            print('\tError: glvary failed')
            continue

        probability = vary_header['PROB']
        print(f'\tprob: {probability:.2f}')

        if probability <= 0.5:
            print('\tresult: not variable')
        elif probability > 0.5 and probability <= (2/3):
            print('\tresult: possibly variable')
        elif probability > (2/3):
            print('\tresult: variable')

        # remove candidate if not variable
        if probability <= 0.5:
            candidates.drop(i, inplace=True)

    # give new candidates
    candidates.reset_index(drop=True, inplace=True)
    candidates.index += 1
    candidates.to_csv(
        f'output/candidates/candidates_w_{window}_ex_{"_".join([str(i) for i in exclusions])}_daterange_{from_date}_{to_date}.csv')


parameters = [
    # {
    #     'date_range': [
    #         '2022-04-01',
    #         ''
    #     ],
    #     'window': 20,
    #     'exclusions': []
    # },
    # {
    #     'date_range': [
    #         '2015-01-01',
    #         '2022-04-01'
    #     ],
    #     'window': 20,
    #     'exclusions': []
    # },
    # {
    #     'date_range': [
    #         '',
    #         '2015-01-01'
    #     ],
    #     'window': 20,
    #     'exclusions': []
    # },
    {
        'date_range': [
            '',
            ''
        ],
        'window': 50,
        'exclusions': [20]
    }
]

excluded_obsids = ['2561']

if __name__ == '__main__':
    for params in parameters:
        candidates = get_candidates(
            params['window'], from_date=params['date_range'][0], to_date=params['date_range'][1], exclusions=params['exclusions']
        )
        candidates = candidates[~candidates['ObsId'].isin(excluded_obsids)]

        filter_variable_check(
            candidates,
            window=params['window'],
            exclusions=params['exclusions'],
            from_date=params['date_range'][0],
            to_date=params['date_range'][1]
        )

    # download_missing_data()
    # get_min_max_dates()
    # test_search_algorithm('20087', 0.0, 0.0, 0.0)
    # get_random_light_curves()
    # random_detections, random_light_curves = get_random_light_curves()
    # plot_random_light_curves(random_detections, random_light_curves)
    # pass

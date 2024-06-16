import time
import pandas as pd
from astropy.table import Table
from auxiliary.search_algorithm import get_wcs_event, off_axis, get_chandra_eef, get_counts_from_event
import glob
import numpy as np
from astropy.stats import poisson_conf_interval
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from astropy.io import fits

CRITERIA = [
    ('Archival X-ray date', [
        'archival_match',
        'chandra_match',
        'erosita_match',
        'xray-binaries_match',
    ]),
    ('Cross-match with stars/Gaia', [
        'gaia_match',
    ]),
    ('NED + SIMBAD + VizieR', [
        'ned_match',
        'simbad_match',
        'vizier_match',
    ]),
]
ALL_FILTERS = [
    filter for _, filters in CRITERIA for filter in filters
]


def gen_light_curve(obsid: str, fxt_ra: float, fxt_dec: float, fxt_theta: float, verbose: int = 0) -> pd.DataFrame:
    """
    ## Generate a light curve for a given source.

    ### Args:
        obsid `str`: Observation ID.
        fxt_ra `float`: Right ascension of the source.
        fxt_dec `float`: Declination of the source.
        fxt_theta `float`: Off-axis angle of the source.
        verbose `int` (optional): Defaults to `0`. Level of verbosity.

    ### Returns:
        `pd.DataFrame`: Light curve for the source.
    """
    # get files
    try:
        files = glob.glob(
            f'/data/jcrans/fxrt-data/obsids/{obsid}/s3_expmap_src.fits', recursive=True)
        src_file = files[0]
        files = glob.glob(
            f'/data/jcrans/fxrt-data/obsids/{obsid}/*evt2.fits', recursive=True)
        event_file = files[0]
    except IndexError:
        print(f'No files found for {obsid}')
        return pd.DataFrame()

    with fits.open(event_file) as hdul:
        information = hdul[1].header
        obs = information['OBS_ID']
        t_start = information['TSTART']
        t_stop = information['TSTOP']

    # Read the event file
    event_data_raw = Table.read(event_file, hdu=1)

    # Read the wcs of the event file
    event_wcs = get_wcs_event(event_file)

    # convert ra, dec to x, y
    fxt_x, fxt_y = event_wcs.all_world2pix(fxt_ra, fxt_dec, 1)

    # Get R90 size
    r90_size = get_chandra_eef(
        np.array([fxt_theta]), R0=1.07, R10=9.65, alpha=2.22)[0]

    # Convert to pixel scale
    acis_pix_size = 0.492
    r90_size /= acis_pix_size

    # Get the aperture size
    aperture_radii = r90_size * 1.5

    event_data = event_data_raw[np.where(
        (event_data_raw['energy'] >= 5e2) &
        (event_data_raw['energy'] <= 7e3)
    )]

    time_step = 1000
    time_bins = np.arange(t_start, t_stop, time_step)
    light_curve_total = np.zeros(len(time_bins)-1, dtype=int)
    light_curve_background = np.zeros(len(time_bins)-1, dtype=int)

    for i in range(len(time_bins)-1):
        event_data_bin = event_data[np.where(
            (event_data['time'] >= time_bins[i]) &
            (event_data['time'] < time_bins[i+1])
        )]

        total_counts, background_counts = get_counts_from_event(
            event_data_bin, fxt_x, fxt_y, aperture_radii, aperture_radii+22)

        # print('counts', total_counts, background_counts)

        light_curve_total[i] = total_counts
        light_curve_background[i] = background_counts
        # TODO: subtract background within source from source counts, still needs to be discussed

    # get errors
    average_background_count_rate = np.mean(light_curve_background) / time_step
    light_curve_error = poisson_conf_interval(
        light_curve_total,
        interval='frequentist-confidence',
        sigma=1
    )

    return pd.DataFrame({
        'time': time_bins[:-1],
        'counts': light_curve_total,
        'error_low': light_curve_error[0],
        'error_high': light_curve_error[1]
    })


def plot_light_curve(obsid: str, fxt_ra: float, fxt_dec: float, fxt_theta: float, verbose: int = 0):
    """
    ## Plots light curve of a source.

    ### Args:
        obsid `str`: Observation ID.
        fxt_ra `float`: Right ascension of the source.
        fxt_dec `float`: Declination of the source.
        fxt_theta `float`: Off-axis angle of the source.
        verbose `int` (optional): Defaults to `0`. Level of verbosity.
    """
    lc = gen_light_curve(
        obsid, fxt_ra, fxt_dec, fxt_theta, verbose=verbose)
    plt.errorbar(
        (lc['time'] - lc['time'][0]) / 1000,
        lc['counts'],
        yerr=[lc['error_low'], lc['error_high']],
        fmt='o'
    )
    plt.xlabel('Time (ks)')
    plt.ylabel('Counts')
    plt.title(f'{obsid} - {fxt_ra}, {fxt_dec}')
    plt.show()


def get_candidate_numbers(from_date: str = '', to_date: str = '', window: int = 20) -> pd.DataFrame:
    """
    ## Get the number of observations, analysed observations, detections and candidates that match no criteria.

    ### Args:
        from_date `str` (optional): Defaults to `''`. Format: 'YYYY-MM-DD'. Date to start range. If empty, no lower bound. Inclusive.
        to_date `str` (optional): Defaults to `''`. Format: 'YYYY-MM-DD'. Date to end range. If empty, no upper bound. Exclusive.
        window `int` (optional): Defaults to `20`. Window size candidates were detected in.

    ### Returns:
        `pd.DataFrame`: Table with the number of observations, analysed observations, detections and candidates that match no criteria.
    """
    filtered = pd.read_csv(f'output/filtered_w{int(window)}.csv',
                           header=0, dtype=str)

    analysed = pd.read_csv(f'output/analysed_w{int(window)}.txt',
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

    if to_date:
        to_date = pd.to_datetime(to_date)

    candidate_numbers = pd.DataFrame(
        columns=['Total', 'Before', 'In', 'After'])

    # observations
    candidate_numbers.at['Observations', 'Total'] = len(obsids)
    if from_date and to_date:
        candidate_numbers.at['Observations', 'In'] = len(
            obsids[(obsids['Public Release Date'] >= from_date) & (obsids['Public Release Date'] < to_date)])
        candidate_numbers.at['Observations', 'Before'] = len(
            obsids[obsids['Public Release Date'] < from_date])
        candidate_numbers.at['Observations', 'After'] = len(
            obsids[obsids['Public Release Date'] >= to_date])
    elif from_date:
        candidate_numbers.at['Observations', 'Before'] = len(
            obsids[obsids['Public Release Date'] < from_date])
        candidate_numbers.at['Observations', 'After'] = len(
            obsids[obsids['Public Release Date'] >= from_date])
    elif to_date:
        candidate_numbers.at['Observations', 'Before'] = len(
            obsids[obsids['Public Release Date'] < to_date])
        candidate_numbers.at['Observations', 'After'] = len(
            obsids[obsids['Public Release Date'] >= to_date])

    # analysed
    obsids = obsids[obsids['Obs ID'].isin(analysed['ObsId'])]
    candidate_numbers.at['Analysed', 'Total'] = len(obsids)
    if from_date and to_date:
        candidate_numbers.at['Analysed', 'In'] = len(
            obsids[(obsids['Public Release Date'] >= from_date) & (obsids['Public Release Date'] < to_date)])
        candidate_numbers.at['Analysed', 'Before'] = len(
            obsids[obsids['Public Release Date'] < from_date])
        candidate_numbers.at['Analysed', 'After'] = len(
            obsids[obsids['Public Release Date'] >= to_date])
    elif from_date:
        candidate_numbers.at['Analysed', 'Before'] = len(
            obsids[obsids['Public Release Date'] < from_date])
        candidate_numbers.at['Analysed', 'After'] = len(
            obsids[obsids['Public Release Date'] >= from_date])
    elif to_date:
        candidate_numbers.at['Analysed', 'Before'] = len(
            obsids[obsids['Public Release Date'] < to_date])
        candidate_numbers.at['Analysed', 'After'] = len(
            obsids[obsids['Public Release Date'] >= to_date])

    # detections
    candidate_numbers.at['Detections', 'Total'] = len(filtered)
    if from_date and to_date:
        candidate_numbers.at['Detections', 'In'] = len(
            filtered[filtered['ObsId'].isin(obsids[(obsids['Public Release Date'] >= from_date) & (obsids['Public Release Date'] < to_date)]['Obs ID'])])
        candidate_numbers.at['Detections', 'Before'] = len(
            filtered[filtered['ObsId'].isin(obsids[obsids['Public Release Date'] < from_date]['Obs ID'])])
        candidate_numbers.at['Detections', 'After'] = len(
            filtered[filtered['ObsId'].isin(obsids[obsids['Public Release Date'] >= to_date]['Obs ID'])])
    elif from_date:
        candidate_numbers.at['Detections', 'Before'] = len(
            filtered[filtered['ObsId'].isin(obsids[obsids['Public Release Date'] < from_date]['Obs ID'])])
        candidate_numbers.at['Detections', 'After'] = len(
            filtered[filtered['ObsId'].isin(obsids[obsids['Public Release Date'] >= from_date]['Obs ID'])])
    elif to_date:
        candidate_numbers.at['Detections', 'Before'] = len(
            filtered[filtered['ObsId'].isin(obsids[obsids['Public Release Date'] < to_date]['Obs ID'])])
        candidate_numbers.at['Detections', 'After'] = len(
            filtered[filtered['ObsId'].isin(obsids[obsids['Public Release Date'] >= to_date]['Obs ID'])])

    # candidates no match
    candidate_numbers.at['Candidates no match', 'Total'] = len(
        filtered[(filtered[ALL_FILTERS] == 'no').all(axis=1)])
    if from_date and to_date:
        candidate_numbers.at['Candidates no match', 'In'] = len(
            filtered[(filtered[ALL_FILTERS] == 'no').all(axis=1) & filtered['ObsId'].isin(obsids[(obsids['Public Release Date'] >= from_date) & (obsids['Public Release Date'] < to_date)]['Obs ID'])])
        candidate_numbers.at['Candidates no match', 'Before'] = len(
            filtered[(filtered[ALL_FILTERS] == 'no').all(axis=1) & filtered['ObsId'].isin(obsids[obsids['Public Release Date'] < from_date]['Obs ID'])])
        candidate_numbers.at['Candidates no match', 'After'] = len(
            filtered[(filtered[ALL_FILTERS] == 'no').all(axis=1) & filtered['ObsId'].isin(obsids[obsids['Public Release Date'] >= to_date]['Obs ID'])])
    elif from_date:
        candidate_numbers.at['Candidates no match', 'Before'] = len(
            filtered[(filtered[ALL_FILTERS] == 'no').all(axis=1) & filtered['ObsId'].isin(obsids[obsids['Public Release Date'] < from_date]['Obs ID'])])
        candidate_numbers.at['Candidates no match', 'After'] = len(
            filtered[(filtered[ALL_FILTERS] == 'no').all(axis=1) & filtered['ObsId'].isin(obsids[obsids['Public Release Date'] >= from_date]['Obs ID'])])
    elif to_date:
        candidate_numbers.at['Candidates no match', 'Before'] = len(
            filtered[(filtered[ALL_FILTERS] == 'no').all(axis=1) & filtered['ObsId'].isin(obsids[obsids['Public Release Date'] < to_date]['Obs ID'])])
        candidate_numbers.at['Candidates no match', 'After'] = len(
            filtered[(filtered[ALL_FILTERS] == 'no').all(axis=1) & filtered['ObsId'].isin(obsids[obsids['Public Release Date'] >= to_date]['Obs ID'])])

    return candidate_numbers


def get_criteria_table(
    from_date: str = '',
    to_date: str = '',
    criteria: List[Tuple[str, List[str]]] = CRITERIA,
    window: int = 20,
):
    """
    Get the number of candidates that match each criterion, the number of candidates that are only matched by that criterion, the number of candidates removed after that criterion and the number of candidates remaining after that criterion.

    Args:
        from_date (str, optional): Start date of range. Defaults to ''. Format: 'YYYY-MM-DD'. If empty, no lower bound. Inclusive.
        to_date (str, optional): End date of range. Defaults to ''. Format: 'YYYY-MM-DD'. If empty, no upper bound. Exclusive.
        criteria (List[Tuple[str, List[str]]], optional): List of criteria and their filter functions. Defaults to [ ('Archival X-ray date', [ 'archival_match', 'chandra_match', 'erosita_match', ]), ('Cross-match with stars/Gaia', [ 'gaia_match', ]), ('NED + SIMBAD + VizieR', [ 'ned_match', 'simbad_match', 'vizier_match', ]), ].
        verbose (int, optional): Verbosity. Defaults to 0.
        window (int, optional): Window size candidates were detected in. Defaults to 20.

    Returns:
        pd.DataFrame: Criteria table as in the paper.
    """
    filtered = pd.read_csv(
        f'output/filtered_w{int(window)}.csv', header=0, dtype=str)

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

    all_criteria = []
    for _, columns in criteria:
        all_criteria += columns

    criteria_table = pd.DataFrame(
        columns=['Matched', 'Unique Matched', 'Removed', 'Remaining'])

    for i, (criterion, columns) in enumerate(criteria):
        # candidates that are matched by this criterion
        matched = filtered[
            (filtered[columns] == 'yes').any(axis=1)
        ]
        criteria_table.at[criterion, 'Matched'] = len(matched)

        # candidates that are only matched by this criterion and not any other
        unique_matched = matched[
            (matched[all_criteria] == 'yes').sum(axis=1) == 1
        ]
        criteria_table.at[criterion, 'Unique Matched'] = len(unique_matched)

        # candidates that are removed by this criterion but not any before
        removed = filtered
        for _, columns_before in criteria[:i]:
            removed = removed[
                (removed[columns_before] == 'no').all(axis=1)
            ]
        removed = removed[
            (removed[columns] == 'yes').any(axis=1)
        ]
        criteria_table.at[criterion, 'Removed'] = len(removed)

        # candidates remaining after this stage, that have no matches by previous criteria and current criterion
        remaining = filtered
        for _, columns_before in criteria[:i]:
            remaining = remaining[
                (remaining[columns_before] == 'no').all(axis=1)
            ]
        remaining = remaining[
            (remaining[columns] == 'no').all(axis=1)
        ]
        criteria_table.at[criterion, 'Remaining'] = len(remaining)

    return criteria_table


def get_detections(obsid: str, ra: float = None, dec: float = None, filtered: bool = False):
    """
    Get the detections for an observation within a radius around given coordinates.

    Args:
        obsid (str): Observation ID.
        ra (float, optional): RA of the source. Defaults to None.
        dec (float, optional): Dec of the source. Defaults to None.
        filtered (bool, optional): Whether to use the filtered detections. Defaults to False.

    Returns:
        pd.DataFrame: Detections for the observation.
    """
    if filtered:
        detections = pd.read_csv(
            'output/filtered_w20.csv', header=0, dtype=str)
    else:
        detections = pd.read_csv('output/detections_w20.txt',
                                 header=0, dtype=str, sep=' ')

    detections = detections[detections['ObsId'] == obsid]

    if ra and dec:
        detections[['RA', 'DEC']] = detections[['RA', 'DEC']].astype(float)
        distances_squared = abs(
            detections['RA'] - ra) ** 2 + abs(detections['DEC'] - dec) ** 2
        smallest_distance = distances_squared.min()
        detections = detections[distances_squared == smallest_distance]

    return detections


def get_no_match_fxts(window: int = 20, from_date: str = '', to_date: str = ''):
    """
    ## Get the FXTs that have no matches.

    ### Args:
        window `int` (optional): Defaults to `20`. Window size candidates were detected in.

    ### Returns:
        `pd.DataFrame`: FXTs that have no matches.
    """
    filtered = pd.read_csv(f'output/filtered_w{int(window)}.csv',
                           header=0, dtype=str)
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

    return filtered[(filtered[ALL_FILTERS] == 'no').all(axis=1)][['ObsId', 'RA', 'DEC', 'THETA', 'POS_ERR']]


if __name__ == '__main__':
    get_candidate_numbers()
    # get_obsid_dates()

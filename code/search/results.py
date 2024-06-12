import time
import pandas as pd
from astropy.table import Table
from auxiliary.search_algorithm import get_wcs_event, off_axis, get_chandra_eef, get_counts_from_event
import glob
import numpy as np
from astropy.stats import poisson_conf_interval
from typing import Dict, List, Tuple

CRITERIA = [
    ('Archival X-ray date', [
        'archival_match',
        'chandra_match',
        'erosita_match',
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


def gen_light_curve(obsid: int, fxt_ra: float, fxt_dec: float, fxt_theta: float, fxt_pos_err: float) -> pd.DataFrame:
    t1 = time.perf_counter()

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

    t2 = time.perf_counter()

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

    t3 = time.perf_counter()

    time_step = 1000
    time_bins = np.arange(event_data_raw['time'].min(
    ), event_data_raw['time'].max(), time_step)
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

    # subtract background from total counts
    light_curve = light_curve_total - light_curve_background

    # get errors
    average_background_count_rate = np.mean(light_curve_background) / time_step
    light_curve_error = poisson_conf_interval(
        light_curve_total, interval='kraft-burrows-nousek', background=average_background_count_rate, confidence_level=0.68)

    t4 = time.perf_counter()

    print(f't2-t1: {t2-t1:.2f}')
    print(f't3-t2: {t3-t2:.2f}')
    print(f't4-t3: {t4-t3:.2f}')

    return pd.DataFrame({
        'time': time_bins[:-1],
        'counts': light_curve,
        'error_low': light_curve_error[0],
        'error_high': light_curve_error[1]
    })


def get_candidate_numbers(from_date: str = '', to_date: str = '') -> pd.DataFrame:
    """
    Get the number of observations, detections and candidates within a date range.

    Args:
        from_date (str, optional): Start date of the range. Defaults to ''. Format: 'YYYY-MM-DD'. If empty, no lower bound. Inclusive.
        to_date (str, optional): End date of the range. Defaults to ''. Format: 'YYYY-MM-DD'. If empty, no upper bound. Exclusive.

    Returns:
        pd.DataFrame: DataFrame with the number of observations, detections and candidates within the date range.
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
):
    """
    Get the number of candidates that match each criterion, the number of candidates that are only matched by that criterion and the number of candidates remaining after that criterion.
    As in the paper.

    Args:
        from_date (str, optional): Start date of range. Defaults to ''. Format: 'YYYY-MM-DD'. If empty, no lower bound. Inclusive.
        to_date (str, optional): End date of range. Defaults to ''. Format: 'YYYY-MM-DD'. If empty, no upper bound. Exclusive.
        criteria (List[Tuple[str, List[str]]], optional): List of criteria and their filter functions. Defaults to [ ('Archival X-ray date', [ 'archival_match', 'chandra_match', 'erosita_match', ]), ('Cross-match with stars/Gaia', [ 'gaia_match', ]), ('NED + SIMBAD + VizieR', [ 'ned_match', 'simbad_match', 'vizier_match', ]), ].
        verbose (int, optional): Verbosity. Defaults to 0.

    Returns:
        pd.DataFrame: Criteria table as in the paper.
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
            (removed[columns] == 'yes').all(axis=1)
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


def get_obsid_dates():
    obsids_after1 = pd.read_csv('obsid_lists/obsids_b+10_220401+.csv',
                                header=0, dtype=str, sep=',', usecols=['Obs ID', 'Public Release Date'])
    obsids_after2 = pd.read_csv('obsid_lists/obsids_b-10_220401+.csv',
                                header=0, dtype=str, sep=',', usecols=['Obs ID', 'Public Release Date'])
    obsids_after = pd.concat([obsids_after1, obsids_after2], ignore_index=True)

    obsids_before1 = pd.read_csv('obsid_lists/obsids_b+10_220401-.csv',
                                 header=0, dtype=str, sep=',', usecols=['Obs ID', 'Public Release Date'])
    obsids_before2 = pd.read_csv('obsid_lists/obsids_b-10_220401-.csv',
                                 header=0, dtype=str, sep=',', usecols=['Obs ID', 'Public Release Date'])
    obsids_before = pd.concat(
        [obsids_before1, obsids_before2], ignore_index=True)

    # calculating newest obsid date
    obsids_after['Public Release Date'] = pd.to_datetime(
        obsids_after['Public Release Date'])
    print(f'newest obsid date: {obsids_after["Public Release Date"].max()}')
    print(f'oldest obsid date: {obsids_after["Public Release Date"].min()}')

    # calculating oldest obsid date
    obsids_before['Public Release Date'] = pd.to_datetime(
        obsids_before['Public Release Date'])
    print(f'newest obsid date: {obsids_before["Public Release Date"].max()}')
    print(f'oldest obsid date: {obsids_before["Public Release Date"].min()}')


if __name__ == '__main__':
    get_candidate_numbers()
    # get_obsid_dates()

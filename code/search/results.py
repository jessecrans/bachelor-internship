import time
import pandas as pd
from astropy.table import Table
from auxiliary.search_algorithm import get_wcs_event, off_axis, get_chandra_eef, get_counts_from_event
from search import download_data, process_data, read_obsids
import glob
import numpy as np
from astropy.stats import poisson_conf_interval
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.stats import poisson
import subprocess
import os
from astropy.time import Time as astrotime

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
NO_ARCHIVAL = [
    filter for _, filters in CRITERIA for filter in filters if filter not in [
        'archival_match', 'chandra_match', 'erosita_match', 'xray-binaries_match']
]
FILENAMES = [  # List of obsids from Chandra
    'obsid_lists/obsids_b+10_220401+.csv',
    'obsid_lists/obsids_b-10_220401+.csv',
    'obsid_lists/obsids_b+10_220401-.csv',
    'obsid_lists/obsids_b-10_220401-.csv',
]

paper_I_fxts = [
    [803, 186.38125, 13.06607, 'XRT 000519'],
    [2025, 167.86792, 55.67253, 'XRT 010908'],
    [8490, 201.24329, -43.04060, 'XRT 070530'],
    [9546, 211.25113, 53.65706, 'XRT 071203'],
    [9548, 170.07296, 12.97189, 'XRT 130822'],
    [14904, 345.49250, 15.94871, 'XRT 130822'],
    [4062, 76.77817, -31.86980, 'XRT 030511'],
    [5885, 318.12646, -63.49914, 'XRT 041230'],
    [9841, 175.00504, -31.91743, 'XRT 080819'],
    [12264, 90.00450, -52.71501, 'XRT 100831'],
    [12884, 212.12063, -27.05784, 'XRT 110103'],
    [13454, 15.93558, -21.81272, 'XRT 110919'],
    [15113, 45.26725, -77.88095, 'XRT 140327'],
    [16454, 53.16158, -27.85940, 'XRT 141001/CDF-S XT1'],
]
paper_II_fxts = [
    [16093, 233.73496, 23.46849, 'XRT 140507'],
    [16453, 53.07672, -27.87345, 'XRT 150322/CDF-S XT2'],
    [18715, 40.82972, 32.32390, 'XRT 151121'],
    [19310, 36.71489, -1.08317, 'XRT 161125'],
    [20635, 356.26437, -42.64494, 'XRT 170901'],
    [21831, 207.34711, 26.58421, 'XRT 191127'],
    [23103, 50.47516, 41.24704, 'XRT 191223'],
    [24604, 207.23523, 26.66230, 'XRT 210423'],
]

DATA_PATH = '/data/jcrans/fxrt-data/obsids/'


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


def get_events_from_obsid(obsid: str) -> tuple[pd.DataFrame, str]:
    """
    ## Get event dataframe for the given obsid.


    ### Args:
        obsid `str`: ObsId to get the events from.

    ### Returns:
        `tuple[pd.DataFrame, str]`: A tuple of the event data and the event file.
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
        return -1

    with fits.open(event_file) as hdul:
        information = hdul[1].header
        obs = information['OBS_ID']
        t_start = information['TSTART']
        t_stop = information['TSTOP']

    # Read the event file
    event_data = Table.read(event_file, hdu=1)
    colnames = [col for col in event_data.colnames if len(
        event_data[col].shape) <= 1]
    event_data = event_data[colnames].to_pandas()

    event_data = event_data[
        (event_data['energy'] >= 5e2) &
        (event_data['energy'] <= 7e3)
    ]

    return event_data, event_file


def get_events_x_y(
    event_data: pd.DataFrame,
    source_x: float,
    source_y: float,
    source_radius: int,
    background_radius: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    ## Get the events from the event table that are within the source and background radii.


    ### Args:
        event_data `dict`: The raw event 2 table.
        source_x `float`: The x coordinate of the source in the observation.
        source_y `float`: The y coordinate of the source in the observation.
        source_radius `int`: Source is extracted within a circle Rin=Rsrc (px).
        background_radius `int`: Background is an annulus, Rout=Rbkg (px).

    ### Returns:
        `tuple[int, int]`: A tuple of total counts and background counts.
    """
    events_in_source = event_data[
        (event_data['x'] - source_x)**2 +
        (event_data['y'] - source_y)**2 <= source_radius**2
    ]
    events_in_background = event_data[
        (event_data['x'] - source_x)**2 +
        (event_data['y'] - source_y)**2 <= background_radius**2
    ]

    return events_in_source, events_in_background


def get_events_ra_dec(obsid: str, ra: float, dec: float, theta: float) -> pd.DataFrame:
    """
    ## Get events from a given detection.


    ### Args:
        detection `pd.Series`: Detection to get all events from.

    ### Returns:
        `pd.DataFrame`: Events of the given detection.
    """
    event_data = get_events_from_obsid(obsid)

    try:
        files = glob.glob(
            f'/data/jcrans/fxrt-data/obsids/{obsid}/s3_expmap_src.fits', recursive=True)
        src_file = files[0]
        files = glob.glob(
            f'/data/jcrans/fxrt-data/obsids/{obsid}/*evt2.fits', recursive=True)
        event_file = files[0]
    except IndexError:
        print(f'No files found for {obsid}')
        return -1

    # Read the wcs of the event file
    event_wcs = get_wcs_event(event_file)

    # convert ra, dec to x, y
    x, y = event_wcs.all_world2pix(ra, dec, 1)

    # Get R90 size
    r90_size = get_chandra_eef(
        np.array([theta]), R0=1.07, R10=9.65, alpha=2.22)[0]

    # Convert to pixel scale
    acis_pix_size = 0.492
    r90_size /= acis_pix_size

    # Get the aperture size
    aperture_radius = r90_size * 1.5

    events_in_source, events_in_background = get_events_x_y(
        event_data, x, y, aperture_radius, aperture_radius+22
    )

    return events_in_source, events_in_background


def get_counts_ra_dec(event_data: pd.DataFrame, event_file: str, ra: float, dec: float) -> tuple[float, float]:
    """
    ## Get the counts for a given source.


    ### Args:
        event_data `pd.DataFrame`: Event data.
        event_file `str`: Event file.
        ra `float`: Right ascension of the source.
        dec `float`: Declination of the source.

    ### Returns:
        `tuple[float, float]`: Total counts and background counts.
    """
    # Read the wcs of the event file
    event_wcs = get_wcs_event(event_file)

    # convert ra, dec to x, y
    x, y = event_wcs.all_world2pix(ra, dec, 1)

    # get theta
    theta = off_axis(event_file, ra, dec)

    # Get R90 size
    r90_size = get_chandra_eef(
        np.array([theta]), R0=1.07, R10=9.65, alpha=2.22)[0]

    # Convert to pixel scale
    acis_pix_size = 0.492
    r90_size /= acis_pix_size

    # Get the aperture size
    aperture_radius = r90_size * 1.5

    source_counts, background_counts = get_counts_from_event(
        event_data, x, y, aperture_radius, aperture_radius+22)

    return source_counts, background_counts


def gen_light_curve(obsid: str, ra: float, dec: float, theta: float, bin: int = 1000, verbose: int = 0) -> pd.DataFrame:
    """
    ## Generate a light curve for a given source.

    ### Args:
        obsid `str`: ObsId where source is in.
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
    x, y = event_wcs.all_world2pix(ra, dec, 1)

    # Get R90 size
    r90_size = get_chandra_eef(
        np.array([theta]), R0=1.07, R10=9.65, alpha=2.22)[0]

    # Convert to pixel scale
    acis_pix_size = 0.492
    r90_size /= acis_pix_size

    # Get the aperture size
    aperture_radii = r90_size * 1.5

    event_data = event_data_raw[np.where(
        (event_data_raw['energy'] >= 5e2) &
        (event_data_raw['energy'] <= 7e3)
    )]

    _, background_counts = get_counts_from_event(
        event_data, x, y, aperture_radii, aperture_radii+22
    )
    background_rate = background_counts / (t_stop - t_start)

    time_step = bin
    time_bins = np.arange(t_start, t_stop, time_step)
    light_curve_total = np.zeros(len(time_bins)-1, dtype=float)

    # counts per time bin
    for i in range(len(time_bins)-1):
        event_data_bin = event_data[np.where(
            (event_data['time'] >= time_bins[i]) &
            (event_data['time'] < time_bins[i+1])
        )]

        total_counts_i, _ = get_counts_from_event(
            event_data_bin, x, y, aperture_radii, aperture_radii+22)

        light_curve_total[i] = total_counts_i - background_rate * time_step

    # get errors
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


def plot_light_curve(obsid: str, ra: float, dec: float, theta: float, bin: int = 1000, verbose: int = 0):
    """
    ## Plots light curve of a source.

    ### Args:
        obsid `str`: ObsId where the source is in.
        fxt_ra `float`: Right ascension of the source.
        fxt_dec `float`: Declination of the source.
        fxt_theta `float`: Off-axis angle of the source.
        verbose `int` (optional): Defaults to `0`. Level of verbosity.
    """
    lc = gen_light_curve(
        obsid, ra, dec, theta, bin, verbose=verbose)
    plt.errorbar(
        (lc['time'] - lc['time'][0]) / 1000,
        lc['counts'],
        yerr=[lc['counts'] - lc['error_low'],
              lc['error_high'] - lc['counts']],
        fmt='o'
    )
    plt.xlabel('Time (ks)')
    plt.ylabel('Counts')
    plt.title(f'{obsid} - {ra}, {dec}')
    plt.savefig(f'plots/light_curves/{obsid}-{ra:.2f}-{dec:.2f}.png')
    plt.close()
    # plt.show()


def get_t_90(obsid: str, ra: float, dec: float, theta: float, verbose: int = 0) -> tuple[float, float, float]:
    """
    ## Get the T90 of a source.

    ### Args:
        obsid `str`: ObsId where source is in.
        ra `float`: Right ascension of the source.
        dec `float`: Declination of the source.
        theta `float`: Off-axis angle of the source.
        verbose `int` (optional): Defaults to `0`. Level of verbosity.

    ### Returns:
        `tuple[float, float, float]`: T90, lower uncertainty, upper uncertainty.
    """
    # get files
    try:
        files = glob.glob(
            f'/data/jcrans/fxrt-data/obsids/{obsid}/*evt2.fits', recursive=True)
        event_file = files[0]
    except IndexError:
        print(f'No files found for {obsid}')
        return -1

    with fits.open(event_file) as hdul:
        information = hdul[1].header
        obs = information['OBS_ID']
        t_start = information['TSTART']
        t_stop = information['TSTOP']

    event_data, event_file = get_events_from_obsid(obsid)

    # Read the wcs of the event file
    event_wcs = get_wcs_event(event_file)
    # convert ra, dec to x, y
    x, y = event_wcs.all_world2pix(ra, dec, 1)

    # Get R90 size
    r90_size = get_chandra_eef(
        np.array([theta]), R0=1.07, R10=9.65, alpha=2.22)[0]

    # Convert to pixel scale
    acis_pix_size = 0.492
    r90_size /= acis_pix_size

    # Get the aperture size
    aperture_radius = r90_size * 1.5

    # getting events and background rate
    events_in_source, _ = get_events_x_y(
        event_data, x, y, aperture_radius, aperture_radius+22
    )
    events_in_source = events_in_source.sort_values('time')

    _, background_counts = get_counts_from_event(
        event_data, x, y, aperture_radius, aperture_radius+22
    )

    background_rate = background_counts / (t_stop - t_start)

    # calculating fxt start and ending time based on background level
    bin = 100  # sec
    counts_bins = np.histogram(
        events_in_source['time'], bins=np.arange(t_start, t_stop+1, bin))[0]

    fxt_start_index = np.where(counts_bins > 5 * background_rate * bin)[0][0]
    fxt_end_index = np.where(counts_bins > 5 * background_rate * bin)[0][-1]

    fxt_start_time = t_start + fxt_start_index * bin
    fxt_end_time = t_start + fxt_end_index * bin

    fxt_events = events_in_source[
        (events_in_source['time'] >= fxt_start_time) &
        (events_in_source['time'] <= fxt_end_time)
    ]

    # calculating t90
    photon_arrival_times = fxt_events['time'].to_numpy(dtype=float)

    total_counts = len(photon_arrival_times)

    counts_5 = 0.05 * total_counts
    counts_95 = 0.95 * total_counts

    index_5 = int(counts_5)
    index_95 = int(counts_95)

    t_5 = photon_arrival_times[index_5]
    t_95 = photon_arrival_times[index_95]

    # Q: Are these the same thing?
    # errors
    lower_5, upper_5 = poisson_conf_interval(
        counts_5, 'frequentist-confidence', 1
    )

    lower_95, upper_95 = poisson_conf_interval(
        counts_95, 'frequentist-confidence', 1
    )

    # lower_5, upper_5 = poisson.ppf(
    #     (1 - 0.68) / 2, counts_5), poisson.ppf((1 - (1 - 0.68)) / 2, counts_5)

    # lower_95, upper_95 = poisson.ppf(
    #     (1 - 0.68) / 2, counts_95), poisson.ppf((1 - (1 - 0.68)) / 2, counts_95)

    # Calculate the uncertainties in time by mapping the photon counts back to time
    delta_t5_lower = photon_arrival_times[int(
        np.clip(lower_5, 0, total_counts - 1))]
    delta_t5_upper = photon_arrival_times[int(
        np.clip(upper_5, 0, total_counts - 1))]
    delta_t95_lower = photon_arrival_times[int(
        np.clip(lower_95, 0, total_counts - 1))]
    delta_t95_upper = photon_arrival_times[int(
        np.clip(upper_95, 0, total_counts - 1))]

    # Calculate the uncertainties for the T90 duration
    t_90_lower = np.sqrt(
        (t_95 - delta_t95_lower)**2 + (t_5 - delta_t5_lower)**2)
    t_90_upper = np.sqrt(
        (delta_t95_upper - t_95)**2 + (delta_t5_upper - t_5)**2)

    if verbose > 0:
        print(f"total_counts: {total_counts}")
        print(f"background rate: {background_rate * bin} c/{bin}s")

        # print(f"counts_bins: {counts_bins}")

        print(
            f"fxt_start: {fxt_start_time - t_start}, fxt_end: {fxt_end_time - t_start}")

        print(
            f"100%: {total_counts}, 5%: {0.05 * total_counts}, 95%: {0.95 * total_counts}")

        print(f"t_5: {t_5 - t_start}, t_95: {t_95 - t_start}")
        print(f"t_90: {t_95 - t_5}")

        print(f"lower_5: {lower_5}, upper_5: {upper_5}")
        print(f"lower_95: {lower_95}, upper_95: {upper_95}")

        print(f"t_90_lower: {t_90_lower}, t_90_upper: {t_90_upper}")

    return t_95 - t_5, t_90_lower, t_90_upper


def get_t_90_str(obsid: str, ra: float, dec: float, theta: float, verbose: int = 0) -> str:
    """
    ## Get the T90 of a source as a string.

    ### Args:
        obsid `str`: ObsId where source is in.
        ra `float`: Right ascension of the source.
        dec `float`: Declination of the source.
        theta `float`: Off-axis angle of the source.
        verbose `int` (optional): Defaults to `0`. Level of verbosity.

    ### Returns:
        `str`: T90 of the source as a string.
    """
    t_90, t_90_lower, t_90_upper = get_t_90(obsid, ra, dec, theta, verbose)

    return '$' + f'{t_90/1000:.1f}^' + '{' + f'{t_90_upper/1000:.1f}' + '}_{' + f'{t_90_lower/1000:.1f}' + '}$'


def get_HR(obsid: str, ra: float, dec: float, theta: float, verbose: int = 0) -> tuple[float, float, float]:
    """
    ## Get the hardness ratio of a source.

    ### Args:
        obsid `str`: ObsId where source is in.
        ra `float`: Right ascension of the source.
        dec `float`: Declination of the source.
        theta `float`: Off-axis angle of the source.
        verbose `int` (optional): Defaults to `0`. Level of verbosity.

    ### Returns:
        `tuple[float, float, float]`: Hardness ratio, lower uncertainty, upper uncertainty.
    """
    # get files
    try:
        files = glob.glob(
            f'/data/jcrans/fxrt-data/obsids/{obsid}/*evt2.fits', recursive=True)
        event_file = files[0]
    except IndexError:
        print(f'No files found for {obsid}')
        return -1

    event_data = get_events_from_obsid(obsid)

    # Read the wcs of the event file
    event_wcs = get_wcs_event(event_file)
    # convert ra, dec to x, y
    x, y = event_wcs.all_world2pix(ra, dec, 1)

    # Get R90 size
    r90_size = get_chandra_eef(
        np.array([theta]), R0=1.07, R10=9.65, alpha=2.22)[0]

    # Convert to pixel scale
    acis_pix_size = 0.492
    r90_size /= acis_pix_size

    # Get the aperture size
    aperture_radius = r90_size * 1.5

    events_in_source, _ = get_events_x_y(
        event_data, x, y, aperture_radius, aperture_radius+22)

    # Get the hardness ratio
    events_in_soft = events_in_source[
        (events_in_source['energy'] >= 5e2) &
        (events_in_source['energy'] <= 2e3)
    ]
    events_in_hard = events_in_source[
        (events_in_source['energy'] >= 2e3) &
        (events_in_source['energy'] <= 7e3)
    ]

    total_counts_soft = len(events_in_soft)
    total_counts_hard = len(events_in_hard)

    HR = (total_counts_hard - total_counts_soft) / \
        (total_counts_hard + total_counts_soft)

    return HR, 0, 0


def get_flux(detection: pd.Series) -> float:
    """
    ## Get the flux of a detection.

    ### Args:
        detection `pd.Series`: The detection to get the flux for.

    ### Returns:
        `float`: Flux of the detection.
    """
    # get files
    try:
        files = glob.glob(
            f"/data/jcrans/fxrt-data/obsids/{detection['ObsId']}/*evt2.fits", recursive=True)
        event_file = files[0]
    except IndexError:
        print(f"No files found for {detection['ObsId']}")
        return -1

    event_data = get_events_from_obsid(detection['ObsId'])

    # Read the wcs of the event file
    event_wcs = get_wcs_event(event_file)
    # convert ra, dec to x, y
    x, y = event_wcs.all_world2pix(
        float(detection['RA']), float(detection['DEC']), 1)

    # Get R90 size
    r90_size = get_chandra_eef(
        np.array([float(detection['THETA'])]), R0=1.07, R10=9.65, alpha=2.22)[0]

    # Convert to pixel scale
    acis_pix_size = 0.492
    r90_size /= acis_pix_size

    # Get the aperture size
    aperture_radius = r90_size * 1.5

    events_in_source, _ = get_events_x_y(
        event_data, x, y, aperture_radius, aperture_radius+22)

    # Get the flux
    flux = 0
    for _, event in events_in_source.iterrows():
        flux += event['energy']

    return flux


def get_fxt_id(detection: pd.Series) -> str:
    if detection['ObsId'] in [str(fxt[0]) for fxt in paper_I_fxts]:
        distances_squared = [
            (fxt[1] - float(detection['RA']))**2 + (fxt[2] - float(detection['DEC']))**2 for fxt in paper_I_fxts
        ]
        index, smallest_distance = min(
            enumerate(distances_squared), key=lambda x: x[1])
        # 3 sigma limit
        if smallest_distance <= 3 * float(detection['POS_ERR']):
            return paper_I_fxts[index][3]
    elif detection['ObsId'] in [str(fxt[0]) for fxt in paper_II_fxts]:
        distances_squared = [
            (fxt[1] - float(detection['RA']))**2 + (fxt[2] - float(detection['DEC']))**2 for fxt in paper_II_fxts
        ]
        index, smallest_distance = min(
            enumerate(distances_squared), key=lambda x: x[1])
        if smallest_distance <= 3 * float(detection['POS_ERR']):
            return paper_II_fxts[index][3]

    obsids = read_obsids(FILENAMES, ['Obs ID', 'Exposure', 'Start Date'])

    start_date = obsids[obsids['Obs ID'] ==
                        detection['ObsId']]['Start Date'].values[0]
    start_date = pd.to_datetime(start_date)
    exposure = float(
        obsids[obsids['Obs ID'] == detection['ObsId']]['Exposure'].values[0])
    end_date = start_date + pd.Timedelta(seconds=exposure*1000)

    return f"XRTC {end_date.strftime('%y%m%d')}"


def find_obsids_matching_detection(detection: pd.Series) -> list[str]:
    """
    ## Find the obsids that have observed the area of the detection
    DEPRECATED: replace with ```get_obsids_by_coords```


    ### Args:
        detection `pd.Series`: The detection to find the obsids for.

    ### Returns:
        `list[str]`: List of obsids that have observed the area of the detection.
    """

    command = f'find_chandra_obsid {detection["RA"]} {detection["DEC"]} instrument=acis grating=none detail=obsid'
    proc = subprocess.run(command, shell=True, stdout=subprocess.PIPE)

    # format: # obsid\n4025\n14904\n
    obsids = proc.stdout.decode().split('\n')[1:-1]

    return obsids


def get_obsids_by_coords(ra: float, dec: float) -> list[str]:
    """
    ## Find the obsids that have observed the given coordinates.


    ### Args:
        ra `float`: Right ascension of the detection.
        dec `float`: Declination of the detection.

    ### Returns:
        `list[str]`: List of obsids that have observed the given coordinates.
    """

    command = f'find_chandra_obsid {ra} {dec} instrument=acis grating=none detail=obsid'
    proc = subprocess.run(command, shell=True, stdout=subprocess.PIPE)

    # format: # obsid\n4025\n14904\n
    obsids = proc.stdout.decode().split('\n')[1:-1]

    return obsids


def get_long_light_curve(detection: pd.Series, verbose: int = 0) -> pd.DataFrame:
    obsids = find_obsids_matching_detection(detection)

    light_curve = []

    for obsid in obsids:
        downloaded_obsids = os.listdir(DATA_PATH)
        if obsid not in downloaded_obsids:
            download_data(obsid, verbose, DATA_PATH)
            if not process_data(obsid, verbose, DATA_PATH):
                # message printed in process_data
                continue
        try:
            event_data, event_file = get_events_from_obsid(obsid)
        except Exception as e:
            print(f'{obsid} - Error: {e}')
            continue
        source_counts, background_counts = get_counts_ra_dec(
            event_data, event_file, float(
                detection['RA']), float(detection['DEC']))
        corrected_counts = np.max([0.0, source_counts - background_counts])

        lower, upper = poisson_conf_interval(
            corrected_counts, 'frequentist-confidence', 1
        )

        with fits.open(event_file) as hdul:
            information = hdul[1].header
            obs = information['OBS_ID']
            mjd = information['MJD-OBS']

        light_curve.append({
            'ObsId': obsid,
            'Time': mjd,
            'Counts': source_counts,
            'Background': background_counts,
            'Corrected Counts': corrected_counts,
            'Lower Error': corrected_counts - lower,
            'Upper Error': upper - corrected_counts
        })

    if verbose > 0:
        print(f"obsids: {obsids}")

    light_curve = pd.DataFrame(light_curve)
    return light_curve


def plot_light_curve_multifig(detections: pd.DataFrame, filename: str = 'multifig') -> None:
    """
    ## Plot light curves for multiple sources.

    ### Args:
        detections `pd.DataFrame`: Detections to plot light curves for.
    """
    fig, axs = plt.subplots(
        len(detections), 2, figsize=(6, 2 * len(detections)))

    min_date, max_date = get_min_max_dates()
    min_date, max_date = astrotime(min_date).mjd, astrotime(max_date).mjd

    for i, (index, detection) in enumerate(detections.iterrows()):
        obsids = read_obsids(FILENAMES, ['Obs ID', 'Exposure', 'Start Date'])
        obsid_date = obsids[obsids['Obs ID'] ==
                            detection['ObsId']]['Start Date'].values[0]
        obsid_date = astrotime(obsid_date).mjd

        fxt_id = get_fxt_id(detection)

        long_light_curve = get_long_light_curve(detection)
        light_curve = gen_light_curve(detection['ObsId'], float(
            detection['RA']), float(detection['DEC']), float(detection['THETA']))

        if len(detections) == 1:
            current_axs = axs
        else:
            current_axs = axs[i]

        current_axs[0].errorbar(
            (light_curve['time'] - light_curve['time'][0]) / 1000,
            light_curve['counts'],
            yerr=[light_curve['counts'] - light_curve['error_low'],
                  light_curve['error_high'] - light_curve['counts']],
            fmt='o'
        )
        current_axs[0].set_title(fxt_id)
        current_axs[0].set_ylabel('Counts')
        current_axs[0].tick_params(direction='in')

        current_axs[1].vlines(obsid_date, 0, 1, colors='red', linestyle='--',
                              alpha=0.5, transform=current_axs[1].get_xaxis_transform())
        current_axs[1].hlines(0, min_date, max_date,
                              colors='red', linestyle='--', alpha=0.5)
        current_axs[1].errorbar(long_light_curve['Time'], long_light_curve['Corrected Counts'], yerr=[
            long_light_curve['Lower Error'], long_light_curve['Upper Error']], fmt='o')
        current_axs[1].set_title(fxt_id)
        current_axs[1].set_xlim(min_date, max_date)
        current_axs[1].tick_params(direction='in')

        if i == len(detections) - 1:
            current_axs[0].set_xlabel('Time (ks)')
            current_axs[1].set_xlabel('MJD (days)')

    # first column should get label counts on y
    # last row should get labels of time and mjd on x

    # current_axs[0].set_xlabel('Time (ks)')
    # current_axs[0].set_ylabel('Counts')
    # current_axs[1].set_xlabel('MJD (days)')
    # current_axs[1].set_ylabel('Counts')
    # for ax in axs.flat:
    #     ax.set(xlabel='Exposure time (ks)', ylabel='Detection probability')
    #     ax.label_outer()
    # for ax in axs.flat:
    #     ax.label_outer()

    fig.tight_layout()
    plt.savefig(f'plots/multifigs/{filename}.png')
    plt.close()


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
    detected = pd.read_csv(f'output/detections_w{int(window)}_forward.txt',
                           header=0, dtype=str, sep=' ')

    filtered = pd.read_csv(f'output/filtered_w{int(window)}_forward.csv',
                           header=0, dtype=str)

    analysed = pd.read_csv(f'output/analysed_w{int(window)}_forward.txt',
                           header=0, dtype=str, sep=' ')

    obsids = read_obsids(FILENAMES)
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
    candidate_numbers.at['Detections', 'Total'] = len(detected)
    if from_date and to_date:
        candidate_numbers.at['Detections', 'In'] = len(
            detected[detected['ObsId'].isin(obsids[(obsids['Public Release Date'] >= from_date) & (obsids['Public Release Date'] < to_date)]['Obs ID'])])
        candidate_numbers.at['Detections', 'Before'] = len(
            detected[detected['ObsId'].isin(obsids[obsids['Public Release Date'] < from_date]['Obs ID'])])
        candidate_numbers.at['Detections', 'After'] = len(
            detected[detected['ObsId'].isin(obsids[obsids['Public Release Date'] >= to_date]['Obs ID'])])
    elif from_date:
        candidate_numbers.at['Detections', 'Before'] = len(
            detected[detected['ObsId'].isin(obsids[obsids['Public Release Date'] < from_date]['Obs ID'])])
        candidate_numbers.at['Detections', 'After'] = len(
            detected[detected['ObsId'].isin(obsids[obsids['Public Release Date'] >= from_date]['Obs ID'])])
    elif to_date:
        candidate_numbers.at['Detections', 'Before'] = len(
            detected[detected['ObsId'].isin(obsids[obsids['Public Release Date'] < to_date]['Obs ID'])])
        candidate_numbers.at['Detections', 'After'] = len(
            detected[detected['ObsId'].isin(obsids[obsids['Public Release Date'] >= to_date]['Obs ID'])])

    # candidates no match
    candidate_numbers.at['Candidates', 'Total'] = len(
        filtered[(filtered[ALL_FILTERS] == 'no').all(axis=1)])
    if from_date and to_date:
        candidate_numbers.at['Candidates', 'In'] = len(
            filtered[(filtered[ALL_FILTERS] == 'no').all(axis=1) & filtered['ObsId'].isin(obsids[(obsids['Public Release Date'] >= from_date) & (obsids['Public Release Date'] < to_date)]['Obs ID'])])
        candidate_numbers.at['Candidates', 'Before'] = len(
            filtered[(filtered[ALL_FILTERS] == 'no').all(axis=1) & filtered['ObsId'].isin(obsids[obsids['Public Release Date'] < from_date]['Obs ID'])])
        candidate_numbers.at['Candidates', 'After'] = len(
            filtered[(filtered[ALL_FILTERS] == 'no').all(axis=1) & filtered['ObsId'].isin(obsids[obsids['Public Release Date'] >= to_date]['Obs ID'])])
    elif from_date:
        candidate_numbers.at['Candidates', 'Before'] = len(
            filtered[(filtered[ALL_FILTERS] == 'no').all(axis=1) & filtered['ObsId'].isin(obsids[obsids['Public Release Date'] < from_date]['Obs ID'])])
        candidate_numbers.at['Candidates', 'After'] = len(
            filtered[(filtered[ALL_FILTERS] == 'no').all(axis=1) & filtered['ObsId'].isin(obsids[obsids['Public Release Date'] >= from_date]['Obs ID'])])
    elif to_date:
        candidate_numbers.at['Candidates', 'Before'] = len(
            filtered[(filtered[ALL_FILTERS] == 'no').all(axis=1) & filtered['ObsId'].isin(obsids[obsids['Public Release Date'] < to_date]['Obs ID'])])
        candidate_numbers.at['Candidates', 'After'] = len(
            filtered[(filtered[ALL_FILTERS] == 'no').all(axis=1) & filtered['ObsId'].isin(obsids[obsids['Public Release Date'] >= to_date]['Obs ID'])])

    return candidate_numbers


def get_criteria_table(from_date: str = '', to_date: str = '', criteria: list[tuple[str, list[str]]] = CRITERIA, window: int = 20) -> pd.DataFrame:
    """
    Get the number of candidates that match each criterion, the number of candidates that are only matched by that criterion, the number of candidates removed after that criterion and the number of candidates remaining after that criterion.

    Args:
        from_date (str, optional): Start date of range. Defaults to ''. Format: 'YYYY-MM-DD'. If empty, no lower bound. Inclusive.
        to_date (str, optional): End date of range. Defaults to ''. Format: 'YYYY-MM-DD'. If empty, no upper bound. Exclusive.
        criteria (List[Tuple[str, List[str]]], optional): List of criteria and their filter functions. Defaults to [ ('Archival X-ray date', [ 'archival_match', 'chandra_match', 'erosita_match', ]), ('Cross-match with stars/Gaia', [ 'gaia_match', ]), ('NED + SIMBAD + VizieR', [ 'ned_match', 'simbad_match', 'vizier_match', ]), ].
        window (int, optional): Window size candidates were detected in. Defaults to 20.

    Returns:
        pd.DataFrame: Criteria table.
    """
    filtered = pd.read_csv(
        f'output/filtered_w{int(window)}.csv', header=0, dtype=str)

    obsids = read_obsids(FILENAMES)
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


def get_detections(obsid: str, ra: float = None, dec: float = None, pos_err: float = None, filtered: bool = False, window: int = 20, exact: bool = False) -> pd.DataFrame:
    """
    ## Get the detections matching the given obsid and coordinates if given.
    DEPRECATED: replace with ```get_detections_by_obsid_and_coords```

    ### Args:
        obsid `str`: Obsid the detections should match.
        ra `float` (optional): Defaults to `None`. Right ascension of the source.
        dec `float` (optional): Defaults to `None`. Declination of the source.
        filtered `bool` (optional): Defaults to `False`. Whether to return the filtered detections.
        window `int` (optional): Defaults to `20`. Window size candidates were detected in.

    ### Returns:
        `pd.DataFrame`: Detections matching the given obsid and coordinates. In the case ra and dec are given only the closest detection is returned if it falls within 3 sigma.
    """
    if filtered:
        detections = pd.read_csv(
            f'output/filtered_w{window}.csv', header=0, dtype=str)
    else:
        detections = pd.read_csv(f'output/detections_w{window}.txt',
                                 header=0, dtype=str, sep=' ')

    detections = detections[detections['ObsId'] == obsid]

    if ra is not None and dec is not None and pos_err is not None:
        if exact:
            detections = detections[(detections['RA'] == str(ra)) & (
                detections['DEC'] == str(dec))]
            return detections

        detections[['RA', 'DEC']] = detections[['RA', 'DEC']].astype(float)
        distances_squared = abs(
            detections['RA'] - ra) ** 2 + abs(detections['DEC'] - dec) ** 2
        smallest_distance = distances_squared.min()
        if smallest_distance > 3 * pos_err:  # 3 sigma limit
            return detections[detections['ObsId'] == '-1']

        detections = detections[distances_squared == smallest_distance]

    return detections


def get_detections_by_obsid_and_coords(obsid: str, ra: float = None, dec: float = None, pos_err: float = None, filtered: bool = False, window: int = 20, exact: bool = False) -> pd.DataFrame:
    """
    ## Get the detections matching the given obsid and coordinates if given.

    ### Args:
        obsid `str`: Obsid the detections should match.
        ra `float` (optional): Defaults to `None`. Right ascension of the source.
        dec `float` (optional): Defaults to `None`. Declination of the source.
        filtered `bool` (optional): Defaults to `False`. Whether to return the filtered detections.
        window `int` (optional): Defaults to `20`. Window size candidates were detected in.

    ### Returns:
        `pd.DataFrame`: Detections matching the given obsid and coordinates. In the case ra and dec are given only the closest detection is returned if it falls within 3 sigma.
    """
    if filtered:
        detections = pd.read_csv(
            f'output/filtered_w{window}.csv', header=0, dtype=str)
    else:
        detections = pd.read_csv(f'output/detections_w{window}.txt',
                                 header=0, dtype=str, sep=' ')

    detections = detections[detections['ObsId'] == obsid]

    if ra is not None and dec is not None and pos_err is not None:
        if exact:
            detections = detections[(detections['RA'] == str(ra)) & (
                detections['DEC'] == str(dec))]
            return detections

        detections[['RA', 'DEC']] = detections[['RA', 'DEC']].astype(float)
        distances_squared = abs(
            detections['RA'] - ra) ** 2 + abs(detections['DEC'] - dec) ** 2
        smallest_distance = distances_squared.min()
        if smallest_distance > 3 * pos_err:  # 3 sigma limit
            return detections[detections['ObsId'] == '-1']

        detections = detections[distances_squared == smallest_distance]

    return detections


def get_no_match_fxts(window: int = 20, from_date: str = '', to_date: str = '') -> pd.DataFrame:
    """
    ## Get the FXTs that have no matches in any filter.
    DEPRECATED: replace with ```get_candidates```

    ### Args:
        window `int` (optional): Defaults to `20`. Window size candidates were detected in.
        from_date `str` (optional): Defaults to `''`. Format: 'YYYY-MM-DD'. Date to start range. If empty, no lower bound. Inclusive.
        to_date `str` (optional): Defaults to `''`. Format: 'YYYY-MM-DD'. Date to end range. If empty, no upper bound. Exclusive.

    ### Returns:
        `pd.DataFrame`: FXTs that have no matches.
    """
    filtered = pd.read_csv(f'output/filtered_w{int(window)}.csv',
                           header=0, dtype=str)

    obsids = read_obsids(FILENAMES)
    obsids['Public Release Date'] = pd.to_datetime(
        obsids['Public Release Date'])

    if from_date:
        from_date = pd.to_datetime(from_date)
        obsids = obsids[obsids['Public Release Date'] >= from_date]

    if to_date:
        to_date = pd.to_datetime(to_date)
        obsids = obsids[obsids['Public Release Date'] < to_date]

    filtered = filtered[filtered['ObsId'].isin(obsids['Obs ID'])]

    return filtered[(filtered[ALL_FILTERS] == 'no').all(axis=1)][['ObsId', 'RA', 'DEC', 'THETA', 'POS_ERR', 'SIGNIFICANCE']]


def get_candidates(window: int = 20, from_date: str = '', to_date: str = '', exclusions: list[float] = []) -> pd.DataFrame:
    """
    ## Get the FXTs that have no matches in any filter.

    ### Args:
        window `int` (optional): Defaults to `20`. Window size candidates were detected in.
        from_date `str` (optional): Defaults to `''`. Format: 'YYYY-MM-DD'. Date to start range. If empty, no lower bound. Inclusive.
        to_date `str` (optional): Defaults to `''`. Format: 'YYYY-MM-DD'. Date to end range. If empty, no upper bound. Exclusive.
        exclusions `list` (optional): Defaults to `[]`. List of window sizes to exclude.

    ### Returns:
        `pd.DataFrame`: FXTs that have no matches.
    """
    filtered = pd.read_csv(f'output/filtered_w{int(window)}.csv',
                           header=0, dtype=str)
    if exclusions:
        filtered_to_exclude = pd.concat([pd.read_csv(f'output/filtered_w{int(exclusion)}.csv',
                                                     header=0, dtype=str) for exclusion in exclusions])

    obsids = read_obsids(FILENAMES)
    obsids['Public Release Date'] = pd.to_datetime(
        obsids['Public Release Date'])

    if from_date:
        from_date = pd.to_datetime(from_date)
        obsids = obsids[obsids['Public Release Date'] >= from_date]

    if to_date:
        to_date = pd.to_datetime(to_date)
        obsids = obsids[obsids['Public Release Date'] < to_date]

    filtered = filtered[filtered['ObsId'].isin(obsids['Obs ID'])]

    if exclusions:
        filtered = filtered[
            ~filtered['RA'].isin(filtered_to_exclude['RA']) |
            ~filtered['DEC'].isin(filtered_to_exclude['DEC'])
        ]

    # filtered = filtered[(filtered[ALL_FILTERS] == 'no').all(axis=1)][
        # ['ObsId', 'RA', 'DEC', 'THETA', 'POS_ERR', 'SIGNIFICANCE']]
    filtered = filtered[(filtered[NO_ARCHIVAL] == 'no').all(axis=1)]

    return filtered


def get_exposure_time(obsid: str) -> float:
    obsids = read_obsids(FILENAMES, ['Obs ID', 'Exposure'])

    return float(obsids[obsids['Obs ID'] == obsid]['Exposure'].values[0])


def get_date(obsid: str) -> str:
    obsids = read_obsids(FILENAMES, ['Obs ID', 'Exposure', 'Start Date'])

    start_date = obsids[obsids['Obs ID'] == obsid]['Start Date'].values[0]
    start_date = pd.to_datetime(start_date)
    exposure = float(obsids[obsids['Obs ID'] == obsid]['Exposure'].values[0])
    end_date = start_date + pd.Timedelta(seconds=exposure*1000)

    if start_date.day == end_date.day:
        return start_date.strftime('%Y-%m-%d')
    else:
        return f'{start_date.strftime("%Y-%m-%d")}/{end_date.strftime("%Y-%m-%d")}'


def get_previous_detection_status(obsid: str, ra: float, dec: float, pos_err: float) -> str:
    if obsid in [str(fxt[0]) for fxt in paper_I_fxts]:
        distances_squared = [
            (fxt[1] - ra)**2 + (fxt[2] - dec)**2 for fxt in paper_I_fxts
        ]
        smallest_distance = min(distances_squared)
        if smallest_distance <= 3 * pos_err:  # 3 sigma limit
            return 'Paper I'
    elif obsid in [str(fxt[0]) for fxt in paper_II_fxts]:
        distances_squared = [
            (fxt[1] - ra)**2 + (fxt[2] - dec)**2 for fxt in paper_II_fxts
        ]
        smallest_distance = min(distances_squared)
        if smallest_distance <= 3 * pos_err:
            return 'Paper II'

    return ''


def get_fxt_table(from_date: str = '', to_date: str = '', window: int = 20, start_index: int = 1) -> pd.DataFrame:
    """
    ## Get the FXT table.

    ### Args:
        from_date `str` (optional): Defaults to `''`. Format: 'YYYY-MM-DD'. Date to start range. If empty, no lower bound. Inclusive.
        to_date `str` (optional): Defaults to `''`. Format: 'YYYY-MM-DD'. Date to end range. If empty, no upper bound. Exclusive.
        window `int` (optional): Defaults to `20`. Window size candidates were detected in.

    ### Returns:
        `pd.DataFrame`: FXT table.
    """
    order = [
        'Id', 'Prev. Det.', 'ObsId', 'Exposure', 'Date', 'T_90', 'RA', 'DEC', 'THETA', 'POS_ERR', 'SIGNIFICANCE'  # , 'HR', 'Flux'
    ]

    column_names = {
        'RA': 'RA (deg)',
        'DEC': 'Dec (deg)',
        'THETA': 'Off. Ang.',
        'SIGNIFICANCE': 'S/N',
        'POS_ERR': 'Pos. Unc.',
        'T_90': '$T_{90}$',
        'Exposure': 'Exp. (ks)',
    }

    no_match = get_no_match_fxts(
        window=window, from_date=from_date, to_date=to_date)

    no_match['Exposure'] = [
        f'${get_exposure_time(obsid):.1f}$' for obsid in no_match['ObsId']]
    no_match['Date'] = [get_date(obsid) for obsid in no_match['ObsId']]
    no_match['Date'] = ['\makecell{' + f'{date.split(" - ")[0]}/' + '\\\\' + f'{date.split(" - ")[1]}' +
                        '}' if ' - ' in date else date for date in no_match['Date']]
    no_match['Id'] = [get_fxt_id(detection)
                      for i, detection in no_match.iterrows()]
    no_match['T_90'] = [
        get_t_90_str(
            detection['ObsId'],
            float(detection['RA']),
            float(detection['DEC']),
            float(detection['THETA'])
        ) for i, detection in no_match.iterrows()
    ]
    # no_match['HR'] = [
    #     f'${float(hr):.1f}$' for hr in [get_HR(
    #         detection['ObsId'],
    #         float(detection['RA']),
    #         float(detection['DEC']),
    #         float(detection['THETA'])
    #     )[0] for i, detection in no_match.iterrows()]
    # ]
    no_match['Prev. Det.'] = [
        get_previous_detection_status(
            detection['ObsId'],
            float(detection['RA']),
            float(detection['DEC']),
            float(detection['POS_ERR'])
        ) for i, detection in no_match.iterrows()
    ]
    # no_match['Flux'] = [
    #     f'${get_flux(detection):.1E}$' for i, detection in no_match.iterrows()
    # ]
    no_match['THETA'] = [
        f'${float(theta):.1f}\\arcmin$' for theta in no_match['THETA']]
    no_match['POS_ERR'] = [
        f'${2*float(pos_err):.2f}\\arcsec$' for pos_err in no_match['POS_ERR']]
    no_match['SIGNIFICANCE'] = [
        f'${float(significance):.1f}$' for significance in no_match['SIGNIFICANCE']]
    no_match['RA'] = [f'${float(ra):.5f}$' for ra in no_match['RA']]
    no_match['DEC'] = [f'${float(dec):.5f}$' for dec in no_match['DEC']]
    no_match['ObsId'] = no_match['ObsId'].astype(int)

    no_match = no_match[order]

    no_match = no_match.rename(columns=column_names)

    no_match.sort_values('ObsId', inplace=True)

    no_match.index = range(start_index, start_index + len(no_match))

    return no_match


def get_fxt_table_detections(detections: pd.DataFrame, start_index: int = 1) -> pd.DataFrame:
    """
    ## Get the FXT table.

    ### Args:
        detections `pd.DataFrame`: Detections to get the FXT table for.
        start_index `int` (optional): Defaults to `1`. Index to start the table at.

    ### Returns:
        `pd.DataFrame`: FXT table.
    """
    order = [
        'Id', 'Previous Detection', 'ObsId', 'Exposure', 'Date', 'T_90', 'RA', 'DEC', 'THETA', 'POS_ERR', 'SIGNIFICANCE'  # , 'HR', 'Flux'
    ]

    column_names = {
        'RA': 'RA (deg)',
        'DEC': 'Dec (deg)',
        'THETA': 'Off. Ang.',
        'SIGNIFICANCE': 'S/N',
        'POS_ERR': 'Pos. Unc.',
        'T_90': '$T_{90}$ (ks)',
        'Exposure': 'Exp. (ks)',
    }

    detections['Exposure'] = [
        f'${get_exposure_time(obsid):.1f}$' for obsid in detections['ObsId']]
    detections['Date'] = [get_date(obsid) for obsid in detections['ObsId']]
    detections['Date'] = ['\makecell{' + f'{date.split(" - ")[0]}/' + '\\\\' + f'{date.split(" - ")[1]}' +
                          '}' if ' - ' in date else date for date in detections['Date']]
    detections['Id'] = [get_fxt_id(detection)
                        for i, detection in detections.iterrows()]
    detections['T_90'] = [
        get_t_90_str(
            detection['ObsId'],
            float(detection['RA']),
            float(detection['DEC']),
            float(detection['THETA'])
        ) for i, detection in detections.iterrows()
    ]
    # no_match['HR'] = [
    #     f'${float(hr):.1f}$' for hr in [get_HR(
    #         detection['ObsId'],
    #         float(detection['RA']),
    #         float(detection['DEC']),
    #         float(detection['THETA'])
    #     )[0] for i, detection in no_match.iterrows()]
    # ]
    detections['Previous Detection'] = [
        get_previous_detection_status(
            detection['ObsId'],
            float(detection['RA']),
            float(detection['DEC']),
            float(detection['POS_ERR'])
        ) for i, detection in detections.iterrows()
    ]
    # no_match['Flux'] = [
    #     f'${get_flux(detection):.1E}$' for i, detection in no_match.iterrows()
    # ]
    detections['THETA'] = [
        f'${float(theta):.1f}^' + '{\prime}$' for theta in detections['THETA']]
    detections['POS_ERR'] = [
        f'${2*float(pos_err):.2f}^' + '{\prime\prime}$' for pos_err in detections['POS_ERR']]
    detections['SIGNIFICANCE'] = [
        f'${float(significance):.1f}$' for significance in detections['SIGNIFICANCE']]
    detections['RA'] = [f'${float(ra):.5f}$' for ra in detections['RA']]
    detections['DEC'] = [f'${float(dec):.5f}$' for dec in detections['DEC']]
    detections['ObsId'] = detections['ObsId'].astype(int)

    detections = detections[order]

    detections = detections.rename(columns=column_names)

    detections.sort_values('ObsId', inplace=True)

    detections.index = range(start_index, start_index + len(detections))

    return detections


def get_fxt_dataframe_detections(detections: pd.DataFrame, start_index: int = 1) -> pd.DataFrame:
    """
    ## Get the FXT table.

    ### Args:
        detections `pd.DataFrame`: Detections to get the FXT table for.
        start_index `int` (optional): Defaults to `1`. Index to start the table at.

    ### Returns:
        `pd.DataFrame`: FXT table.
    """
    order = [
        'Id', 'Search', 'ObsId', 'Exposure', 'Date', 'T_90', 'RA', 'DEC', 'THETA', 'POS_ERR', 'SIGNIFICANCE'  # , 'HR', 'Flux'
    ]

    column_names = {
        'RA': 'RA (deg)',
        'DEC': 'Dec (deg)',
        'THETA': 'Off. Ang.',
        'SIGNIFICANCE': 'S/N',
        'POS_ERR': 'Pos. Unc.',
        'Exposure': 'Exp. (ks)',
    }

    detections['Exposure'] = [
        f'{get_exposure_time(obsid):.1f}' for obsid in detections['ObsId']]
    detections['Date'] = [get_date(obsid) for obsid in detections['ObsId']]
    detections['Id'] = [get_fxt_id(detection)
                        for i, detection in detections.iterrows()]
    detections['T_90'] = [
        get_t_90(
            detection['ObsId'],
            float(detection['RA']),
            float(detection['DEC']),
            float(detection['THETA'])
        ) for i, detection in detections.iterrows()
    ]
    detections['T_90'] = [
        f'{t_90/1000:.1f} +{t_90_upper/1000:.1f} -{t_90_lower/1000:.1f}' for t_90, t_90_upper, t_90_lower in detections['T_90']
    ]
    # no_match['HR'] = [
    #     f'${float(hr):.1f}$' for hr in [get_HR(
    #         detection['ObsId'],
    #         float(detection['RA']),
    #         float(detection['DEC']),
    #         float(detection['THETA'])
    #     )[0] for i, detection in no_match.iterrows()]
    # ]
    # detections['Previous Detection'] = [
    #     get_previous_detection_status(
    #         detection['ObsId'],
    #         float(detection['RA']),
    #         float(detection['DEC']),
    #         float(detection['POS_ERR'])
    #     ) for i, detection in detections.iterrows()
    # ]
    detections['Search'] = detections['search']
    # no_match['Flux'] = [
    #     f'${get_flux(detection):.1E}$' for i, detection in no_match.iterrows()
    # ]
    detections['THETA'] = [
        f'{float(theta):.1f}' for theta in detections['THETA']]
    detections['POS_ERR'] = [
        f'{2*float(pos_err):.2f}' for pos_err in detections['POS_ERR']]
    detections['SIGNIFICANCE'] = [
        f'{float(significance):.1f}' for significance in detections['SIGNIFICANCE']]
    detections['RA'] = [f'{float(ra):.5f}' for ra in detections['RA']]
    detections['DEC'] = [f'{float(dec):.5f}' for dec in detections['DEC']]
    detections['ObsId'] = detections['ObsId'].astype(int)

    detections = detections[order]

    detections = detections.rename(columns=column_names)

    detections.sort_values('ObsId', inplace=True)

    detections.index = range(start_index, start_index + len(detections))

    return detections


def dataframe_to_latex(dataframe: pd.DataFrame) -> str:
    latex = '\\begin{table}[h]' + '\n'
    latex += '    \\centering' + '\n'
    latex += '    \\caption{}' + '\n'
    latex += '    \\begin{tabular}{' + \
        f'{(len(dataframe.columns) + 1) * "c"}' + '}\n'
    columns = ['\headercell{' + column + '}' for column in dataframe.columns]
    latex += '        \headerrow & ' + \
        ' & '.join(columns) + ' \\\\' + '\n'
    for i, (index, row) in enumerate(dataframe.iterrows()):
        row = row.astype(str).tolist()

        if i % 2 == 0:
            latex += f'        \oddrow {index} & ' + \
                ' & '.join(row) + ' \\\\' + '\n'
        else:
            latex += f'        \evenrow {index} & ' + \
                ' & '.join(row) + ' \\\\' + '\n'
    latex += '        \\multicolumn{' + \
        f'{(len(dataframe.columns) + 1)}' + \
        '}{p{0.5\linewidth}}{\\textbf{Notes:} }' + '\n'
    latex += '    \\end{tabular}' + '\n'
    latex += '    \\label{tab:enter_label}' + '\n'
    latex += '\\end{table}' + '\n'

    return latex


def get_new_detections(window: float, no_match: bool = False) -> pd.DataFrame:
    """
    ## Get the detections that are found in a window but not in the 20ks window.
    DEPRECATED: replace with ```get_candidates```


    ### Args:
        window `float`: Window that the detections are found in.
        no_match `bool` (optional): Defaults to `False`. Wether to only give detections that have no match in any filters.

    ### Returns:
        `pd.DataFrame`: New detections.
    """
    filtered_20 = pd.read_csv('output/filtered_w20.csv', dtype=str)
    filtered_w = pd.read_csv(f'output/filtered_w{int(window)}.csv', dtype=str)

    new_detections = filtered_w[~filtered_w['RA'].isin(
        filtered_20['RA']) | ~filtered_w['DEC'].isin(filtered_20['DEC'])]

    if no_match:
        for f in ALL_FILTERS:
            new_detections = new_detections[new_detections[f] == 'no']

    return new_detections


if __name__ == '__main__':
    pass

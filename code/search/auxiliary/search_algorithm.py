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
from typing import List, Tuple
import time


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


def get_chandra_eef(thetas: np.ndarray, R0: float = 1.32, R10: float = 10.1, alpha: float = 2.42) -> list:
    """
    ## Calculates Chandra EEF (encircled energy fraction) radius from Vito+16


    ### Args:
        thetas `np.ndarray`: A list of off-axis angle (') in arcmin.
        R0 `float` (optional): EEF radius (") for off-axis angle = 0. Defaults to `1.32`. 1.07 (90% EEF, from Table A1 of Vito+16)
        R10 `float` (optional): EEF radius (") for off-axis angle = 0. Defaults to `10.1`. 9.65 (90% EEF, from Table A1 of Vito+16)
        alpha `float` (optional): Powerlaw index. Defaults to `2.42`.

    ### Returns:
        `list`: EEF radia from thetas (") in arcmin.
    """
    # create EEF array
    EEF_radius = np.zeros(len(thetas)) - 99.

    # get sources with a positive off-axis angle
    positive_theta_sources = np.where(thetas >= 0)[0]

    # calculate eef
    EEF_radius[positive_theta_sources] = \
        R0 + R10 * (thetas[positive_theta_sources] / 10.)**alpha

    # get number of sources with negative off-axis angle
    bad_source_count = len(thetas) - len(positive_theta_sources)
    if bad_source_count > 0:
        print(
            f'warning: {bad_source_count} sources are not calculated due to negative off-axis angle')

    return EEF_radius


def get_counts_from_event(
    event_data: dict,
    source_x: float,
    source_y: float,
    source_radius: int,
    background_radius: int
) -> tuple[int, int]:
    """
    ## Get the total counts and background counts from the event data.


    ### Args:
        evt_data `dict`: The raw event 2 table.
        src_x `float`: The physical x coordinate of the source in the observation.
        src_y `float`: The physical y coordinate of the source in the observation.
        Rsrc `int`: Source is extracted within a circle Rin=Rsrc (px).
        Rbkg `int`: Background is an annulus, Rout=Rbkg (px).

    ### Returns:
        `tuple[int, int]`: A tuple of total counts and background counts.
    """
    # Get source area and background area
    source_area = np.pi * source_radius**2
    background_area = np.pi * (background_radius**2 - source_radius**2)

    # Calculate background scale factor
    scale_factor = source_area / background_area

    # Calculate distance from event to source, squared
    distance_squared = \
        (event_data['x'] - source_x)**2 + \
        (event_data['y'] - source_y)**2

    # Select events within the aperture
    events_in_source = \
        np.where(distance_squared <= source_radius**2)[0]
    events_in_background = \
        np.where(distance_squared <= background_radius**2)[0]

    # Get the counts in source extraction aperture
    total_counts = len(events_in_source)

    # Get background counts
    background_counts = len(events_in_background) - total_counts

    return total_counts, background_counts * scale_factor


def get_before_after_counts(
    event_data_raw: dict,
    source_xs: list[float],
    source_ys: list[float],
    aperture_radii: list[int],
    t_begin: float,
    t_end: float,
    lower_energy: float = 5e2,
    upper_energy: float = 7e3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ## A function to calculate N1, N2, and N1', N2' described in the draft.


    ### Args:
        event_data_raw `dict`: The raw event 2 table.
        source_xs `list[float]`: The physical x coordinate of sources in the observation.
        source_ys `list[float]`: The physical y coordinate of sources in the observation.
        aperture_radii `list[int]`: The aperture size for each source (px).
        t_begin `float`: The start time of the exposure.
        t_end `float`: The end time of the exposure.
        lower_energy `float` (optional): Defaults to `5e2`. The lower limit of the Chandra energy band used.
        upper_energy `float` (optional): Defaults to `7e3`. The upper limit of the Chandra energy band used. Defaults to `7e3`.

    ### Returns:
        `tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`: A tuple of N1 (counts before med_exp), N2 (counts after med_exp), N1' (counts at the edge), N2' (counts at the center).
    """
    # Only use 0.5-7 keV range
    event_data = \
        event_data_raw[np.where(
            (event_data_raw['energy'] >= lower_energy) &
            (event_data_raw['energy'] <= upper_energy)
        )]

    # Get the time bins, divided into quartiles
    t_start, t_1, t_2, t_3, t_stop = np.percentile(
        [t_begin, t_end], [0, 25, 50, 75, 100]
    )

    # Event data before/after the middle of the given frame of the exposure
    events_before = np.where(event_data['time'] < t_2)[0]
    events_after = np.where(event_data['time'] >= t_2)[0]

    # Event data in first/last quartiles and middle two quartiles
    events_edges = np.where(
        (event_data['time'] < t_1) |
        (event_data['time'] >= t_3)
    )[0]
    events_center = np.where(
        (event_data['time'] >= t_1) &
        (event_data['time'] < t_3)
    )[0]

    # Extract event x, y
    event_xs, event_ys = event_data['x'], event_data['y']

    # Calculate events before and after
    before_counts,  after_counts, edge_counts, center_counts = [], [], [], []

    for i, (source_x, source_y) in enumerate(zip(source_xs, source_ys)):
        source_radius, background_radius = aperture_radii[i], aperture_radii[i] + 22.
        total_counts, background_counts = \
            get_counts_from_event(
                event_data,
                source_x,
                source_y,
                source_radius,
                background_radius
            )

        # Check if background is too strong
        if total_counts < 5 * background_counts:  # condition i in paper
            before_counts.append(-99)
            after_counts.append(-99)
            edge_counts.append(-99)
            center_counts.append(-99)
        else:
            # Calculate distance square
            distance_to_source_squared = \
                (event_xs - source_x)**2 + \
                (event_ys - source_y)**2

            # Select events within the aperture
            events_in_aperture = np.where(
                distance_to_source_squared <= source_radius**2
            )[0]

            # Count events before and after med_exp
            before_counts.append(len(np.intersect1d(
                events_in_aperture, events_before
            )))
            after_counts.append(len(np.intersect1d(
                events_in_aperture, events_after
            )))
            edge_counts.append(len(np.intersect1d(
                events_in_aperture, events_edges
            )))
            center_counts.append(len(np.intersect1d(
                events_in_aperture, events_center
            )))

    # Convert lists to arrays
    before_counts = np.array(before_counts)
    after_counts = np.array(after_counts)
    edge_counts = np.array(edge_counts)
    center_counts = np.array(center_counts)

    return before_counts, after_counts, edge_counts, center_counts


def get_transient_candidates(counts_1: np.ndarray[int], counts_2: np.ndarray[int]) -> np.ndarray:
    """
    ## Select transient candidates based on the counts before and after events.

    ### Args:
        counts_1 `np.ndarray[int]`: The counts before the events.
        counts_2 `np.ndarray[int]`: The counts after the events.

    ### Returns:
        `np.ndarray`: A boolean array indicating whether a source is a transient candidate.
    """
    transient_candidates = np.zeros(len(counts_1), dtype=bool)

    # Only calculate sources with S/N not too low
    good_sources_1 = np.where(counts_1 >= 0)[0]
    good_sources_2 = np.where(counts_2 >= 0)[0]
    good_sources = np.intersect1d(good_sources_1, good_sources_2)

    # Calculate counts upper and lower Poisson limit
    counts_1_lower,  counts_1_upper = \
        poisson_conf_interval(
            counts_1[good_sources],
            interval='frequentist-confidence',
            sigma=5
        )
    counts_2_lower, counts_2_upper = \
        poisson_conf_interval(
            counts_2[good_sources],
            interval='frequentist-confidence',
            sigma=5
        )

    # Select XT candidates
    transient_candidates[good_sources] = (
        (counts_2[good_sources] > counts_1_upper) |  # condition ii in paper
        (counts_2[good_sources] < counts_1_lower)
    ) & (
        (counts_1[good_sources] > counts_2_upper) |  # condition ii in paper
        (counts_1[good_sources] < counts_2_lower)
    ) & (
        # condition iii in paper
        (counts_1[good_sources] > 5 * counts_2[good_sources]) |
        (counts_2[good_sources] > 5 * counts_1[good_sources])
    )

    return transient_candidates


def transient_selection(
    event_data_raw: Table,
    source_xs: List[float],
    source_ys: List[float],
    aperture_radii: List[int],
    t_begin: float,
    t_end: float
) -> np.ndarray:
    """
    ## Select transient candidates based on the counts before and after the event.

    ### Args:
        event_data_raw `Table`: Raw event 2 table.
        source_xs `List[float]`: List of x coordinates of sources.
        source_ys `List[float]`: List of y coordinates of sources.
        aperture_radii `List[int]`: List of aperture sizes, in pixels.
        t_begin `float`: Start time of (part of the) observation.
        t_end `float`: End time of (part of the) observation.

    ### Returns:
        `np.ndarray`: A boolean array indicating whether the source is a transient candidate.
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

    return transient_candidates

# Below are an example of usage
# Note: a helpful Chandra observation list can be found at
# https://heasarc.gsfc.nasa.gov/db-perl/W3Browse/w3table.pl?tablehead=name%3Dchanmaster&Action=More+Options


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

    print(start_end_times)

    return start_end_times


def Yang_search(
    filename: str,
    ra: List[float],
    dec: List[float],
    theta: List[float],
    position_error: List[float],
    significance: List[float],
    window: float = 20.0,
    verbose: int = 0
) -> None:
    """
    ## Search for transient candidates in the given observation.

    ### Args:
        filename `str`: The filename of the event file.
        ra `List[float]`: The right ascension of the sources.
        dec `List[float]`: The declination of the sources.
        theta `List[float]`: The off-axis angle of the sources.
        position_error `List[float]`: The position error of the sources.
        significance `List[float]`: The significance of the sources.
        window `float` (optional): Defaults to `20.0`. The window size in kiloseconds.
        verbose `int` (optional): Defaults to `0`. The level of verbosity.
    """
    with fits.open(filename) as hdul:
        information = hdul[1].header
        obs = information['OBS_ID']
        t_start = information['TSTART']
        t_stop = information['TSTOP']

    obs_id = int(obs)

    # Set pixel scale (units: arcsec per pix)
    acis_pix_size = 0.492

    # Read the event file
    event_data_raw = Table.read(filename, hdu=1)

    # Read the wcs of the event file
    event_wcs = get_wcs_event(filename)

    # Extract ra, dec of sources in the observation
    source_ras, source_decs, source_thetas = \
        np.array(ra), np.array(dec), np.array(theta)
    source_pos_err, source_sig = \
        np.array(position_error), np.array(significance)

    # Convert ra,dec to x,y
    source_xs, source_ys = \
        event_wcs.all_world2pix(source_ras, source_decs, 1)

    # Get R90 size
    r90_size = get_chandra_eef(source_thetas, R0=1.07, R10=9.65, alpha=2.22)

    # Convert to pixel scale
    r90_size /= acis_pix_size

    # Get the aperture size
    aperture_radii = r90_size * 1.5

    candidates = []

    # full observation
    new_candidates = transient_selection(
        event_data_raw,
        source_xs,
        source_ys,
        aperture_radii,
        t_start,
        t_stop
    )
    candidates.extend(new_candidates)

    # split the observation
    for t_begin, t_end in get_start_end_times((t_stop - t_start) / 1000.0, window):
        t_begin, t_end = t_begin * 1000.0 + t_start, t_end * 1000.0 + t_start

        event_data = event_data_raw[np.where(
            (event_data_raw['time'] >= t_begin) &
            (event_data_raw['time'] < t_end)
        )]

        if (len(event_data) == 0):
            continue

        new_candidates = transient_selection(
            event_data,
            source_xs,
            source_ys,
            aperture_radii,
            t_begin,
            t_end
        )
        candidates.extend(new_candidates)

    candidates = pd.unique(np.array(candidates)).tolist()

    with open(f"{sys.argv[1]}/output/detections_w{int(window)}.txt", "a") as f:
        for i, candidate in enumerate(candidates):
            f.write(
                f'{obs_id} {source_ras[candidate]} {source_decs[candidate]} {source_thetas[candidate]} {source_pos_err[candidate]} {source_sig[candidate]}\n'
            )

    with open(f"{sys.argv[1]}/output/analysed_w{int(window)}.txt", "a") as f:
        f.write(f'{obs_id}\n')


def off_axis(event_file: str, ra: float, dec: float) -> float:
    """
    ## Calculate the off-axis angle of a source in an event file.

    ### Args:
        event_file `str`: Event file to search for the source.
        ra `float`: Right ascension of the source.
        dec `float`: Declination of the source.

    ### Returns:
        `float`: Off-axis angle of the source.
    """
    t1 = time.perf_counter()

    command = f'dmcoords {event_file} op=cel ra={ra} dec={dec} verbose=0'
    proc = subprocess.run(command, shell=True)

    t2 = time.perf_counter()

    command = f'pget dmcoords theta'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    out = proc.stdout

    # for i, line in enumerate(output):
    #     if (line == b'THETA,PHI'):
    #         z = output[i + 1]
    #         break

    print(f"\tproc: {t2-t1:.2f} seconds")

    return float(out)


def search_candidates(src_file: str, event_file: str, window: float = 20.0, verbose: int = 0):
    """
    ## Search for transient candidates in the given observation.

    ### Args:
        src_file `str`: Source file to search for candidates.
        event_file `str`: Event file to search for candidates.
        window `float` (optional): Defaults to `20.0`. Window size in kiloseconds.
    """
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

    t1 = time.perf_counter()

    for i, _ in enumerate(RA):
        a = off_axis(event_file, RA[i], DEC[i])
        THETA.append(a)
        err_pos.append(np.sqrt(X_err[i]**2+Y_err[i]**2)*0.492)

    t2 = time.perf_counter()

    print(f"off_axis calc: {t2-t1:.2f} seconds")

    Yang_search(event_file, RA, DEC, THETA, err_pos,
                significance, window, verbose)


if __name__ == '__main__':
    try:
        files = glob.glob('s3_expmap_src.fits', recursive=True)
        src_file = files[0]

        files = glob.glob('*evt2.fits', recursive=True)
        event_file = files[0]

        try:
            window_value = float(sys.argv[2])
        except IndexError:
            window_value = 20.0  # default value

        search_candidates(src_file, event_file,
                          window_value, verbose=sys.argv[3])
    except Exception as e:
        print('Error with search - ', e)

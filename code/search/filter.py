import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from typing import Callable, Dict

from auxiliary.filter_functions import filter_gaia, filter_archival, filter_chandra, filter_erosita, filter_ned, filter_simbad, filter_vizier, filter_Xray_binaries


def update_catalogs(dataframe: pd.DataFrame, catalogs: Dict[str, Callable], verbose: int = 0) -> pd.DataFrame:
    """
    ## Return a new dataframe with the catalog match columns added.

    ### Args:
        dataframe `pd.DataFrame`: Dataframe to add the catalog match columns to.
        catalogs `Dict[str, Callable]`: Catalogs to add to the dataframe.
        verbose `int` (optional): Defaults to `0`. Level of verbosity.

    ### Returns:
        `pd.DataFrame`: Dataframe with the catalog match columns added.
    """
    dataframe = dataframe.copy()

    for catalog, _ in catalogs.items():
        if f'{catalog}_match' not in dataframe.columns:
            dataframe.insert(len(dataframe.columns),
                             f'{catalog}_match', 'unknown')

            if verbose > 2:
                print(f'Added {catalog}_match column.')

    return dataframe


def update_detections(detections: pd.DataFrame, filtered: pd.DataFrame, catalogs: Dict[str, Callable], verbose: int = 0) -> pd.DataFrame:
    """
    ## Adds new detections to the filtered dataframe.

    ### Args:
        detections `pd.DataFrame`: Dataframe of detections to add to the filtered dataframe.
        filtered `pd.DataFrame`: Dataframe of filtered detections.
        catalogs `Dict[str, Callable]`: Dictionary of catalogs to add to the dataframe.

    ### Returns:
        `pd.DataFrame`: The updated filtered dataframe with new detections added.
    """
    detections = detections.copy()
    filtered = filtered.copy()

    detections = update_catalogs(detections, catalogs)
    filtered = update_catalogs(filtered, catalogs)

    # add new detections to filtered dataframe
    for i, detection in detections.iterrows():
        if (
            detection.at['ObsId'] in filtered['ObsId'].values and
            detection.at['RA'] in filtered['RA'].values and
            detection.at['DEC'] in filtered['DEC'].values and
            detection.at['THETA'] in filtered['THETA'].values and
            detection.at['POS_ERR'] in filtered['POS_ERR'].values and
            detection.at['SIGNIFICANCE'] in filtered['SIGNIFICANCE'].values
        ):
            if verbose > 2:
                print(
                    f'{i}: {detection.at["ObsId"]} - Detection already in filtered.')
            continue

        filtered.loc[len(filtered)] = detection

        if verbose > 2:
            print(
                f'{i}: {detection.at["ObsId"]} - Added detection to filtered.')

    return filtered


def filter_detections(detections: pd.DataFrame, filtered: pd.DataFrame, catalogs: Dict[str, Callable], verbose: int = 0) -> pd.DataFrame:
    """
    ## Filter the detections in the detections dataframe.

    ### Args:
        detections `pd.DataFrame`: Dataframe of detections to filter.
        filtered `pd.DataFrame`: Dataframe of filtered detections.
        catalogs `Dict[str, Callable]`: Catalogs to filter the detections with.
        verbose `int` (optional): Defaults to `0`. Level of verbosity.

    ### Returns:
        `pd.DataFrame`: Filtered dataframe of detections.
    """
    # add new detections to filtered dataframe
    filtered = update_detections(detections, filtered, catalogs)

    for i, detection in filtered.iterrows():
        for catalog, filter_func in catalogs.items():
            if detection.at[f'{catalog}_match'] == 'unknown':
                try:
                    result = 'yes' if filter_func(detection) else 'no'
                    filtered.at[i, f'{catalog}_match'] = result

                    if verbose > 1:
                        print(
                            f'{i}: {detection.at["ObsId"]} - {catalog} match: {filtered.at[i, f"{catalog}_match"]}')
                except Exception as e:
                    print(f'{i}: {detection.at["ObsId"]} - {e}')
                    continue
            else:
                if verbose > 1:
                    print(
                        f'{i}: {detection.at["ObsId"]} - {catalog} match: already known.')

    return filtered


def filter_detection_file(detections_filename: str, filtered_filename: str, catalogs: Dict[str, Callable], verbose: int = 0) -> None:
    """
    ## Filter the detections in the detections file.

    ### Args:
        detections_filename `str`: Filename of the detections file.
        filtered_filename `str`: Filename of the filtered file.
        catalogs `Dict[str, Callable]`: Catalogs to filter the detections with.
        verbose `int` (optional): Defaults to `0`. Level of verbosity.
    """
    detections = pd.read_csv(detections_filename, sep=' ', header=0, dtype=str)
    filtered = pd.read_csv(filtered_filename, sep=',', header=0, dtype=str)

    if verbose > 0:
        print(
            f'Analysing {len(detections)} detections from {detections_filename}'
        )

    filtered = filter_detections(detections, filtered, catalogs, verbose)

    filtered.to_csv(filtered_filename, index=False)


def clear_filter_matches(filtered_filename: str, catalog: str) -> None:
    """
    ## Clear the matches for a specific catalog in the filtered file.

    ### Args:
        filtered_filename `str`: Filename of the filtered file.
        catalog `str`: Catalog to clear the matches for.
    """
    filtered = pd.read_csv(filtered_filename, sep=',', header=0, dtype=str)

    for i, detection in filtered.iterrows():
        if f'{catalog}_match' in filtered.columns:
            filtered.at[i, f'{catalog}_match'] = 'unknown'

    filtered.to_csv(filtered_filename, index=False)


DETECTIONS_FILENAME = 'output/detections_w20.txt'
FILTERED_FILENAME = 'output/filtered_w20.csv'
CATALOGS = {
    'gaia': filter_gaia,
    'archival': filter_archival,
    'chandra': filter_chandra,
    'erosita': filter_erosita,
    'ned': filter_ned,
    'simbad': filter_simbad,
    'vizier': filter_vizier,
    'xray-binaries': filter_Xray_binaries
}
VERBOSE = 2

if __name__ == '__main__':
    filter_detection_file(
        DETECTIONS_FILENAME,
        FILTERED_FILENAME,
        CATALOGS,
        VERBOSE
    )

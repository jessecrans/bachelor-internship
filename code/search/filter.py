import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from typing import Callable, Dict

from filter_functions import filter_on_gaia

"""
workflow: 

"""

# we only need 1 row because we are only checking if there is a match or not
Gaia.ROW_LIMIT = 1

DETECTIONS_FILENAME = 'detections_w20.txt'
FILTERED_FILENAME = 'filtered_w20.csv'

CATALOGS = {
    'gaia': filter_on_gaia,
}


def update_catalogs(dataframe: pd.DataFrame, catalogs: Dict[str, Callable], verbose: bool = False) -> pd.DataFrame:
    """
    Returns a new dataframe with columns added for each missing catalog.

    Args:
        dataframe (pd.DataFrame): Dataframe to update with missing columns.
        catalogs (Dict[str, Callable]): Dictionary of catalogs to add to the dataframe.

    Returns:
        pd.DataFrame: The updated dataframe with missing columns added.
    """
    dataframe = dataframe.copy()

    for catalog, filter_func in catalogs.items():
        if f'{catalog}_match' not in dataframe.columns:
            dataframe.insert(len(dataframe.columns),
                             f'{catalog}_match', 'unknown')

            if verbose:
                print(f'Added {catalog}_match column.')

    return dataframe


def update_detections(detections: pd.DataFrame, filtered: pd.DataFrame, catalogs: Dict[str, Callable], verbose: bool = False) -> pd.DataFrame:
    """
    Adds new detections to the filtered dataframe.

    Args:
        detections (pd.DataFrame): Dataframe of detections to add to the filtered dataframe.
        filtered (pd.DataFrame): Dataframe of filtered detections.
        catalogs (Dict[str, Callable]): Dictionary of catalogs to add to the dataframe.

    Returns:
        pd.DataFrame: The updated filtered dataframe with new detections added.
    """
    detections = detections.copy()
    filtered = filtered.copy()

    detections = update_catalogs(detections, catalogs)
    filtered = update_catalogs(filtered, catalogs)

    # add new detections to filtered dataframe
    for i, detection in detections.iterrows():
        if (
            detection['ObsId'] in filtered['ObsId'].values and
            detection['RA'] in filtered['RA'].values and
            detection['DEC'] in filtered['DEC'].values and
            detection['THETA'] in filtered['THETA'].values and
            detection['POS_ERR'] in filtered['POS_ERR'].values and
            detection['SIGNIFICANCE'] in filtered['SIGNIFICANCE'].values
        ):
            if verbose:
                print(
                    f'{i}: {detection["ObsId"]} - Detection already in filtered.')
            continue

        filtered.loc[len(filtered)] = detection

        if verbose:
            print(f'{i}: {detection["ObsId"]} - Added detection to filtered.')

    return filtered


def filter_detections(detections: pd.DataFrame, filtered: pd.DataFrame, catalogs: Dict[str, Callable], verbose: bool = False) -> pd.DataFrame:
    # add new detections to filtered dataframe
    filtered = update_detections(detections, filtered, catalogs)

    for i, detection in filtered.iterrows():
        for catalog, filter_func in catalogs.items():
            if detection[f'{catalog}_match'] == 'unknown':
                try:
                    result = 'yes' if filter_func(detection) else 'no'
                    filtered.loc[i, f'{catalog}_match'] = result

                    if verbose:
                        print(
                            f'{i}: {detection["ObsId"]} - {catalog} match: {detection[f"{catalog}_match"]}')
                except Exception as e:
                    print(f'{i}: {detection["ObsId"]} - {e}')
                    continue
            else:
                if verbose:
                    print(
                        f'{i}: {detection["ObsId"]} - {catalog} match: already known.')

    return filtered


def filter_detection_file(detections_filename: str, filtered_filename: str, catalogs: Dict[str, Callable], verbose: bool = False) -> None:
    detections = pd.read_csv(detections_filename, sep=' ', header=0, dtype=str)
    filtered = pd.read_csv(filtered_filename, sep=',', header=0, dtype=str)

    filtered = filter_detections(detections, filtered, catalogs, verbose)

    filtered.to_csv(filtered_filename, index=False)


if __name__ == '__main__':
    filter_detection_file(
        DETECTIONS_FILENAME,
        FILTERED_FILENAME,
        CATALOGS,
        True
    )
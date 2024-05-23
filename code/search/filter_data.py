import numpy as np
from numpy.typing import NDArray

import pandas as pd

import astropy.units as u
from astropy.coordinates import SkyCoord

from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

from typing import Callable, Dict

# we only need 1 row because we are only checking if there is a match or not
Gaia.ROW_LIMIT = 1

DETECTIONS_FILENAME = 'detections_w20.txt'
FILTERED_FILENAME = 'filtered_w20.csv'


def filter_on_gaia(detection: pd.Series) -> bool:
    """
    Checks if the given detection has a match in the Gaia catalog.

    Args:
        detection (pd.Series): Detection to check if it has a match in the Gaia catalog.

    Returns:
        bool: True if the detection has a match in the Gaia catalog, False otherwise.
    """
    coords = SkyCoord(
        ra=detection['RA'],
        dec=detection['DEC'],
        unit=(u.degree, u.degree),
        frame='icrs',
    )
    job = Gaia.cone_search_async(
        coords,
        radius=u.Quantity(
            # 3sigma + 0.5" boresight correction + 5" proper motion margin
            3 * detection['POS_ERR'] + 0.5 + 5,
            u.arcsec
        )
    )

    result = job.get_results()

    if result is None or len(result) == 0:
        return False

    return True


def filter_on_XMM(detection: pd.Series) -> bool:
    pass


CATALOGS = {
    'gaia': filter_on_gaia,
}


def update_catalogs(dataframe: pd.DataFrame, catalogs: Dict[str, Callable]) -> pd.DataFrame:
    for catalog, filter_func in catalogs.items():
        if f'{catalog}_match' not in dataframe.columns:
            dataframe.insert(len(dataframe.columns), f'{catalog}_match', -1)

    return dataframe


def update_detections(in_file: str, out_file: str, catalogs: Dict[str, Callable]) -> None:
    detections = pd.read_csv(in_file, delimiter=' ', header=0)
    detections = update_catalogs(detections, catalogs)

    filtered = pd.read_csv(out_file, delimiter=',', header=0)
    filtered = update_catalogs(filtered, catalogs)

    # add new detections to filtered file
    for i, detection in detections.iterrows():
        if (
            detection['ObsId'] in filtered['ObsId'].values and
            detection['RA'] in filtered['RA'].values and
            detection['DEC'] in filtered['DEC'].values and
            detection['THETA'] in filtered['THETA'].values and
            detection['POS_ERR'] in filtered['POS_ERR'].values and
            detection['SIGNIFICANCE'] in filtered['SIGNIFICANCE'].values
        ):
            continue

        filtered.loc[len(filtered)] = detection

    filtered.to_csv(out_file, index=False)


def filter_detections(in_file: str, out_file: str, catalogs: Dict[str, Callable], logging: bool = False) -> None:
    if logging:
        print(
            f'Filtering detections from {in_file} to {out_file} using catalogs: {catalogs.keys()}...')

    update_detections(in_file, out_file, catalogs)
    filtered = pd.read_csv(out_file, delimiter=',', header=0)

    for i, detection in filtered.iterrows():
        for catalog, filter_func in catalogs.items():
            if int(detection[f'{catalog}_match']) == -1:
                try:
                    detection[f'{catalog}_match'] = \
                        1 if filter_func(detection) else 0
                except Exception as e:
                    print(
                        f"\t{i}: {int(detection['ObsId'])} {catalog} failed with error: {e}")
                    continue

                if logging:
                    print(
                        f"\t{i}: {int(detection['ObsId'])} {catalog} filtered")
            else:
                if logging:
                    print(
                        f"\t{i}: {int(detection['ObsId'])} {catalog} already filtered")

    filtered.to_csv(out_file, index=False)


if __name__ == '__main__':
    filter_detections(DETECTIONS_FILENAME, FILTERED_FILENAME, CATALOGS, True)

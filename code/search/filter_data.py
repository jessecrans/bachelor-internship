import numpy as np
from numpy.typing import NDArray

import pandas as pd

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

# we only need 1 row because we are only checking if there is a match or not
Gaia.ROW_LIMIT = 1

detections = pd.read_csv('detections_w20.txt', delimiter=' ', header=0)
queried_detections = pd.read_csv(
    'queried_detections_w20.txt', delimiter=' ', header=0)
queried_detections_no_match = queried_detections.filter(regex='^(?!.*_MATCH)')


def occurs_in(detection: pd.Series, queried_detections: pd.DataFrame) -> pd.Series:
    """
    Returns a boolean Series indicating whether the detection has been queried before per catalog.

    Args:
        detection (pd.Series): Detection to check if it has been queried before.
        queried_detections (pd.DataFrame): DataFrame containing the queried detections.

    Returns:
        pd.Series: Boolean Series indicating whether the detection has been queried before per catalog.
    """


def filter_on_gaia(detection: pd.Series) -> Table | None:
    coords = SkyCoord(
        ra=detection['RA'],
        dec=detection['DEC'],
        unit=(u.degree, u.degree),
    )
    job = Gaia.cone_search_async(
        coords,
        radius=u.Quantity(detection['POS_ERR'], u.arcmin),
    )
    return job.get_results()


def check_detections(detections: pd.DataFrame, catalog: str, logging: bool) -> pd.DataFrame:
    detections = detections.assign(**{f'{catalog.upper()}_MATCH': False})

    for i, detection in detections.iterrows():
        if occurs_in(detection, queried_detections_no_match):
            print(
                'Detection {i} with obsid {detections.loc[i, "ObsId"]} has already been queried for {catalog}')
            continue

        result = filter_on_gaia(detection)

        if len(result) > 0:
            detections.loc[i, f'{catalog.upper()}_MATCH'] = True

            if logging:
                print(
                    f"Match found for detection {i} with obsid {detections.loc[i, 'ObsId']} in {catalog}")

    return detections


# detections_copy = detections.loc[detections['ObsId'] == 803].copy()
detections_copy = detections.loc[:].copy()

detections_copy = check_detections(detections_copy, 'gaia', False)

detections_copy.to_csv('queried_detections_w20.txt', sep=' ', index=False)

# TODO: do not repeat queries, get result from queried_detections
# TODO: if a detection has been queried for one catalog, allow it to be queried for another catalog
# TODO: save the result of a check so that we dont have to query unnecessarily
# TODO: before querying, check if the detection has already been queried
# TODO: so: if a detections values are the same as the values of a queried_detection, retrieve the result from the queried_detection

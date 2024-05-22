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

detections_file = 'detections_w20.txt'
queried_detections_file = 'queried_detections_w20.txt'

CATALOGS = [
    'gaia',
]
DETECTIONS = pd.read_csv(detections_file, delimiter=' ', header=0)
QUERIED = pd.read_csv(
    queried_detections_file,
    delimiter=' ',
    header=0
)
QUERIED_NO_MATCH = QUERIED.filter(
    regex='^(?!.*_MATCH$)'
)


def get_queried_detection(detection: pd.Series) -> pd.DataFrame:
    """
    Returns the already queried detection from the queried_detections file.

    Args:
        detection (pd.Series): Detection to check if it has been queried before.

    Returns:
        pd.Dataframe: Queried detection if it has been queried before.
    """
    return QUERIED.loc[
        QUERIED_NO_MATCH.eq(detection).all(axis=1)
    ]


def filter_on_gaia(detection: pd.Series) -> Table | None:
    """
    Checks if the given detection has a match in the Gaia catalog.

    Args:
        detection (pd.Series): Detection to check if it has a match in the Gaia catalog.

    Returns:
        Table | None: Result of the query.
    """
    coords = SkyCoord(
        ra=detection['RA'],
        dec=detection['DEC'],
        unit=(u.degree, u.degree),
        frame='icrs',
    )
    job = Gaia.cone_search_async(
        coords,
        radius=u.Quantity(detection['POS_ERR'], u.arcmin),
    )
    return job.get_results()


def check_detection(detection: pd.Series, catalog: str) -> pd.Series:
    pass

# TODO: do not repeat queries, get result from queried_detections
# TODO: if a detection has been queried for one catalog, allow it to be queried for another catalog
# TODO: so: if a detections values are the same as the values of a queried_detection, retrieve the result from the queried_detection

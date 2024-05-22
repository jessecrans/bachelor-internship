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

DETECTIONS = pd.read_csv(detections_file, delimiter=' ', header=0)


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


CATALOGS = {
    'gaia': filter_on_gaia,
}

"""
- voor elke detection in de doorgegeven lijst
- check of het al gequeried is
    - zo niet
        - start een nieuwe match dictionary
        - query alle catalogi met de dict van filter functies
        - voeg het resultaat toe aan de match dictionary
        - voeg de filtered detection to aan de nieuwe lijst met de dus de match dict
    - zo ja
        - krijg de gequeriede detectie
        - krijg de match dict
        - voor elke catalogus check of het al een entry heeft
        - zo ja niks doen
        - zo niet query en voeg toe
"""

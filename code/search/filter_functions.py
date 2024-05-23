import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.gaia import Gaia


def filter_on_gaia(detection: pd.Series) -> bool:
    """
    Checks if the given detection has a match in the Gaia catalog.

    Args:
        detection (NDArray): Detection to check if it has a match in the Gaia catalog.

    Returns:
        bool: True if the detection has a match in the Gaia catalog, False otherwise.
    """
    coords = SkyCoord(
        ra=float(detection['RA']),
        dec=float(detection['DEC']),
        unit=(u.degree, u.degree),
        frame='icrs',
    )
    job = Gaia.cone_search_async(
        coords,
        radius=u.Quantity(
            # 3sigma + 0.5" boresight correction + 5" proper motion margin
            3 * float(detection['POS_ERR']) + 0.5 + 5,
            u.arcsec
        )
    )

    result = job.get_results()

    if result is None or len(result) == 0:
        return False

    return True

import numpy as np
from numpy.typing import NDArray

import pandas as pd

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

Gaia.ROW_LIMIT = 1

detections = pd.read_csv('detections_w20.txt', delimiter=' ', header=0)


def filter_on_gaia(detection: pd.Series) -> Table | None:
    coords = SkyCoord(
        ra=detection['RA'],
        dec=detection['DEC'],
        unit=(u.degree, u.degree),
    )
    job = Gaia.cone_search_async(
        coords,
        radius=u.Quantity(detection['POS_ERR'], u.degree),
    )
    return job.get_results()


def check_detections(detections: pd.DataFrame, catalog: str) -> NDArray:
    flags = np.zeros(len(detections), dtype=bool)

    for i, detection in detections.iterrows():
        result = filter_on_gaia(detection)

        if result is not None:
            flags[i] = True

    return flags


print(check_detections(detections.iloc[2:5], 'gaia'))

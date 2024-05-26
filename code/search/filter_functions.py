import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier

import subprocess


def filter_on_gaia(detection: pd.Series) -> bool:
    """
    Checks if the given detection has a match in the Gaia catalog.

    Args:
        detection (pd.Series): Detection to check if it has a match in the Gaia catalog.

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

    # check if it has non zero proper motion

    if len(result) > 0:
        print(result)

    return True


def filter_on_archival(detection: pd.Series) -> bool:
    """
    Checks if the given detection has a match in archival x-ray data.

    Args:
        detection (pd.Series): Detection to check if it has a match in the archival catalog.

    Returns:
        bool: True if the detection has a match in the archival catalogs, False otherwise.
    """

    catalog_list = Vizier.find_catalogs([
        'XMMSL2', '2SXPS', '4XMM-DR13'
    ])

    for catalog in catalog_list.keys():
        coords = SkyCoord(
            ra=float(detection['RA']),
            dec=float(detection['DEC']),
            unit=(u.degree, u.degree),
            frame='icrs',
        )

        v = Vizier(
            row_limit=1,
        )

        result = v.query_region(
            coords,
            radius=u.Quantity(
                3 * float(detection['POS_ERR']) + 0.5,
                u.arcsec,
            ),
            catalog=catalog
        )

        if result is None or len(result) == 0:
            return False

        return True


def filter_on_chandra(detection: pd.Series) -> bool:
    """
    Checks if the given detection has a match in the Chandra catalog.

    Args:
        detection (pd.Series): Detection to check if it has a match in the Chandra catalog.

    Returns:
        bool: True if the detection has a match in the Chandra catalog, False otherwise.
    """
    command = f'search_csc pos=\"{detection["RA"]},{detection["DEC"]}\" radius={3 * detection["POS_ERR"] + 0.5} outfile=\"filter_on_chandra_result.tsv\" radunit=arcsec catalog=csc2.1 clobber=yes verbose=5'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    # Q? The process is not returning any output in the outfile.
    result = pd.read_csv('filter_on_chandra_result.tsv')
    result.pprint()

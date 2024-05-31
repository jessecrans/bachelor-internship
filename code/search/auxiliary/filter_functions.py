import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
from astroquery.ipac.ned import Ned
from astroquery.simbad import Simbad, SimbadClass

import requests
import subprocess


def filter_gaia(detection: pd.Series, verbose=False) -> bool:
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

    if verbose:
        result.pprint()

    # check if it has non zero proper motion

    if len(result) > 0:
        proper_motion = result['pm']
        proper_motion.fill_value = 0.0
        proper_motion = proper_motion.filled()
        # print(proper_motion[0])
        if proper_motion[0] != 0.0:
            # print('has pm')
            return True  # has proper motion

        # print('no pm')
        return False

    # print('no result')    ehh
    return False


def filter_archival(detection: pd.Series, verbose=False) -> bool:
    """
    Checks if the given detection has a match in archival x-ray data.

    Args:
        detection (pd.Series): Detection to check if it has a match in the archival catalog.

    Returns:
        bool: True if the detection has a match in the archival catalogs, False otherwise.
    """

    catalog_list = Vizier.find_catalogs([
        'XMMSL2', '2SXPS', '4XMM-DR13', 'IX10A'
    ])

    for catalog in catalog_list.keys():
        coords = SkyCoord(
            ra=float(detection['RA']),
            dec=float(detection['DEC']),
            unit=(u.degree, u.degree),
            frame='icrs',
        )

        # check
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

        if verbose:
            result.pprint()

        if result is None or len(result) == 0:
            return False

        return True


def filter_chandra(detection: pd.Series, verbose=False) -> bool:
    """
    Checks if the given detection has a match in the Chandra catalog.

    Args:
        detection (pd.Series): Detection to check if it has a match in the Chandra catalog.

    Returns:
        bool: True if the detection has a match in the Chandra catalog, False otherwise.
    """
    command = f'search_csc pos=\"{float(detection["RA"])},{detection["DEC"]}\" radius={3 * float(detection["POS_ERR"]) + 0.5} outfile=\"query_results/search_csc_result.tsv\" radunit=arcsec catalog=csc2.1 clobber=yes verbose=5'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    # Q? The process is not returning any output in the outfile.
    result = pd.read_csv('query_results/search_csc_result.tsv',
                         sep='\t', header=64, dtype=str)
    result['flux_significance_b'] = result['flux_significance_b'].astype(float)
    result['flux_significance_b'] = result['flux_significance_b'].fillna(0.0)

    significant_detections = result[
        (result['flux_significance_b'] > 3.0) &
        (result['obsid'].str.strip() != detection['ObsId'])
    ]

    if verbose:
        print(result[['obsid', 'flux_significance_b']])

    if len(significant_detections) > 0:
        return True

    return False


'''
supernova include because it will be discarded if galactic by other filters
'''


def filter_ned(detection: pd.Series, verbose=False) -> bool:
    """
    Checks if the given detection has a match in the NED catalog.

    Args:
        detection (pd.Series): Detection to check if it has a match in the NED catalog.

    Returns:
        bool: True if the detection has a match in the NED catalog, False otherwise.
    """
    coords = SkyCoord(
        ra=float(detection['RA']),
        dec=float(detection['DEC']),
        unit=(u.degree, u.degree),
        frame='icrs',
    )

    result = Ned.query_region(
        coords,
        radius=u.Quantity(
            3 * float(detection['POS_ERR']) + 0.5,
            u.arcsec,
        )
    )

    if verbose and result is not None:
        result.pprint_all()

    result = result.to_pandas()
    filtered_result = result[
        result['Type'] in [
            'EmLS'
        ]
    ]

    if result is None or len(result) == 0:
        return False

    return True


def filter_simbad(detection: pd.Series, verbose=False) -> bool:
    """
    Checks if the given detection has a match in the Simbad catalog.

    Args:
        detection (pd.Series): Detection to check if it has a match in the Simbad catalog.

    Returns:
        bool: True if the detection has a match in the Simbad catalog, False otherwise.
    """
    coords = SkyCoord(
        ra=float(detection['RA']),
        dec=float(detection['DEC']),
        unit=(u.degree, u.degree),
        frame='icrs',
    )

    Simbad.add_votable_fields('otype')

    result = Simbad.query_region(
        coords,
        radius=u.Quantity(
            3 * float(detection['POS_ERR']) + 0.5,
            u.arcsec,
        )
    )

    if verbose and result is not None:
        result.pprint_all()

    if result is None or len(result) == 0:
        return False

    return True


def filter_erosita(detection: pd.Series, verbose=False) -> bool:
    """
    Checks if the given detection has a match in the eROSITA catalog.

    Args:
        detection (pd.Series): Detection to check if it has a match in the eROSITA catalog.

    Returns:
        bool: True if the detection has a match in the eROSITA catalog, False otherwise.
    """
    ra = float(detection['RA'])
    dec = float(detection['DEC'])
    radius = (3 * float(detection['POS_ERR']) + 0.5) / 60.0**2
    link = f'https://erosita.mpe.mpg.de/dr1/erodat/catalogue/SCS?CAT=DR1_Main&RA={ra}&DEC={dec}&SR={radius}&VERB={1}'
    response = requests.get(link)
    with open('query_results/erosita_result.xml', 'w') as f:
        f.write(response.text)
    result = Table.read('query_results/erosita_result.xml', format='votable')

    # TODO add verbose levels and print a message if the result is none for all filters
    if verbose and result is not None:
        result.pprint_all()

    if result is None or len(result) == 0:
        return False

    return True


def filter_vizier(detection: pd.DataFrame, verbose: bool = False):
    """
    Checks if the given detection has a match in the Vizier catalog.

    Args:
        detection (pd.Series): Detection to check if it has a match in the Vizier catalog.

    Returns:
        bool: True if the detection has a match in the Vizier catalog, False otherwise.
    """
    coords = SkyCoord(
        ra=detection['RA'],
        dec=detection['DEC'],
        unit=(u.degree, u.degree),
        frame='icrs',
    )

    result = Vizier.query_region(
        coords,
        radius=u.Quantity(
            3 * float(detection['POS_ERR']) + 0.5,
            u.arcsec,
        )
    )

    if verbose and result is not None:
        for i, table in enumerate(result):
            print('table', i)
            table.pprint_all()

    if result is None or len(result) == 0:
        return False

    return True

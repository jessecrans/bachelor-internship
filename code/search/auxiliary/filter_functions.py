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
            # 3sigma + 5" proper motion margin
            3 * float(detection['POS_ERR']) + 5,
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
        'XMMSL2', '2SXPS', '4XMM-DR13', '1SXPS', '2RXS'  # IX/30 is ROSAT
    ])

    coords = SkyCoord(
        ra=float(detection['RA']),
        dec=float(detection['DEC']),
        unit=(u.degree, u.degree),
        frame='icrs',
    )

    radius = u.Quantity(
        5 * float(detection['POS_ERR']),
        u.arcsec,
    )

    result = Vizier.query_region(
        coords,
        radius=radius,
        catalog=list(catalog_list.keys())
    )

    if verbose:
        for table in result:
            if table is not None:
                table.pprint_all()

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
    command = f'search_csc pos=\"{float(detection["RA"])},{detection["DEC"]}\" radius={3 * float(detection["POS_ERR"])} outfile=\"query_results/search_csc_result.tsv\" radunit=arcsec catalog=csc2.1 clobber=yes verbose=5'
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
            3 * float(detection['POS_ERR']),
            u.arcsec,
        )
    )

    object_types_include = [
        'EmLS', 'EmObj', 'G', 'GammaS', 'GClstr', 'GGroup', 'GPair', 'GTrpl', 'G_Lens', 'IrS', 'Other', 'PofG', 'QGroup', 'QSO', 'Q_Lens', 'RadioS', 'SN', 'UvS', 'VisS', 'XrayS'
    ]

    result = result.to_pandas()
    filtered_result = result.loc[
        ~result['Type'].isin(object_types_include)
    ]

    if verbose and result is not None:
        print(filtered_result[
            ['Object Name', 'RA', 'DEC', 'Type']
        ])

    if filtered_result is None or len(filtered_result) == 0:
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

    Simbad.add_votable_fields('otype', 'otypes')

    result = Simbad.query_region(
        coords,
        radius=u.Quantity(
            3 * float(detection['POS_ERR']),
            u.arcsec,
        )
    )

    object_types_include = [
        '', 'G', 'LSB', 'bCG', 'SBG', 'H2G', 'EmG', 'AGN', 'SyG', 'Sy1', 'Sy2', 'rG', 'LIN', 'QSO', 'Bla', 'BLL', 'GiP', 'GiG', 'GiC', 'BiC', 'IG', 'PaG', 'GrG', 'CGG', 'ClG', 'PCG', 'SCG', 'vid', 'grv', 'Lev', 'gLS', 'gLe', 'Lel', 'LeG', 'LeQ', 'BH', 'GWE', 'ev', 'var', 'Rad', 'mR', 'cm', 'mm', 'smm', 'Hl', 'rB', 'Mas', 'IR', 'FIR', 'MIR', 'NIR', 'Opt', 'EmO', 'blu', 'UV', 'X', 'ULX', 'gam', 'gB', 'PoG'
    ]

    if result is None:
        return False

    result = result.to_pandas()
    filtered_result = result[
        ~result['OTYPES'].str.split('|').apply(lambda x: any(
            [otype in object_types_include for otype in x]
        ))
    ]

    if verbose and filtered_result is not None:
        print(filtered_result[
            ['MAIN_ID', 'RA', 'DEC', 'OTYPE', 'OTYPES']
        ])

    if filtered_result is None or len(filtered_result) == 0:
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
    radius = (5 * float(detection['POS_ERR'])) / 60.0**2
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


def filter_function_Stellar_Xray_Activity(result: pd.DataFrame, verbose: bool = False):
    if result is None or len(result) == 0:
        return False

    if verbose and result is not None:
        result.pprint_all()

    return True


def filter_function_Galactic_Clusters_Survey(result: pd.DataFrame, verbose: bool = False):
    if result is None or len(result) == 0:
        return False

    object_types_include = [
        '-3', '0', '1', -3, 0, 1  # probable galaxy, noise, galaxy
    ]

    filtered_result = result.loc[
        ~result['cl'].isin(object_types_include)
    ]

    if verbose and filtered_result is not None:
        print(filtered_result[
            ['RAJ2000', 'DEJ2000', 'cl']
        ])

    if filtered_result is None or len(filtered_result) == 0:
        return False

    return True


def filter_function_SDSS16(result: pd.DataFrame, verbose: bool = False):
    if result is None or len(result) == 0:
        return False

    object_types_exclude = [
        '6', 6  # star
    ]

    filtered_result = result.loc[
        result['class'].isin(object_types_exclude)
    ]

    if verbose and filtered_result is not None:
        print(filtered_result[
            ['RA_ICRS', 'DE_ICRS', 'class']
        ])

    if filtered_result is None or len(filtered_result) == 0:
        return False

    return True


def filter_function_VHS(result: pd.DataFrame, verbose: bool = False):
    if result is None or len(result) == 0:
        return False

    object_types_include = [
        '1', '0', '-3', 1, 0, 3  # galaxy, noise, probable galaxy
    ]

    filtered_result = result.loc[
        ~result['Mclass'].isin(object_types_include)
    ]

    if verbose and filtered_result is not None:
        print(filtered_result[
            ['RAJ2000', 'DEJ2000', 'Mclass']
        ])

    if filtered_result is None or len(filtered_result) == 0:
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
    catalog_list = Vizier.find_catalogs([
        'J/ApJ/902/114/table2',
        'II/319/gcs9',
        'V/154/sdss16',
        'II/367/vhs_dr5',
    ])

    filter_functions_dict = {
        'J/ApJ/902/114/table2': filter_function_Stellar_Xray_Activity,
        'II/319/gcs9': filter_function_Galactic_Clusters_Survey,
        'V/154/sdss16': filter_function_SDSS16,
        'II/367/vhs_dr5': filter_function_VHS,
    }

    coords = SkyCoord(
        ra=float(detection['RA']),
        dec=float(detection['DEC']),
        unit=(u.degree, u.degree),
        frame='icrs',
    )

    radius = u.Quantity(
        3 * float(detection['POS_ERR']),
        u.arcsec,
    )

    for catalog in catalog_list.keys():
        if verbose:
            print(f'Vizier: {catalog}')

        result = Vizier.query_region(
            coords,
            radius=radius,
            catalog=[catalog]
        )

        if result is None or len(result) == 0:
            if verbose:
                print(f'\tno match')
            continue

        result = result[0]
        result = result.to_pandas()

        if filter_functions_dict[catalog](result, verbose):
            if verbose:
                print(f'\tmatch')
            return True
        else:
            if verbose:
                print(f'\tno match')

    return False


def filter_Xray_binaries(detection: pd.Series, verbose=False) -> bool:
    """
    Checks if the given detection has a match in the X-ray binaries catalog.

    Args:
        detection (pd.Series): Detection to check if it has a match in the X-ray binaries catalog.

    Returns:
        bool: True if the detection has a match in the X-ray binaries catalog, False otherwise.
    """
    coords = SkyCoord(
        ra=float(detection['RA']),
        dec=float(detection['DEC']),
        unit=(u.degree, u.degree),
        frame='icrs',
    )

    radius = u.Quantity(
        3 * float(detection['POS_ERR']),
        u.arcsec,
    )

    result = Vizier.query_region(
        coords,
        radius=radius,
        catalog='J/MNRAS/498/4790/table2'
    )

    if result is None or len(result) == 0:
        return False

    if result is not None and verbose:
        result[0].pprint_all()

    return True

# TODO: take out 0.5 boresight correction

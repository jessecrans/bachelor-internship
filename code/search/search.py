import subprocess
import numpy as np
import glob
import os
import time
import pandas as pd
from results import read_obsids

DEFAULT_DATA_PATH = "/data/jcrans/fxrt-data/obsids"


def download_data(obsid: str, verbose: int = 0, data_path: str = DEFAULT_DATA_PATH):
    """
    ## Download data for a given obsid.

    ### Args:
        obsid `str`: Obsid to download data for.
        verbose `int` (optional): Defaults to `0`. Level of verbosity.
        data_path `str` (optional): Defaults to `DEFAULT_DATA_PATH`. Path where all downloaded data is stored.
    """
    if verbose > 0:
        print(f"\tDownloading")
        start_time = time.perf_counter()

    current_path = os.getcwd()
    os.chdir(f"{data_path}")

    command = f'download_chandra_obsid {obsid} evt2,fov,asol,bpix,msk'
    proc = subprocess.run(command, shell=True)

    command = f'cp {obsid}/primary/* {obsid}/'
    proc = subprocess.run(command, shell=True)

    command = f'gunzip {obsid}/*.gz'
    proc = subprocess.run(command, shell=True)

    os.chdir(f"{current_path}")

    if verbose > 0:
        end_time = time.perf_counter()
        print(f"\tFinished downloading")
        print(f"\tDownload time: {end_time - start_time:.0f} seconds")


def process_data(obsid: str, verbose: int = 0, data_path: str = DEFAULT_DATA_PATH) -> bool:
    """
    ## Process the data for a given obsid.

    ### Args:
        obsid `str`: Obsid to process data for.
        verbose `int` (optional): Defaults to `0`. Level of verbosity.
        data_path `str` (optional): Defaults to `DEFAULT_DATA_PATH`. Path where all downloaded data is stored.

    ### Returns:
        `bool`: Whether the data was processed successfully. Returns `False` if no data was found, `True` otherwise.
    """
    if verbose > 0:
        print(f"\tProcessing")
        start_time = time.perf_counter()

    current_path = os.getcwd()

    try:  # try to change directory, skip if fails so that rest of the obsids can be processed
        os.chdir(f"{data_path}/{obsid}/")
    except FileNotFoundError:
        if verbose > 0:
            print(f"\tNo data found")
        return False

    event_file = glob.glob('*evt2.fits', recursive=True)
    fov_file = glob.glob('*N*fov1.fits', recursive=True)

    # mask
    command = 'dmcopy "' + fov_file[0] + '" s3.fov clobber=yes'
    proc = subprocess.run(command, shell=True)

    command = 'dmcopy "' + event_file[0] + \
        '[sky=region(s3.fov)]" 578_evt2_filtered.fits clobber=yes'
    proc = subprocess.run(command, shell=True)

    command = 'fluximage 578_evt2_filtered.fits binsize=1 bands=broad outroot=s3 psfecf=0.393 clobber=yes'
    proc = subprocess.run(command, shell=True)

    command = 'mkpsfmap s3_broad_thresh.img outfile=s3_psfmap.fits energy=1.4967 ecf=0.393 clobber=yes'
    proc = subprocess.run(command, shell=True)

    # detection
    command = 'punlearn wavdetect'
    proc = subprocess.run(command, shell=True)

    command = 'pset wavdetect infile=s3_broad_thresh.img mode=h'
    proc = subprocess.run(command, shell=True)

    command = 'pset wavdetect psffile=s3_broad_thresh.psfmap mode=h'
    proc = subprocess.run(command, shell=True)

    command = 'pset wavdetect expfile=s3_broad_thresh.expmap mode=h'
    proc = subprocess.run(command, shell=True)

    command = 'pset wavdetect outfile=s3_expmap_src.fits mode=h'
    proc = subprocess.run(command, shell=True)

    command = 'pset wavdetect scellfile=s3_expmap_scell.fits mode=h'
    proc = subprocess.run(command, shell=True)

    command = 'pset wavdetect imagefile=s3_expmap_imgfile.fits mode=h'
    proc = subprocess.run(command, shell=True)

    command = 'pset wavdetect defnbkgfile=s3_expmap_nbgd.fits mode=h'
    proc = subprocess.run(command, shell=True)

    command = 'pset wavdetect regfile=s3_expmap_src.reg mode=h'
    proc = subprocess.run(command, shell=True)

    command = 'wavdetect scales="1 2 4 8 16 32" clobber=yes'
    proc = subprocess.run(command, shell=True)

    os.chdir(f"{current_path}")

    if verbose > 0:
        end_time = time.perf_counter()
        print(f"\tFinished processing")
        print(f"\tProcessing time: {end_time - start_time:.0f} seconds")

    return True


def search_data(obsid: str, window_size: float = 20.0, verbose: int = 0, data_path: str = DEFAULT_DATA_PATH):
    """
    ## Search the data for a given obsid.

    ### Args:
        obsid `str`: Obsid to search data for.
        window_size `float` (optional): Defaults to `20.0`. Size of the window to use for the search.
        verbose `int` (optional): Defaults to `0`. Level of verbosity.
        data_path `str` (optional): Defaults to `DEFAULT_DATA_PATH`. Path where all downloaded data is stored.
    """
    if verbose > 0:
        print(f"\tSearching")
        start_time = time.perf_counter()

    current_path = os.getcwd()

    command = f"cp auxiliary/search_algorithm.py {data_path}/{obsid}/"
    proc = subprocess.run(command, shell=True)

    os.chdir(f"{data_path}/{obsid}/")

    command = f"python search_algorithm.py {current_path} {window_size} {verbose}"
    proc = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
    if verbose > 1:
        print(proc.stdout)

    os.chdir(f"{current_path}")

    if verbose > 0:
        end_time = time.perf_counter()
        print(f'\tFinished searching')
        print(f'\tSearching time: {end_time - start_time:.0f} seconds')


def pipeline(obsid: str, window_size: float = 20.0, verbose: int = 0, data_path: str = DEFAULT_DATA_PATH) -> None:
    """
    ## Pipeline to download, process and search data for a given obsid.

    ### Args:
        obsid `str`: Obsid to search data for.
        window_size `float` (optional): Defaults to `20.0`. Window size to use for the search.
        verbose `int` (optional): Defaults to `0`. Level of verbosity.
        data_path `str` (optional): Defaults to `DEFAULT_DATA_PATH`. Path where all downloaded data is stored.
    """
    if verbose > 0:
        print(f'Starting obsid: {obsid}')
        start_time = time.perf_counter()

    # download data if not already downloaded
    dirs = os.listdir(f"{data_path}/")
    if obsid in dirs:
        if verbose > 0:
            print(f'\tObsid data already downloaded')
    else:
        download_data(obsid, verbose, data_path)
        if not process_data(obsid, verbose, data_path):
            # message printed in process_data
            return

    # analyse data if not already analysed
    analysed = pd.read_csv(f'output/analysed_w20.txt',
                           header=0, sep=' ', dtype=str)
    if obsid in analysed['ObsId'].values:
        if verbose > 0:
            print(f'\tObsid data already searched')
    else:
        search_data(obsid, window_size, verbose, data_path)

    if verbose > 0:
        end_time = time.perf_counter()
        print(f'Finished obsid: {obsid}')
        print(f'Time: {end_time - start_time:.0f} seconds')


def start_search(filenames: list, window_size: float = 20.0, data_path: str = DEFAULT_DATA_PATH, limit_observations: bool = False, verbose: int = 0) -> None:
    """
    ## Start the search for a list of filenames.

    ### Args:
        filenames `list`: List of filenames to search.
        window_size `float` (optional): Defaults to `20.0`. Window size to use for the search.
        data_path `str` (optional): Defaults to `DEFAULT_DATA_PATH`. Path where all downloaded data is stored.
        limit_observations `bool` (optional): Defaults to `False`. Whether to only search observations with an exposure time longer than window_size.
        verbose `int` (optional): Defaults to `0`. Level of verbosity.
    """
    obsids = read_obsids(filenames, ['Obs ID'])

    for i, obsid in obsids.iterrows():
        print(f"progress: {(i+1)/len(obsids) * 100:.2f}%")
        pipeline(obsid['Obs ID'], window_size, verbose, data_path)


DATA_PATH = "/data/jcrans/fxrt-data/obsids"
FILENAMES = [  # List of filenames to search
    'obsid_lists/obsids_b+10_220401+.csv',
    'obsid_lists/obsids_b-10_220401+.csv',
    'obsid_lists/obsids_b+10_220401-.csv',
    'obsid_lists/obsids_b-10_220401-.csv',
]
WINDOW_SIZE = 20.0  # The window size to use for the search
VERBOSE = 2  # Level of verbosity for the search functions

if __name__ == '__main__':
    # start_search(FILENAMES, WINDOW_SIZE, VERBOSE, DATA_PATH)
    pipeline('8490', WINDOW_SIZE, VERBOSE, DATA_PATH)

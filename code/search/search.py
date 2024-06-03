import subprocess
import numpy as np
import glob
import os
import time

DEFAULT_DATA_PATH = "/data/jcrans/fxrt-data/obsids"


def download_data(obsid: str, verbose: int = 0, data_path: str = DEFAULT_DATA_PATH):
    """
    Download the data for a given obsid.

    Args:
        obsid (str): Obsid to download data for.
        verbose (int, optional): Level of verbosity. Defaults to 0.
        data_path (str, optional): Path to store all downloaded data in. Defaults to DEFAULT_DATA_PATH.
    """
    if verbose > 0:
        print(f"\tDownloading")
        start_time = time.perf_counter()

    current_path = os.getcwd()
    os.chdir(f"{data_path}")

    command = f'download_chandra_obsid {obsid} evt2,fov,asol,bpix,msk'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    # print(proc.stdout)

    command = f'cp {obsid}/primary/* {obsid}/'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    command = f'gunzip {obsid}/*.gz'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    os.chdir(f"{current_path}")

    if verbose > 0:
        end_time = time.perf_counter()
        print(f"\tFinished downloading")
        print(f"\tDownload time: {end_time - start_time:.0f} seconds")


def process_data(obsid: str, verbose: int = 0, data_path: str = DEFAULT_DATA_PATH) -> bool:
    """
    Process the data for a given obsid.

    Args:
        obsid (str): Obsid to process data for.
        verbose (int, optional): Level of verbosity. Defaults to 0.
        data_path (str, optional): Path to store all downloaded data in. Defaults to DEFAULT_DATA_PATH.

    Returns:
        bool: Whether the data was processed successfully. Returns False if no data was found, True otherwise.
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
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    command = 'dmcopy "' + event_file[0] + \
        '[sky=region(s3.fov)]" 578_evt2_filtered.fits clobber=yes'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    command = 'fluximage 578_evt2_filtered.fits binsize=1 bands=broad outroot=s3 psfecf=0.393 clobber=yes'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    command = 'mkpsfmap s3_broad_thresh.img outfile=s3_psfmap.fits energy=1.4967 ecf=0.393 clobber=yes'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    # detection
    command = 'punlearn wavdetect'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    command = 'pset wavdetect infile=s3_broad_thresh.img mode=h'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    command = 'pset wavdetect psffile=s3_broad_thresh.psfmap mode=h'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    command = 'pset wavdetect expfile=s3_broad_thresh.expmap mode=h'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    command = 'pset wavdetect outfile=s3_expmap_src.fits mode=h'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    command = 'pset wavdetect scellfile=s3_expmap_scell.fits mode=h'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    command = 'pset wavdetect imagefile=s3_expmap_imgfile.fits mode=h'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    command = 'pset wavdetect defnbkgfile=s3_expmap_nbgd.fits mode=h'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    command = 'pset wavdetect regfile=s3_expmap_src.reg mode=h'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    command = 'wavdetect scales="1 2 4 8 16 32" clobber=yes'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    os.chdir(f"{current_path}")

    if verbose > 0:
        end_time = time.perf_counter()
        print(f"\tFinished processing")
        print(f"\tProcessing time: {end_time - start_time:.0f} seconds")

    return True


def search_data(obsid: str, window_size: float, verbose: int = 0, data_path: str = DEFAULT_DATA_PATH):
    """
    Search the data for a given obsid.

    Args:
        obsid (str): Obsid to search data for.
        window_size (float): Window size to use for the search.
        verbose (int, optional): Level of verbosity. Defaults to 0.
        data_path (str, optional): Path to store all downloaded data in. Defaults to DEFAULT_DATA_PATH.
    """
    if verbose > 0:
        print(f"\tSearching")
        start_time = time.perf_counter()

    current_path = os.getcwd()

    command = f"cp auxiliary/search_algorithm.py {data_path}/{obsid}/"
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    os.chdir(f"{data_path}/{obsid}/")

    command = f"python search_algorithm.py {current_path} {window_size}"
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    os.chdir(f"{current_path}")

    if verbose > 0:
        end_time = time.perf_counter()
        print(f'\tFinished searching')
        print(f'\tSearching time: {end_time - start_time:.0f} seconds')


def pipeline(obsid: str, window_size: float, verbose: int = 0, data_path: str = DEFAULT_DATA_PATH) -> None:
    """
    Pipeline function to download, process, and search data for a given obsid.

    Args:
        obsid (str): Obsid to search data for.
        window_size (float): Window size to use for the search.
        verbose (int, optional): Level of verbosity. Defaults to 0.
        data_path (str, optional): Path to store all downloaded data in. Defaults to DEFAULT_DATA_PATH.
    """
    if verbose > 0:
        print(f'Starting obsid: {obsid}')
        start_time = time.perf_counter()

    dirs = os.listdir(f"{data_path}/")
    if str(obsid) in dirs:
        if verbose > 0:
            print(f'\tObsid data already downloaded')
    else:
        download_data(obsid, verbose, data_path)
        if not process_data(obsid, verbose, data_path):
            # message printed in process_data
            return

    analysed = np.genfromtxt(
        f'output/analysed_w20.txt', dtype='str', delimiter='\n')
    if str(obsid) in analysed:
        if verbose > 0:
            print(f"\tObsid data already searched")
    else:
        search_data(obsid, window_size, verbose, data_path)

    if verbose > 0:
        end_time = time.perf_counter()
        print(f'Finished obsid: {obsid}')
        print(f'Time: {end_time - start_time:.0f} seconds')


def start_search(filenames: list, window_size: float, verbose: int = 0, data_path: str = DEFAULT_DATA_PATH) -> None:
    """
    Start a search for candidate detections in a list of filenames.

    Args:
        filenames (list): List of filenames to search.
        window_size (float): Window size to use for the search.
        verbose (int, optional): Level of verbosity. Defaults to 0.
        data_path (str, optional): Path to store all downloaded data in. Defaults to DEFAULT_DATA_PATH.
    """
    Obsids = []
    for filename in filenames:
        if verbose > 0:
            print(f"Reading {filename}")
        Obsids.extend(np.genfromtxt(
            f"{filename}", dtype='str', skip_header=1, delimiter=',', usecols=[1]
        ))

    Obsids = np.array(Obsids)

    for Obsid in Obsids:
        pipeline(Obsid, window_size, verbose, data_path)


DATA_PATH = "/data/jcrans/fxrt-data/obsids"
FILENAMES = [  # List of filenames to search
    'obsid_lists/obsids_b+10_220401-.csv',
    'obsid_lists/obsids_b-10_220401-.csv',
]
WINDOW_SIZE = 20.0  # The window size to use for the search
VERBOSE = 1  # Level of verbosity for the search functions

if __name__ == '__main__':
    start_search(FILENAMES, WINDOW_SIZE, VERBOSE, DATA_PATH)

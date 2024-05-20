import subprocess
import numpy as np
import glob
import os
import time

CURRENT_PATH = os.getcwd()
DATA_PATH = "/data/jcrans/fxrt-data"


def download_data(obsid: str, logging: bool):
    if logging:
        print(f"\tDownloading")
        start_time = time.perf_counter()

    os.chdir(f"{DATA_PATH}/obsids/")

    command = f'download_chandra_obsid {obsid} evt2,fov,asol,bpix,msk'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    command = f'cp {obsid}/primary/* {obsid}/'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    command = f'gunzip {obsid}/*.gz'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    os.chdir(f"{CURRENT_PATH}")

    if logging:
        end_time = time.perf_counter()
        print(f"\tFinished downloading")
        print(f"\tDownload time: {end_time - start_time:.0f} seconds")


def process_data(obsid: str, logging: bool):
    if logging:
        print(f"\tProcessing")
        start_time = time.perf_counter()

    os.chdir(f"{DATA_PATH}/obsids/{obsid}/")

    event_file = glob.glob('*evt2.fits', recursive=True)
    fov_file = glob.glob('*N*fov1.fits', recursive=True)

    # mask
    command = 'dmcopy "' + fov_file[0] + '" s3.fov'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    command = 'dmcopy "' + event_file[0] + \
        '[sky=region(s3.fov)]" 578_evt2_filtered.fits'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    command = 'fluximage 578_evt2_filtered.fits binsize=1 bands=broad outroot=s3 psfecf=0.393'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    command = 'mkpsfmap s3_broad_thresh.img outfile=s3_psfmap.fits energy=1.4967 ecf=0.393'
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

    command = 'wavdetect scales="1 2 4 8 16 32"'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    os.chdir(f"{CURRENT_PATH}")

    if logging:
        end_time = time.perf_counter()
        print(f"\tFinished processing")
        print(f"\tProcessing time: {end_time - start_time:.0f} seconds")


def search_data(obsid: str, logging: bool):
    if logging:
        print(f"\tSearching")
        start_time = time.perf_counter()

    command = f"cp search_data.py {DATA_PATH}/obsids/{obsid}/"
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    os.chdir(f"{DATA_PATH}/obsids/{obsid}/")

    command = f"python search_data.py {CURRENT_PATH}"
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    os.chdir(f"{CURRENT_PATH}")

    if logging:
        end_time = time.perf_counter()
        print(f'\tFinished searching')
        print(f'\tSearching time: {end_time - start_time:.0f} seconds')


def pipeline(obsid: str, logging: bool):
    if logging:
        print(f'Starting obsid: {obsid}')
        start_time = time.perf_counter()

    dirs = os.listdir(f"{DATA_PATH}/obsids/")
    if str(obsid) in dirs:
        print(f'\tObsid data already downloaded')
    else:
        download_data(obsid, logging)
        process_data(obsid, logging)

    analysed = np.genfromtxt(
        f'analysed_w20.txt', dtype='str', delimiter='\n')
    if str(obsid) in analysed:
        print(f"\tObsid data already searched")
    else:
        search_data(obsid, logging)

    if logging:
        end_time = time.perf_counter()
        print(f'Finished obsid: {obsid}')
        print(f'Time: {end_time - start_time:.0f} seconds')

# filenames = [
#     'obsids_2022-04-01+_Texp8+_b10+.csv',
#     'obsids_2022-04-01+_Texp8+_b-10+.csv',
# ]

# Obsids = []
# for filename in filenames:
#     Obsids.extend(np.genfromtxt(
#         filename, dtype='str', delimiter=',', skip_header=1, usecols=1
#     ))
# Obsids = np.array(Obsids)


Obsids = [
    803,
    2025,
    8490,
    9546,
    9548,
    14904,
    4062,
    5885,
    9841,
    12264,
    12884,
    13454,
    15113,
    16454,
]

for Obsid in Obsids:
    pipeline(Obsid, True)

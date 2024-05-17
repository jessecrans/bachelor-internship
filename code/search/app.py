import subprocess
import numpy as np
import glob
import os
import time


def download_data(obsid: str, logging: bool):
    if logging:
        print(f"\tDownloading")
        start_time = time.perf_counter()

    command = f'download_chandra_obsid {obsid} evt2,fov,asol,bpix,msk'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    command = f'cp search_data.py {obsid}/'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    command = f'cp {obsid}/primary/* {obsid}/'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    command = f'gunzip {obsid}/*.gz'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    # command = f'mkdir obsids/{obsid}'
    # proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    command = f'mv {obsid} obsids/'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    if logging:
        end_time = time.perf_counter()
        print(f"\tFinished downloading")
        print(f"\tDownload time: {end_time - start_time:.0f} seconds")


def process_data(obsid: str, logging: bool):
    if logging:
        print(f"\tProcessing")
        start_time = time.perf_counter()

    os.chdir(f'obsids/{obsid}/')
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

    os.chdir('../../')

    if logging:
        end_time = time.perf_counter()
        print(f"\tFinished processing")
        print(f"\tProcessing time: {end_time - start_time:.0f} seconds")


def search_data(obsid: str, logging: bool):
    if logging:
        print(f"\tSearching")
        start_time = time.perf_counter()

    command = 'python search_data.py'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    if logging:
        end_time = time.perf_counter()
        print(f'\tFinished searching')
        print(f'\tSearching time: {end_time - start_time:.0f} seconds')


def pipeline(obsid: str, logging: bool):
    if logging:
        print(f'Starting obsid: {obsid}')
        start_time = time.perf_counter()

    dirs = os.listdir('obsids/')
    if str(obsid) in dirs:
        print(f'\tObsid data already downloaded')

        # copy latest version of search_data.py to make sure it is up to date
        command = f'cp search_data.py obsids/{obsid}/'
        proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    else:
        download_data(obsid, logging)
        process_data(obsid, logging)

    analysed = np.genfromtxt(
        f'analysed.txt', dtype='str', delimiter='\n')
    if str(obsid) in analysed:
        print(f"\tObsid data already searched")
    else:
        os.chdir(f'obsids/{obsid}/')
        search_data(obsid, logging)
        os.chdir('../../')

    if logging:
        end_time = time.perf_counter()
        print(f'Finished obsid: {obsid}')
        print(f'Time: {end_time - start_time:.0f} seconds')


# Obsid = [803]
# Obsid=[23570,23831,24275,24276,24669,23728,23696,23635,24835,22542,23556,23631,23699,23695,23691,26139,23749,23568,23557,23572]
# Obsid=[26146,23812,26150,23740,23741,23394,23770,26175,23692,23725,26197,23744,23832,24472,24473,24474,24836,24837,24844]
# Obsid=[26137,26168,23719,26183,26166,26148,26135,23688,26186,26151,26145,23627,26147,23686,26159,26162,23705,26229,26286,22616,22617,22974,22975,22976,24892,20611,26156,23793]
Obsid = [
    803,
    2025,
    8490,
    9546,
    9548,
    14904
]

for i in range(0, len(Obsid)):
    pipeline(Obsid[i], True)

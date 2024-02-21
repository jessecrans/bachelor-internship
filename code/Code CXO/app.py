import subprocess
import numpy as np
import re
import glob
import os

from flask import Flask, render_template
from flask import request

app = Flask(__name__)


@app.route("/")
def index():
    obsid = request.args.get("obsid", "")
    if obsid:
        FXRTs = function(obsid)
    else:
        FXRTs = ""
    return (
        """<form action="" method="get">
                ObsId: <input type="text" name="obsid">
                <input type="submit" value="Apply our method">
            </form>"""
        + "FXRTs (ObsId, RA, DEC, ERR, SIG): "
        + FXRTs
    )


def function(obsid):
    command = 'download_chandra_obsid ' + str(obsid) + ' evt2,fov,asol,bpix,msk'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    # command='cd '+str(obsid)
    # proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    command = 'cp search_data.py ' + str(obsid) + '/'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    command = 'cp '+str(obsid) + '/primary/* ' + str(obsid) + '/'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
    command = 'gunzip ' + str(obsid) + '/*.gz'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    os.chdir(str(obsid) + '/')
    event_file = glob.glob('*evt2.fits', recursive=True)
    fov_file = glob.glob('*N*fov1.fits', recursive=True)
    print(event_file, fov_file)

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

    # Algorithm
    command = 'python search_data.py'
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    os.chdir('../')
    command = 'rm -rf ' + str(obsid)
    proc = subprocess.run(command, stdout=subprocess.PIPE, shell=True)

    return


# if __name__ == "__main__":
#    app.run(host="0.0.0.0", port=8080, debug=False)
Obsid = [16454, 16453, 803]
# Obsid=[23570,23831,24275,24276,24669,23728,23696,23635,24835,22542,23556,23631,23699,23695,23691,26139,23749,23568,23557,23572]
# Obsid=[26146,23812,26150,23740,23741,23394,23770,26175,23692,23725,26197,23744,23832,24472,24473,24474,24836,24837,24844]
# Obsid=[26137,26168,23719,26183,26166,26148,26135,23688,26186,26151,26145,23627,26147,23686,26159,26162,23705,26229,26286,22616,22617,22974,22975,22976,24892,20611,26156,23793]
for i in range(0, len(Obsid)):
    print(Obsid[i])
    function(Obsid[i])

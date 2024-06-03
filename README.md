# Bachelor Internship
This repository is for organizing all code to do with my bachelor internship.

## code/search
In the `code/search` directory there are python files which can be run to analyse Chandra data.

### search.py

With this script we can look for candidate detections in Chandra observations using the window search algorithm. 

At the bottom of the script there are some parameters in uppercase which can be set depending the specific task.

- `DATA_PATH`: Where to store the downloaded Chandra data. Make sure this directory has enough space.
- `FILENAMES`: A list of Obsids obtained from the [Chandra Archive](https://cda.harvard.edu/chaser/mainEntry) using the options mentioned below. These options give the best observations to detect FXTs. The suggested options leave out the Galactic Disk but are not strictly necessary.
  - `Status: Archived`
  - `Instrument: ACIS`
  - `Grating: None`
  - `Exposure Mode: ACIS TE`
  - Suggested options:
    - `Exposure Time (ks): 8-`
    - `Range Search:`
      - `RA (0, 360)`
      - `DEC(_, -10) or DEC(10, _)`
      - `Coord System: Galactic`
    - `Public Release Date`: The bigger the range the more observations there will be, so mind space for data and time for execution.
- `WINDOW_SIZE`: Size of the window in the window search algorithm.
- `VERBOSE`: Level of verbosity
  - `0`: No print messages.
  - `1`: Progress messages for the search pipeline. Number of detections being filtered from what file.
  - `2`: Filter status of every filtered detection.
  - `3`: Progress messages for filter setup code.

### filter.py

With this script we can filter the candidate detections based on if their appearance in known catalogs.

At the bottom of the script there are some parameters in uppercase which can be set depending on the specific task.

- `DETECTIONS_FILENAME`: File name of the candidate detections found by the `search.py` script. Normally, this filename should be of the form: `output/detections_w{WINDOW_SIZE}.txt`. Where `WINDOW_SIZE` is then the same as chosen when running `search.py`.
- `FILTERED_FILENAME`: File name of filtered candidate detections. This is the output file for this script. Normally, this filename should be of the form: `output/filtered_w{WINDOW_SIZE}.txt`. Where `WINDOW_SIZE` is then the same as chosen when running `search.py`.
- `CATALOGS`: All catalogs that can be filtered on. Simply (un)comment catalogs to decide which ones are taken into account in the filtering.
- `VERBOSE`: Level of verbosity
  - `0`: No print messages.
  - `1`: Progress messages for the search pipeline. Number of detections being filtered from what file.
  - `2`: Filter status of every filtered detection.
  - `3`: Progress messages for filter setup code.

### code/simulation
In this directory we can run simulations for modeled FXTs. With these we can simulate the detection probabilities for different FXTs and window sizes.
# Bachelor Internship
This repository is for organizing all code to do with my bachelor internship.

### code/search
In the code/search directory there is a main.py file. With this file we can search through given chandra obsids by running this file in the command line. Below is explained the way to do this.

```
python main.py [action] [filenames] [window_size] [verbose]
```

- `action (str)`: This parameter decides which action to perform on the given filenames.
    - `search`: Will perform the search algorithm on the given filenames with the given window_size and given verbosity level.
    - `filter`: Will perform the candidate filtering on the output detections with the given window size and given verbosity level.
    - `all`: Will perform all of the above.
- `filenames (list[str])`: Is a list of filenames obtained from the [Chandra Archive](https://cda.harvard.edu/chaser/mainEntry) using the options mentioned below. These options give the best observations to detect FXTs. The suggested options leave out the Galactic Disk but are not strictly necessary.
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
- `window_size (float)`: This parameter decides the window size with which to perform the search algorithm. It also decides which files to perform the filtering on.
- `verbose (int)`:
  - `0`: No print messages.
  - `1`: Progress messages for the search pipeline. Number of detections being filtered from what file.
  - `2`: Filter status of every filtered detection.
  - `3`: Progress messages for filter setup code.

### code/simulation
In this directory we can run simulations for modeled FXTs. With these we can simulate the detection probabilities for different FXTs and window sizes.
import sys
from search import start_search
from filter import filter_detection_file

# TODO: To make this work on the server copy the latest analysed detections and filtered files to the output folder.
# TODO: Do this only when the server is done searching the entire archive.


def main():
    # sys.argv[]
    # 0: main.py
    # 1: actions: 'search' or 'filter' or 'both'
    # 2: window_size: float
    # 3: verbose: 'True' or 'False'
    # 4+: filenames: list of filenames

    action = sys.argv[1]
    window_size = float(sys.argv[2])
    verbose = sys.argv[3] == 'True'
    filenames = sys.argv[4:]

    if action == 'search' or action == 'both':
        start_search(filenames, window_size, verbose)

    if action == 'filter' or action == 'both':
        filter_detection_file(
            f'output/detections_w{window_size}.txt',
            f'output/filtered_w{window_size}.csv',
            verbose
        )


if __name__ == '__main__':
    main()

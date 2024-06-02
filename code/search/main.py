import sys
from search import start_search
from filter import filter_detection_file
import os


def main(action: str, window_size: float, verbose: int, filenames: list[str]):
    """
    Main function to run the search and filter functions. 
    Search the given filenames for candidate detections and filter the complete list of detections.

    Args:
        action (str): The action to perform. Either 'search', 'filter', or 'all'.
        window_size (float): The window size to use for the search.
        verbose (int): Level of verbosity for the search and filter functions.
        filenames (list[str]): List of filenames to search.
    """

    if action == 'search' or action == 'all':
        if filenames == ['default']:
            filenames = os.listdir('obsid_lists')

        start_search(filenames, window_size, verbose)

    if action == 'filter' or action == 'all':
        filter_detection_file(
            f'output/detections_w{int(window_size)}.txt',
            f'output/filtered_w{int(window_size)}.csv',
            verbose=verbose
        )


if __name__ == '__main__':

    main(
        sys.argv[1],
        float(sys.argv[-2]),
        int(sys.argv[-1]),
        sys.argv[2:-2]
    )

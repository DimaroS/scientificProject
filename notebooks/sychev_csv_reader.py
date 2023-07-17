"""Lightweight one-line-at-a-time CSV-reader.

Not yet implemented.
"""

import csv
from typing import List


# TODO: replace with per-line loading
def load_float_csv(filename: str) -> List[List[float]]:
    """One-line-at-a-time read should allow to read GB-sized files.

    Args:
        filename: String representing path to the csv-file.

    Returns:
        List of lists representing rows of the file and containing
        converted to floating point values of data from file.
    """
    rows = []
    with open(filename, newline="") as csvfile:
        for row in csv.reader(csvfile, delimiter=" ", quotechar="|"):
            rows.append([float(item) for item in row])
    return rows


def load_str_csv(filename: str) -> List[List[str]]:
    """One-line-at-a-time read should allow to read GB-sized files.

    Args:
        filename: String representing path to the csv-file.

    Returns:
        List of lists representing rows of the file and containing
        converted values of data from file as strings.
    """
    rows = []
    with open(filename, newline="") as csvfile:
        for row in csv.reader(csvfile, delimiter=" ", quotechar="|"):
            rows.append(row)
    return rows

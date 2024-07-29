import configparser
import itertools
import re
from typing import Sequence, TextIO

# Rainfall amount lines look like: "3600    0.0000019756"
_RAINFALL_AMOUNT_RE = re.compile(r"\d+\s+(\d+\.\d+)")


def read_rainfall_amounts(rain_fd: TextIO) -> Sequence[float]:
    """Returns a list of rainfall amounts at each timestep in the CityCAT config.

    Args:
      rain_fd: The file containing the CityCAT rainfall configuration.

    Returns:
      The rainfall pattern as a sequence of rainfall amounts.
    """
    entries = []
    for line in rain_fd:
        rainfall_line = _RAINFALL_AMOUNT_RE.match(line)
        if rainfall_line is None:
            continue
        entries.append(float(rainfall_line.group(1)))
    return entries


def read_key_value_pairs(fd: TextIO) -> dict:
    """Reads a config file and returns a dictionary of k/v pairs.

    Example format of file contents:
    percentile = 99
    simulation_year = 2012
    simulation_months = JJA

    Args:
      fd: The file containing the key/value pairs

    Returns:
      A dictionary of the key/values pairs in the file
    """
    # Read lines from file first. If the first line is not blank and doesn't look
    # like a section header, add [DEAULT] section so it is compliant with
    # what ConfigParser expects
    config = configparser.ConfigParser()
    config.read_file(itertools.chain(["[DEFAULT]\n"], fd))
    config_dict = {key: value for key, value in config["DEFAULT"].items()}
    return config_dict

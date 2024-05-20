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

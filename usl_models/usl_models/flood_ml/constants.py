"""Constant definitions for Flood CNN."""

# Geospatial constants
GEO_FEATURES = 8
MAP_HEIGHT = 1000
MAP_WIDTH = 1000

# Temporal parameters. May be tuned.
N_FLOOD_MAPS = 5
M_RAINFALL = 6
MAX_RAINFALL_DURATION = 864  # (60 / 5) * 24 * 3

"""Utilities for reading the outputs of physics-based simulations."""

from typing import TextIO

import numpy
from numpy.typing import NDArray

from usl_lib.shared import geo_data


def read_city_cat_result_as_raster(
    fd: TextIO, geo_header: geo_data.ElevationHeader
) -> NDArray[numpy.float32]:
    """Rasterizes an rsl output file produce by CityCAT as a numpy array.

    Args:
        fd: The rsl result with predicted flood depths file to read.
        geo_header: A header describing the geography to rasterize the file into.

    Returns:
        The predicted flood depths rasterized as a numpy array.
    """
    sim_outputs = numpy.loadtxt(
        fd,
        skiprows=1,
        dtype=numpy.float32,
        usecols=(0, 1, 2),  # We just need the first 'XCen, YCen, Depth' columns.
    )

    # Convert the X/Y coordinates from the results to raster indices.
    fwd_transform = geo_header.forward_transform()
    indices = numpy.apply_along_axis(
        lambda row: fwd_transform * row, 1, sim_outputs[:, :2]
    )
    # Round the indices into integers.
    indices = numpy.rint(indices).astype(numpy.int64)

    # The simulation will not predict values for every raster cell. Set empty to 0.
    depth = numpy.zeros(
        (geo_header.row_count, geo_header.col_count), dtype=numpy.float32
    )
    # Set the simulation outputs where we have them.
    depth[indices[:, 1], indices[:, 0]] = sim_outputs[:, 2]

    return depth

import typing

import numpy

from usl_pipeline.shared.geo_data import Elevation


def write_to_esri_ascii_raster_file(elevation: Elevation, file: typing.TextIO) -> None:
    """Writes elevation data to a text file in Esri ASCII format.

    Args:
        elevation: Elevation data.
        file: Output file/stream to write data to.
    """
    if elevation.data is None:
        raise ValueError("Elevation data must be present")

    elv_header = elevation.header
    file.write("ncols {}\n".format(elv_header.col_count))
    file.write("nrows {}\n".format(elv_header.row_count))
    file.write("xllcorner {}\n".format(elv_header.x_ll_corner))
    file.write("yllcorner {}\n".format(elv_header.y_ll_corner))
    file.write("cellsize {}\n".format(elv_header.cell_size))
    file.write("NODATA_value {}\n".format(elv_header.nodata_value))

    numpy.savetxt(file, elevation.data, delimiter=" ", fmt="%s")

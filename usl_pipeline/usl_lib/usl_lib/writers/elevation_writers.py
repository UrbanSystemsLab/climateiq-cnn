import typing

import numpy

from usl_lib.shared import geo_data


def write_header_to_esri_ascii_raster_file(
    header: geo_data.ElevationHeader, file: typing.TextIO
) -> None:
    """Writes elevation header to a text file stream in Esri ASCII format.

    Args:
        header: Elevation header.
        file: Output file/stream to write data to.
    """
    file.write("ncols {}\n".format(header.col_count))
    file.write("nrows {}\n".format(header.row_count))
    file.write("xllcorner {}\n".format(header.x_ll_corner))
    file.write("yllcorner {}\n".format(header.y_ll_corner))
    file.write("cellsize {}\n".format(header.cell_size))
    file.write("NODATA_value {}\n".format(header.nodata_value))


def write_to_esri_ascii_raster_file(
    elevation: geo_data.Elevation, file: typing.TextIO
) -> None:
    """Writes elevation data to a text file in Esri ASCII format.

    Args:
        elevation: Elevation data.
        file: Output file/stream to write data to.
    """
    if elevation.data is None:
        raise ValueError("Elevation data must be present")

    write_header_to_esri_ascii_raster_file(elevation.header, file)
    numpy.savetxt(file, elevation.data, delimiter=" ", fmt="%s")

import typing

from usl_pipeline.shared.geo_data import Elevation


def write_to_esri_ascii_raster_file(elevation: Elevation, file: typing.TextIO) -> None:
    """Writes elevation data to a text file in Esri ASCII format.

    Args:
        elevation: Elevation data.
        file: Output file/stream to write data to.
    """
    elv_header = elevation.header
    file.write("ncols {}\n".format(elv_header.col_count))
    file.write("nrows {}\n".format(elv_header.row_count))
    file.write("xllcorner {}\n".format(elv_header.x_ll_corner))
    file.write("yllcorner {}\n".format(elv_header.y_ll_corner))
    file.write("cellsize {}\n".format(elv_header.cell_size))
    file.write("NODATA_value {}\n".format(elv_header.nodata_value))

    if elevation.data is not None:
        elv_data = elevation.data.tolist()
        for row_index in range(0, elv_header.row_count):
            row_values = elv_data[row_index]
            string_values = list(map(lambda e: str(e), row_values))
            file.write("{}\n".format(" ".join(string_values)))

from io import StringIO

import numpy

from usl_pipeline.writers.elevation_writers import write_to_esri_ascii_raster_file
from usl_pipeline.shared.geo_data import Elevation, ElevationHeader


def test_write_to_esri_ascii_raster_file():
    header = ElevationHeader(
        col_count=3,
        row_count=2,
        x_ll_corner=0.0,
        y_ll_corner=4.0,
        cell_size=2.0,
        nodata_value=0.0,
    )
    data = numpy.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0123456789]])
    elevation = Elevation(header=header, data=data)
    buffer = StringIO()
    write_to_esri_ascii_raster_file(elevation, buffer)
    assert buffer.getvalue() == (
        "ncols 3\n"
        "nrows 2\n"
        "xllcorner 0.0\n"
        "yllcorner 4.0\n"
        "cellsize 2.0\n"
        "NODATA_value 0.0\n"
        "0.0 1.0 2.0\n"
        "3.0 4.0 5.0123456789\n"
    )

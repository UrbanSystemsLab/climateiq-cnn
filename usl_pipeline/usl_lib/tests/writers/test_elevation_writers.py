import pathlib
import tempfile
from io import StringIO

import numpy
from numpy import testing
import rasterio

from usl_lib.readers import elevation_readers
from usl_lib.writers import elevation_writers
from usl_lib.shared import geo_data


def test_write_to_esri_ascii_raster_file():
    header = geo_data.ElevationHeader(
        col_count=3,
        row_count=2,
        x_ll_corner=0.0,
        y_ll_corner=4.0,
        cell_size=2.0,
        nodata_value=0.0,
    )
    data = numpy.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0123456789]])
    elevation = geo_data.Elevation(header=header, data=data)
    buffer = StringIO()
    elevation_writers.write_to_esri_ascii_raster_file(elevation, buffer)
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


def test_write_to_geotiff():
    with tempfile.TemporaryDirectory() as temp_dir:
        elevation_file_path = pathlib.Path(temp_dir) / "elevation.tif"
        header = geo_data.ElevationHeader(
            col_count=3,
            row_count=2,
            x_ll_corner=0.0,
            y_ll_corner=4.0,
            cell_size=2.0,
            nodata_value=0.0,
        )
        data = [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        elevation_writers.write_to_geotiff(
            geo_data.Elevation(header=header, data=numpy.asarray(data)),
            elevation_file_path,
        )

        # Test what was written:
        with open(elevation_file_path, "rb") as input_fd:
            elevation2 = elevation_readers.read_from_geotiff(input_fd)
        assert elevation2.header == header
        testing.assert_array_equal(elevation2.data, data)


def test_write_elevation_header_to_json():
    header = geo_data.ElevationHeader(
        col_count=3,
        row_count=4,
        x_ll_corner=10.0,
        y_ll_corner=20.0,
        cell_size=1.0,
        nodata_value=-30.0,
        crs=rasterio.CRS({"init": "EPSG:4326"}),
    )
    buffer = StringIO()
    elevation_writers.write_header_to_json_file(header, buffer)
    assert buffer.getvalue() == (
        "{\n"
        + '    "col_count": 3,\n'
        + '    "row_count": 4,\n'
        + '    "x_ll_corner": 10.0,\n'
        + '    "y_ll_corner": 20.0,\n'
        + '    "cell_size": 1.0,\n'
        + '    "nodata_value": -30.0,\n'
        + '    "crs": "EPSG:4326"\n'
        + "}"
    )

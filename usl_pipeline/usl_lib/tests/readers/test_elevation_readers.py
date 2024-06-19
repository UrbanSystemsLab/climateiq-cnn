import io

import numpy
from numpy import testing
import rasterio
import rasterio.io

from usl_lib.readers import elevation_readers
from usl_lib.shared import geo_data


def prepare_test_geotiff_elevation_memory_file(mem_file: rasterio.io.MemoryFile):
    tiff_array = numpy.array(
        [
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
        ]
    )
    with mem_file.open(
        driver="GTiff",
        dtype=rasterio.float32,
        nodata=0.0,
        width=3,
        height=2,
        count=1,
        crs=rasterio.CRS.from_epsg(32618),
        transform=rasterio.Affine(2.0, 0.0, 100.0, 0.0, -2.0, 500.0),
    ) as raster:
        raster.write(tiff_array.astype(rasterio.float32), 1)


def test_load_elevation_from_geotiff_default_no_data():
    with rasterio.io.MemoryFile() as mem_file:
        prepare_test_geotiff_elevation_memory_file(mem_file)
        elevation = elevation_readers.read_from_geotiff(mem_file)
        assert elevation.header == geo_data.ElevationHeader(
            col_count=3,
            row_count=2,
            x_ll_corner=100.0,
            y_ll_corner=496.0,
            cell_size=2.0,
            nodata_value=0.0,
            crs=rasterio.CRS({"init": "EPSG:32618"}),
        )
        testing.assert_array_equal(
            elevation.data,
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
            ],
        )
        assert not elevation.header.crs.is_geographic


def test_load_elevation_from_geotiff_with_changed_no_data():
    with rasterio.io.MemoryFile() as mem_file:
        prepare_test_geotiff_elevation_memory_file(mem_file)
        elevation = elevation_readers.read_from_geotiff(mem_file, no_data_value=-9999.0)
        assert elevation.header == geo_data.ElevationHeader(
            col_count=3,
            row_count=2,
            x_ll_corner=100.0,
            y_ll_corner=496.0,
            cell_size=2.0,
            nodata_value=-9999.0,
            crs=rasterio.CRS({"init": "EPSG:32618"}),
        )
        testing.assert_array_equal(
            elevation.data,
            [
                [-9999.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
            ],
        )


def test_load_elevation_from_esri_ascii_default_no_data():
    with io.StringIO(
        "\n".join(
            (
                "ncols 3",
                "nrows 2",
                "xllcorner 100.0",
                "yllcorner 496.0",
                "cellsize 2.0",
                "NODATA_value 0.0",
                "0.0 1.0 2.0",
                "3.0 4.0 5.0",
            )
        )
    ) as esri_file:
        elevation = elevation_readers.read_from_esri_ascii(esri_file)
        assert elevation.header == geo_data.ElevationHeader(
            col_count=3,
            row_count=2,
            x_ll_corner=100.0,
            y_ll_corner=496.0,
            cell_size=2.0,
            nodata_value=0.0,
        )
        testing.assert_array_equal(
            elevation.data,
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
            ],
        )


def test_load_elevation_from_esri_ascii_with_changed_no_data():
    with io.StringIO(
        "\n".join(
            (
                "ncols 3",
                "nrows 2",
                "xllcorner 100.0",
                "yllcorner 496.0",
                "cellsize 2.0",
                "NODATA_value 0.0",
                "0.0 1.0 2.0",
                "3.0 4.0 5.0",
            )
        )
    ) as esri_file:
        elevation = elevation_readers.read_from_esri_ascii(
            esri_file, no_data_value=-9999.0
        )
        assert elevation.header == geo_data.ElevationHeader(
            col_count=3,
            row_count=2,
            x_ll_corner=100.0,
            y_ll_corner=496.0,
            cell_size=2.0,
            nodata_value=-9999.0,
        )
        testing.assert_array_equal(
            elevation.data,
            [
                [-9999.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
            ],
        )


def test_load_elevation_from_esri_ascii_with_missing_nodata_header():
    """Ensures we interpret an absent NO_DATA header as a NO_DATA value of -9999."""
    with io.StringIO(
        "\n".join(
            (
                "ncols 3",
                "nrows 2",
                "xllcorner 100.0",
                "yllcorner 496.0",
                "cellsize 2.0",
                "0.0 1.0 2.0",
                "3.0 4.0 5.0",
            )
        )
    ) as esri_file:
        elevation = elevation_readers.read_from_esri_ascii(
            esri_file, no_data_value=-9999.0
        )
        assert elevation.header == geo_data.ElevationHeader(
            col_count=3,
            row_count=2,
            x_ll_corner=100.0,
            y_ll_corner=496.0,
            cell_size=2.0,
            nodata_value=-9999.0,
        )
        testing.assert_array_equal(
            elevation.data,
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
            ],
        )


def test_read_elevation_header_from_json():
    with io.StringIO(
        "{\n"
        + '    "col_count": 3,\n'
        + '    "row_count": 4,\n'
        + '    "x_ll_corner": 10.0,\n'
        + '    "y_ll_corner": 20.0,\n'
        + '    "cell_size": 1.0,\n'
        + '    "nodata_value": -30.0,\n'
        + '    "crs": "EPSG:32618"\n'
        + "}"
    ) as json_file:
        assert elevation_readers.read_header_from_json_file(
            json_file
        ) == geo_data.ElevationHeader(
            col_count=3,
            row_count=4,
            x_ll_corner=10.0,
            y_ll_corner=20.0,
            cell_size=1.0,
            nodata_value=-30.0,
            crs=rasterio.CRS({"init": "EPSG:32618"}),
        )

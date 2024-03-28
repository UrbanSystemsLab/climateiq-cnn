import io

import numpy
from numpy.testing import assert_array_equal
import rasterio
from rasterio import Affine, CRS
from rasterio.io import MemoryFile

from usl_pipeline.readers.elevation import read_from_geotiff, read_from_esri_ascii
from usl_pipeline.shared.geo_data import ElevationHeader


def prepare_test_geotiff_elevation_memory_file(mem_file: MemoryFile):
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
        crs=CRS.from_epsg(32618),
        transform=Affine(2.0, 0.0, 100.0, 0.0, -2.0, 500.0),
    ) as raster:
        raster.write(tiff_array.astype(rasterio.float32), 1)


def test_load_elevation_from_geotiff_default_no_data():
    with rasterio.io.MemoryFile() as mem_file:
        prepare_test_geotiff_elevation_memory_file(mem_file)
        elevation = read_from_geotiff(mem_file)
        assert elevation.header == ElevationHeader(
            col_count=3,
            row_count=2,
            x_ll_corner=100.0,
            y_ll_corner=496.0,
            cell_size=2.0,
            nodata_value=0.0,
            crs=CRS({"init": "EPSG:32618"}),
        )
        assert_array_equal(
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
        elevation = read_from_geotiff(mem_file, no_data_value=-9999.0)
        assert elevation.header == ElevationHeader(
            col_count=3,
            row_count=2,
            x_ll_corner=100.0,
            y_ll_corner=496.0,
            cell_size=2.0,
            nodata_value=-9999.0,
            crs=CRS({"init": "EPSG:32618"}),
        )
        assert_array_equal(
            elevation.data,
            [
                [-9999.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
            ],
        )


def prepare_test_esri_ascii_file_wrapper():
    lines = [
        "ncols 3",
        "nrows 2",
        "xllcorner 100.0",
        "yllcorner 496.0",
        "cellsize 2.0",
        "NODATA_value 0.0",
        "0.0 1.0 2.0",
        "3.0 4.0 5.0",
    ]
    reader = io.BufferedReader(io.BytesIO("\n".join(lines).encode("utf-8")))
    return io.TextIOWrapper(reader)


def test_load_elevation_from_esri_ascii_default_no_data():
    with prepare_test_esri_ascii_file_wrapper() as file:
        elevation = read_from_esri_ascii(file)
        assert elevation.header == ElevationHeader(
            col_count=3,
            row_count=2,
            x_ll_corner=100.0,
            y_ll_corner=496.0,
            cell_size=2.0,
            nodata_value=0.0,
        )
        assert_array_equal(
            elevation.data,
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
            ],
        )


def test_load_elevation_from_esri_ascii_with_changed_no_data():
    with prepare_test_esri_ascii_file_wrapper() as file:
        elevation = read_from_esri_ascii(file, no_data_value=-9999.0)
        assert elevation.header == ElevationHeader(
            col_count=3,
            row_count=2,
            x_ll_corner=100.0,
            y_ll_corner=496.0,
            cell_size=2.0,
            nodata_value=-9999.0,
        )
        assert_array_equal(
            elevation.data,
            [
                [-9999.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
            ],
        )

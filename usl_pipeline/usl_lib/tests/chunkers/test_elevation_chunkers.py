import tempfile

import numpy
from numpy import testing
import rasterio

from usl_lib.readers import elevation_readers
from usl_lib.chunkers import elevation_chunkers
from usl_lib.shared import geo_data
from usl_lib.shared import suppliers


def prepare_test_geotiff_elevation_memory_file(file_path: str) -> None:
    tiff_array = numpy.array(
        [
            [0.0, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
        ]
    )
    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        dtype=rasterio.float32,
        nodata=0.0,
        width=3,
        height=3,
        count=1,
        crs=rasterio.CRS.from_epsg(32618),
        transform=rasterio.Affine(2.0, 0.0, 10.0, 0.0, -2.0, 20.0),
    ) as raster:
        raster.write(tiff_array.astype(rasterio.float32), 1)


def assert_chunk_data_equal(
    chunk_path_generator: suppliers.ChunkFilePathGenerator,
    y_chunk_index: int,
    x_chunk_index: int,
    expected_header: geo_data.ElevationHeader,
    expected_data: list,
) -> None:
    file_path = chunk_path_generator.generate(y_chunk_index, x_chunk_index)
    with open(file_path, "rb") as input_file:
        elevation = elevation_readers.read_from_geotiff(input_file)
    assert elevation.header == expected_header
    testing.assert_array_equal(
        elevation.data,  # type: ignore
        expected_data,
    )


def test_split_geotiff_into_chunks() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        chunk_file_path_generator = suppliers.PatternBasedChunkFilePathGenerator(
            temp_dir, "chunk_{0}_{1}_elevation.tif"
        )
        with tempfile.NamedTemporaryFile() as temp_file:
            input_file_path = temp_file.name
            prepare_test_geotiff_elevation_memory_file(input_file_path)
            (y_chunk_count, x_chunk_count) = (
                elevation_chunkers.split_geotiff_into_chunks(
                    input_file_path, 2, chunk_file_path_generator
                )
            )

            assert y_chunk_count == 2
            assert x_chunk_count == 2
            # Chunk: y=0, x=0
            assert_chunk_data_equal(
                chunk_file_path_generator,
                0,
                0,
                geo_data.ElevationHeader(
                    col_count=2,
                    row_count=2,
                    x_ll_corner=10.0,
                    y_ll_corner=16.0,
                    cell_size=2.0,
                    nodata_value=0.0,
                    crs=rasterio.CRS({"init": "EPSG:32618"}),
                ),
                [[0.0, 1.0], [3.0, 4.0]],
            )

            # Chunk: y=0, x=1
            assert_chunk_data_equal(
                chunk_file_path_generator,
                0,
                1,
                geo_data.ElevationHeader(
                    col_count=1,
                    row_count=2,
                    x_ll_corner=14.0,
                    y_ll_corner=16.0,
                    cell_size=2.0,
                    nodata_value=0.0,
                    crs=rasterio.CRS({"init": "EPSG:32618"}),
                ),
                [[2.0], [5.0]],
            )

            # Chunk: y=1, x=0
            assert_chunk_data_equal(
                chunk_file_path_generator,
                1,
                0,
                geo_data.ElevationHeader(
                    col_count=2,
                    row_count=1,
                    x_ll_corner=10.0,
                    y_ll_corner=14.0,
                    cell_size=2.0,
                    nodata_value=0.0,
                    crs=rasterio.CRS({"init": "EPSG:32618"}),
                ),
                [[6.0, 7.0]],
            )

            # Chunk: y=1, x=1
            assert_chunk_data_equal(
                chunk_file_path_generator,
                1,
                1,
                geo_data.ElevationHeader(
                    col_count=1,
                    row_count=1,
                    x_ll_corner=14.0,
                    y_ll_corner=14.0,
                    cell_size=2.0,
                    nodata_value=0.0,
                    crs=rasterio.CRS({"init": "EPSG:32618"}),
                ),
                [[8.0]],
            )

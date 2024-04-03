import tempfile

import numpy
from numpy import testing
from osgeo import gdal
import rasterio

from usl_pipeline.readers import elevation_readers
from usl_pipeline.chunkers import elevation_chunkers
from usl_pipeline.shared import geo_data
from usl_pipeline.shared import suppliers


def prepare_test_geotiff_elevation_memory_file(file_path: str):
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


class MemoryChunkPathGenerator(suppliers.ChunkFilePathGenerator):

    def __init__(self, memory_root: str):
        """Test implementation for ChunkFilePathGenerator based on memory files.

        Args:
            memory_root: Root folder name for in-memory files.
        """
        self.memory_root = memory_root

    def generate(self, y_chunk_index: int, x_chunk_index: int) -> str:
        """Test implementation for a method generating path for in-memory chunk."""
        # Path in memory FS that GDAL library can work with
        return "/vsimem/{}/chunk_{}_{}.tif".format(
            self.memory_root, y_chunk_index, x_chunk_index
        )


def get_chunk_elevation_from_memory(
    chunk_path_generator: MemoryChunkPathGenerator,
    y_chunk_index: int,
    x_chunk_index: int,
):
    file_path = chunk_path_generator.generate(y_chunk_index, x_chunk_index)
    stat = gdal.VSIStatL(file_path, gdal.VSI_STAT_SIZE_FLAG)
    # open memory file
    vsi_file = gdal.VSIFOpenL(file_path, "r")
    # read entire contents
    byte_content = gdal.VSIFReadL(1, stat.size, vsi_file)
    # write bytes to temporary file in order for rasterio library to read it
    with tempfile.NamedTemporaryFile() as tmpfile:
        with open(tmpfile.name, "wb") as output_file:
            output_file.write(byte_content)
        with open(tmpfile.name, "rb") as input_file:
            return elevation_readers.read_from_geotiff(input_file)


def check_chunk_data(
    chunk_path_generator: MemoryChunkPathGenerator,
    y_chunk_index: int,
    x_chunk_index: int,
    expected_header: geo_data.ElevationHeader,
    expected_data: list,
):
    elevation = get_chunk_elevation_from_memory(
        chunk_path_generator, y_chunk_index, x_chunk_index
    )
    assert elevation.header == expected_header
    testing.assert_array_equal(
        elevation.data,
        expected_data,
    )


def test_load_elevation_from_geotiff_default_no_data():
    chunk_file_path_generator = MemoryChunkPathGenerator("test01")
    with tempfile.NamedTemporaryFile() as temp_file:
        input_file_path = temp_file.name
        prepare_test_geotiff_elevation_memory_file(input_file_path)
        chunker = elevation_chunkers.GeoTiffChunker(input_file_path, 2)
        chunker.split_into_chunks(chunk_file_path_generator)

        # Chunk: y=0, x=0
        check_chunk_data(
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
        check_chunk_data(
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
        check_chunk_data(
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
        check_chunk_data(
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

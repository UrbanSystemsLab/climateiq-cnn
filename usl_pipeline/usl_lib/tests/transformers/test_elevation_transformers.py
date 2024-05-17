import io

import numpy
import rasterio
import rasterio.io
from shapely import geometry

from usl_lib.transformers import elevation_transformers


def prepare_test_geotiff_elevation_memory_file(
    data: list[list[float]],
    mem_file: rasterio.io.MemoryFile,
) -> None:
    height = len(data)
    width = len(data[0])
    with mem_file.open(
        driver="GTiff",
        dtype=rasterio.float32,
        nodata=0.0,
        width=width,
        height=height,
        count=1,
        crs=rasterio.CRS.from_epsg(32618),
        # The lower-left corner in the following transform is (0, 0) and cell_size is 1.
        transform=rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, height),
    ) as raster:
        raster.write(numpy.array(data).astype(rasterio.float32), 1)


def test_load_elevation_from_geotiff_default_no_data():
    with rasterio.io.MemoryFile() as mem_file:
        prepare_test_geotiff_elevation_memory_file(
            [
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0, 24.0],
                [25.0, 26.0, 27.0, 28.0, 29.0],
                [30.0, 31.0, 32.0, 33.0, 34.0],
            ],
            mem_file,
        )
        boundaries_polygons = [
            (geometry.Polygon([(0, 2.5), (2.5, 5), (5, 2.5), (2.5, 0), (0, 2.5)]), 1)
        ]
        output_buffer = io.StringIO()
        elevation_transformers.transform_geotiff_with_boundaries_to_esri_ascii(
            mem_file,
            output_buffer,
            no_data_value=-1.0,
            boundaries_polygons=boundaries_polygons,
        )

        assert output_buffer.getvalue() == (
            "ncols 5\n"
            + "nrows 5\n"
            + "xllcorner 0.0\n"
            + "yllcorner 0.0\n"
            + "cellsize 1.0\n"
            + "NODATA_value -1.0\n"
            + "-1.0 -1.0 12.0 -1.0 -1.0\n"
            + "-1.0 16.0 17.0 18.0 -1.0\n"
            + "20.0 21.0 22.0 23.0 24.0\n"
            + "-1.0 26.0 27.0 28.0 -1.0\n"
            + "-1.0 -1.0 32.0 -1.0 -1.0\n"
        )

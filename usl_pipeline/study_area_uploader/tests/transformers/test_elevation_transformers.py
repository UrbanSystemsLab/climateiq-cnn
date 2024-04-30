import pathlib
import tempfile

import numpy
from numpy import testing
import rasterio

from study_area_uploader.transformers import elevation_transformers
from usl_lib.readers import elevation_readers
from usl_lib.shared import geo_data


def assert_crop_geotiff(
    input_file_path: str | pathlib.Path,
    bounding_box: geo_data.BoundingBox,
    expected_elevation_header: geo_data.ElevationHeader,
    expected_elevation_data: list[list[float]],
    border_cell_count: int = 0,
):
    with tempfile.NamedTemporaryFile() as temp_file:
        output_file_path = temp_file.name
        elevation_transformers.crop_geotiff_to_sub_area(
            input_file_path,
            output_file_path,
            bounding_box,
            border_cell_count=border_cell_count,
        )
        with pathlib.Path(output_file_path).open("rb") as output_file:
            output_elevation = elevation_readers.read_from_geotiff(output_file)
        assert output_elevation.header == expected_elevation_header
        testing.assert_array_equal(output_elevation.data, expected_elevation_data)


def test_crop_geotiff_to_sub_area():
    # In the following
    tiff_array = numpy.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0, 13.0, 14.0],
            [15.0, 16.0, 17.0, 18.0, 19.0],
            [20.0, 21.0, 22.0, 23.0, 24.0],
        ]
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file_path = pathlib.Path(temp_dir) / "input.tif"
        with rasterio.open(
            str(input_file_path),
            "w",
            driver="GTiff",
            dtype=rasterio.float32,
            nodata=0.0,
            width=5,
            height=5,
            count=1,
            crs=rasterio.CRS.from_epsg(32618),
            transform=rasterio.Affine(2.0, 0.0, 10.0, 0.0, -2.0, 30.0),
        ) as raster:
            raster.write(tiff_array.astype(rasterio.float32), 1)

        assert_crop_geotiff(
            input_file_path,
            (13.0, 25.0, 15.0, 27.0),
            geo_data.ElevationHeader(
                col_count=2,
                row_count=2,
                x_ll_corner=12.0,
                y_ll_corner=24.0,
                cell_size=2.0,
                nodata_value=0.0,
                crs=rasterio.CRS.from_epsg(32618),
            ),
            [[6.0, 7.0], [11.0, 12.0]],
        )

        assert_crop_geotiff(
            input_file_path,
            (14.0, 24.0, 16.0, 26.0),
            geo_data.ElevationHeader(
                col_count=1,
                row_count=1,
                x_ll_corner=14.0,
                y_ll_corner=24.0,
                cell_size=2.0,
                nodata_value=0.0,
                crs=rasterio.CRS.from_epsg(32618),
            ),
            [[12.0]],
        )

        assert_crop_geotiff(
            input_file_path,
            (14.0, 24.0, 16.0, 26.0),
            geo_data.ElevationHeader(
                col_count=3,
                row_count=3,
                x_ll_corner=12.0,
                y_ll_corner=22.0,
                cell_size=2.0,
                nodata_value=0.0,
                crs=rasterio.CRS.from_epsg(32618),
            ),
            [[6.0, 7.0, 8.0], [11.0, 12.0, 13], [16.0, 17.0, 18.0]],
            border_cell_count=1,
        )

import io
import pathlib
import tempfile

import numpy
from numpy import testing
import rasterio
import rasterio.io
from shapely import geometry

from study_area_uploader.transformers import elevation_transformers
from usl_lib.readers import elevation_readers
from usl_lib.shared import geo_data


def prepare_test_geotiff_elevation_file(
    data: list[list[float]],
    file_path: pathlib.Path,
    x_ll_corner: float = 0.0,
    y_ll_corner: float = 0.0,
    cell_size: float = 1.0,
) -> None:
    height = len(data)
    width = len(data[0])
    with rasterio.open(
        str(file_path),
        "w",
        driver="GTiff",
        dtype=rasterio.float32,
        nodata=0.0,
        width=width,
        height=height,
        count=1,
        crs=rasterio.CRS.from_epsg(32618),
        transform=rasterio.Affine(
            cell_size,
            0.0,
            x_ll_corner,
            0.0,
            -cell_size,
            y_ll_corner + cell_size * height,
        ),
    ) as raster:
        raster.write(numpy.array(data).astype(rasterio.float32), 1)


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
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file_path = pathlib.Path(temp_dir) / "input.tif"
        prepare_test_geotiff_elevation_file(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0, 24.0],
            ],
            input_file_path,
            x_ll_corner=10,
            y_ll_corner=20,
            cell_size=2,
        )

        assert_crop_geotiff(
            input_file_path,
            geo_data.BoundingBox.from_tuple((13.0, 25.0, 15.0, 27.0)),
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
            geo_data.BoundingBox.from_tuple((14.0, 24.0, 16.0, 26.0)),
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
            geo_data.BoundingBox.from_tuple((14.0, 24.0, 16.0, 26.0)),
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


def test_transform_geotiff_with_boundaries_to_esri_ascii():
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file_path = pathlib.Path(temp_dir) / "input.tif"
        prepare_test_geotiff_elevation_file(
            [
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0, 24.0],
                [25.0, 26.0, 27.0, 28.0, 29.0],
                [30.0, 31.0, 32.0, 33.0, 34.0],
            ],
            input_file_path,
        )
        boundaries_polygons = [
            (geometry.Polygon([(0, 2.5), (2.5, 5), (5, 2.5), (2.5, 0), (0, 2.5)]), 1)
        ]
        temp_buffer_file_path = pathlib.Path(temp_dir) / "temp.tif"
        output_buffer = io.StringIO()
        elevation_transformers.transform_geotiff_with_boundaries_to_esri_ascii(
            input_file_path,
            temp_buffer_file_path,
            output_buffer,
            no_data_value=-1.0,
            boundaries_polygons=boundaries_polygons,
            row_buffer_size=2,
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

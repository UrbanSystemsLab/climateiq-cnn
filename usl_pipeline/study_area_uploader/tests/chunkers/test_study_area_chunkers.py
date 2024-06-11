import pathlib
import shutil
import tarfile
import tempfile
from typing import Optional, Tuple
import typing.io
from unittest import mock

from google.cloud import storage
import numpy
from numpy import testing
import rasterio
from shapely import geometry

from study_area_uploader.chunkers import study_area_chunkers
from study_area_uploader.transformers import study_area_transformers
from usl_lib.readers import elevation_readers, polygon_readers
from usl_lib.shared import geo_data
from usl_lib.storage import file_names


def prepare_test_geotiff_elevation_file(
    data: list[list[float]],
    file_path: pathlib.Path,
    x_ll_corner: float = 0.0,
    y_ll_corner: float = 0.0,
    cell_size: float = 1.0,
) -> None:
    """Creates a test GeoTIFF elevation file."""
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


def bbox_polygon(x1: float, y1: float, x2: float, y2: float) -> geometry.Polygon:
    """Converts a bounding box defined by two opposite corners to a Polygon."""
    return geometry.Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)])


def load_polygons_from_stream(
    fd: typing.io.IO[bytes],
) -> list[Tuple[geometry.Polygon, int]]:
    """Loads polygon/mask pairs from stream and returns in form of list."""
    return list(polygon_readers.read_polygons_from_text_file(fd))


def assert_polygon_masks_equal(
    pm1: list[Tuple[geometry.Polygon, int]],
    pm2: list[Tuple[geometry.Polygon, int]],
) -> None:
    """Checks that two lists of polygon/mask pairs are equal (with polygon equality)."""
    assert len(pm1) == len(pm2)
    for i in range(len(pm1)):
        assert pm1[i][1] == pm2[i][1]
        assert pm1[i][0].equals(pm2[i][0])


def assert_check_tar(
    chunk_tar_file: typing.BinaryIO,
    col_count: int,
    row_count: int,
    x_ll_corner: float,
    y_ll_corner: float,
    elevation_data: list[list[float]],
    boundaries_polygons: list[Tuple[geometry.Polygon, int]] = [],
    buildings_polygons: list[Tuple[geometry.Polygon, int]] = [],
    green_areas_polygons: list[Tuple[geometry.Polygon, int]] = [],
    soil_classes_polygons: list[Tuple[geometry.Polygon, int]] = [],
) -> None:
    """Checks that contents of TAR-file match the expected elevation/polygons data."""
    observed_elevation: Optional[geo_data.Elevation] = None
    observed_boundaries_polygons = []
    observed_buildings_polygons = []
    observed_green_areas_polygons = []
    observed_soil_classes_polygons = []
    chunk_tar_file.seek(0)
    with tarfile.TarFile(
        fileobj=chunk_tar_file,
    ) as tar, tempfile.TemporaryDirectory() as temp_dir:
        for member in tar:
            fd = tar.extractfile(member)
            if fd is None:
                continue

            name = pathlib.PurePosixPath(member.name).name
            if name == file_names.ELEVATION_TIF:
                temp_elevation_file = pathlib.Path(temp_dir) / "elevation.tif"
                with open(temp_elevation_file, "wb") as output_fd:
                    shutil.copyfileobj(fd, output_fd)
                with open(temp_elevation_file, "rb") as input_fd:
                    observed_elevation = elevation_readers.read_from_geotiff(input_fd)
            elif name == file_names.BOUNDARIES_TXT:
                observed_boundaries_polygons = load_polygons_from_stream(fd)
            elif name == file_names.BUILDINGS_TXT:
                observed_buildings_polygons = load_polygons_from_stream(fd)
            elif name == file_names.GREEN_AREAS_TXT:
                observed_green_areas_polygons = load_polygons_from_stream(fd)
            elif name == file_names.SOIL_CLASSES_TXT:
                observed_soil_classes_polygons = load_polygons_from_stream(fd)
    assert observed_elevation is not None
    assert observed_elevation.header == geo_data.ElevationHeader(
        col_count=col_count,
        row_count=row_count,
        x_ll_corner=x_ll_corner,
        y_ll_corner=y_ll_corner,
        cell_size=1,
        nodata_value=-9999.0,
        crs=rasterio.CRS.from_epsg(32618),
    )
    testing.assert_array_equal(observed_elevation.data, elevation_data)
    assert_polygon_masks_equal(observed_boundaries_polygons, boundaries_polygons)
    assert_polygon_masks_equal(observed_buildings_polygons, buildings_polygons)
    assert_polygon_masks_equal(observed_green_areas_polygons, green_areas_polygons)
    assert_polygon_masks_equal(observed_soil_classes_polygons, soil_classes_polygons)


def test_build_and_upload_chunks():
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = pathlib.Path(temp_dir)

        # Input data:
        elevation_input_file_path = work_dir / "input_elevation.tif"
        prepare_test_geotiff_elevation_file(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0, 24.0],
            ],
            elevation_input_file_path,
        )
        boundaries_polygons = [(bbox_polygon(1, 1, 4, 4), 1)]  # touches all chunks
        buildings_polygons = [(bbox_polygon(0, 4, 1, 5), 1)]  # fits into chunk 0-0
        green_areas_polygons = [(bbox_polygon(0, 2, 1, 3), 1)]  # touches 0-0, 1-0
        soil_classes_polygons = [(bbox_polygon(2, 4, 3, 5), 9)]  # touches 0-0, 0-1

        chunk_0_0_blob = mock.MagicMock(spec=storage.Blob)
        chunk_0_1_blob = mock.MagicMock(spec=storage.Blob)
        chunk_1_0_blob = mock.MagicMock(spec=storage.Blob)
        chunk_1_1_blob = mock.MagicMock(spec=storage.Blob)
        mock_bucket = mock.MagicMock(spec=storage.Bucket)
        mock_bucket.blob.side_effect = [
            chunk_0_0_blob,
            chunk_0_1_blob,
            chunk_1_0_blob,
            chunk_1_1_blob,
        ]
        study_area_chunkers.build_and_upload_chunks(
            "TestArea1",
            study_area_transformers.PreparedInputData(
                elevation_file_path=elevation_input_file_path,
                boundaries_polygons=boundaries_polygons,
                buildings_polygons=buildings_polygons,
                green_areas_polygons=green_areas_polygons,
                soil_classes_polygons=soil_classes_polygons,
            ),
            work_dir,
            # mock_cloud_bucket(work_dir),
            mock_bucket,
            chunk_size=3,
            extend_soil_classes_chunk_border=False,
        )

        chunk_0_0_tar_path = chunk_0_0_blob.upload_from_file.mock_calls[0].args[0]
        chunk_0_1_tar_path = chunk_0_1_blob.upload_from_file.mock_calls[0].args[0]
        chunk_1_0_tar_path = chunk_1_0_blob.upload_from_file.mock_calls[0].args[0]
        chunk_1_1_tar_path = chunk_1_1_blob.upload_from_file.mock_calls[0].args[0]

        assert_check_tar(
            chunk_0_0_tar_path,
            3,
            3,
            0.0,
            2.0,
            [
                [-9999.0, 1.0, 2.0],
                [5.0, 6.0, 7.0],
                [10.0, 11.0, 12.0],
            ],
            boundaries_polygons=[(bbox_polygon(1, 1, 4, 4), 1)],
            buildings_polygons=[(bbox_polygon(0, 4, 1, 5), 1)],
            green_areas_polygons=[(bbox_polygon(0, 2, 1, 3), 1)],
            soil_classes_polygons=[(bbox_polygon(2, 4, 3, 5), 9)],
        )
        assert_check_tar(
            chunk_0_1_tar_path,
            3,
            3,
            3.0,
            2.0,
            [
                [3.0, 4.0, -9999.0],
                [8.0, 9.0, -9999.0],
                [13.0, 14.0, -9999.0],
            ],
            boundaries_polygons=[(bbox_polygon(1, 1, 4, 4), 1)],
            soil_classes_polygons=[(bbox_polygon(2, 4, 3, 5), 9)],
        )
        assert_check_tar(
            chunk_1_0_tar_path,
            3,
            3,
            0.0,
            -1.0,
            [
                [15.0, 16.0, 17.0],
                [20.0, 21.0, 22.0],
                [-9999.0, -9999.0, -9999.0],
            ],
            boundaries_polygons=[(bbox_polygon(1, 1, 4, 4), 1)],
            green_areas_polygons=[(bbox_polygon(0, 2, 1, 3), 1)],
        )
        assert_check_tar(
            chunk_1_1_tar_path,
            3,
            3,
            3.0,
            -1.0,
            [
                [18.0, 19.0, -9999.0],
                [23.0, 24.0, -9999.0],
                [-9999.0, -9999.0, -9999.0],
            ],
            boundaries_polygons=[(bbox_polygon(1, 1, 4, 4), 1)],
        )

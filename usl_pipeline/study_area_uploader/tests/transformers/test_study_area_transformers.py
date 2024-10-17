from io import StringIO
import pathlib
import tempfile
from typing import Tuple
from unittest import mock

import fiona
from google.cloud import storage
import numpy
from numpy import testing
import rasterio
from shapely import geometry

from study_area_uploader.transformers import study_area_transformers
from usl_lib.readers import elevation_readers
from usl_lib.shared import geo_data


def polygon_points(polygon: geometry.Polygon) -> list[Tuple[float, float]]:
    xx, yy = polygon.exterior.coords.xy
    return list(zip(xx, yy))


def prepare_shape_file(
    file_path: pathlib.Path,
    polygon_masks: list[Tuple[geometry.Polygon, int]],
    soil_class_prop: str = "unused_one",
) -> None:
    crs = "EPSG:4326"
    schema = {"geometry": "Polygon", "properties": {soil_class_prop: "int"}}
    with fiona.open(
        file_path,
        "w",
        crs=crs,
        driver="ESRI Shapefile",
        schema=schema,
    ) as output:
        for polygon_mask in polygon_masks:
            output.write(
                {
                    "geometry": {
                        "coordinates": [polygon_points(polygon_mask[0])],
                        "type": "Polygon",
                    },
                    "id": "1",
                    "properties": {soil_class_prop: polygon_mask[1]},
                    "type": "Feature",
                }
            )


def bbox_polygon(x1: float, y1: float, x2: float, y2: float) -> geometry.Polygon:
    return geometry.Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)])


def prepare_elevation_data(
    elevation_file_path: pathlib.Path, data: list[list[float]]
) -> None:
    with rasterio.open(
        str(elevation_file_path),
        "w",
        driver="GTiff",
        dtype=rasterio.float32,
        nodata=0.0,
        width=5,
        height=5,
        count=1,
        crs=rasterio.CRS.from_epsg(4326),
        # The lower-left corner in the following transform is (0, 0) and cell_size is 1.
        transform=rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 5.0),
    ) as raster:
        raster.write(numpy.array(data).astype(rasterio.float32), 1)


def test_transform_shape_file_no_polygon_masks():
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = pathlib.Path(temp_dir)
        buildings_input_file_path = input_dir / "buildings.shp"
        prepare_shape_file(
            buildings_input_file_path,
            [
                (bbox_polygon(2, 1, 4, 3), 1),
            ],
        )

        cropped_buildings = list(
            study_area_transformers.transform_shape_file(
                buildings_input_file_path,
                geo_data.BoundingBox(min_x=1, min_y=1, max_x=3, max_y=3),
                "EPSG:4326",
            )
        )

        assert len(cropped_buildings) == 1
        assert cropped_buildings[0][0].equals(bbox_polygon(2, 1, 3, 3))


def test_transform_shape_file_with_polygon_masks():
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = pathlib.Path(temp_dir)
        soil_classes_input_file_path = input_dir / "soil_classes.shp"
        soil_class_prop = "soil_class"
        prepare_shape_file(
            soil_classes_input_file_path,
            [
                (bbox_polygon(0, 0, 5, 5), 0),  # It has 0 mask and will be skipped
                (bbox_polygon(2, 1, 4, 3), 8),
                (bbox_polygon(1, 0, 3, 2), 9),
            ],
            soil_class_prop=soil_class_prop,
        )

        soil_classes = list(
            study_area_transformers.transform_shape_file(
                soil_classes_input_file_path,
                None,
                "EPSG:4326",
                mask_value_feature_property=soil_class_prop,
            )
        )

        assert len(soil_classes) == 2
        assert soil_classes[0][0].equals(bbox_polygon(2, 1, 4, 3))
        assert soil_classes[0][1] == 8
        assert soil_classes[1][0].equals(bbox_polygon(1, 0, 3, 2))
        assert soil_classes[1][1] == 9


def test_prepare_and_upload_study_area_files_no_boundaries_buildings_only():
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = pathlib.Path(temp_dir)
        elevation_input_file_path = work_dir / "input_elevation.tif"
        prepare_elevation_data(
            elevation_input_file_path,
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0, 24.0],
            ],
        )
        buildings_input_file_path = work_dir / "buildings.shp"
        prepare_shape_file(
            buildings_input_file_path,
            [(bbox_polygon(2, 1, 4, 3), 1)],
        )

        # Elevation header storage file mocks
        header_cloud_storage_buffer = StringIO()
        header_cloud_storage_buffer.close = lambda: None  # Want to check content
        header_blob = mock.MagicMock(spec=storage.Blob)
        header_blob.open.return_value = header_cloud_storage_buffer
        # Buildings storage file mocks
        buildings_cloud_storage_buffer = StringIO()
        buildings_cloud_storage_buffer.close = lambda: None  # Want to check content
        buildings_blob = mock.MagicMock(spec=storage.Blob)
        buildings_blob.open.return_value = buildings_cloud_storage_buffer

        study_area_bucket = mock.MagicMock(spec=storage.Bucket)
        elevation_blob = mock.MagicMock(spec=storage.Blob)
        study_area_bucket.blob.side_effect = [
            header_blob,
            elevation_blob,
            buildings_blob,
        ]
        prepared_inputs = study_area_transformers.prepare_and_upload_study_area_files(
            "TestArea1",
            elevation_input_file_path,
            None,
            buildings_input_file_path,
            None,
            None,
            None,
            work_dir,
            study_area_bucket,
        )

        assert prepared_inputs.elevation_file_path == elevation_input_file_path
        assert len(prepared_inputs.buildings_polygons) == 1
        assert prepared_inputs.buildings_polygons[0][0].equals(bbox_polygon(2, 1, 4, 3))
        assert study_area_bucket.mock_calls == [
            mock.call.blob("TestArea1/header.json"),
            mock.call.blob("TestArea1/elevation.tif"),
            mock.call.blob("TestArea1/buildings.txt"),
        ]
        assert header_blob.mock_calls == [mock.call.open("w")]
        assert elevation_blob.mock_calls == [
            mock.call.upload_from_filename(str(prepared_inputs.elevation_file_path))
        ]
        assert buildings_blob.mock_calls == [mock.call.open("w")]
        assert (
            header_cloud_storage_buffer.getvalue()
            == "{\n"
            + '    "col_count": 5,\n'
            + '    "row_count": 5,\n'
            + '    "x_ll_corner": 0.0,\n'
            + '    "y_ll_corner": 0.0,\n'
            + '    "cell_size": 1.0,\n'
            + '    "nodata_value": 0.0,\n'
            + '    "crs": "EPSG:4326"\n'
            + "}"
        )
        assert (
            buildings_cloud_storage_buffer.getvalue()
            == "1\n" + "5 2.0 2.0 4.0 4.0 2.0 1.0 3.0 3.0 1.0 1.0\n"
        )


def test_prepare_and_upload_study_area_files_with_boundaries_green_areas_soil_classes():
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = pathlib.Path(temp_dir)
        elevation_input_file_path = work_dir / "input_elevation.tif"
        prepare_elevation_data(
            elevation_input_file_path,
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0, 24.0],
            ],
        )
        # boundaries
        boundaries_input_file_path = work_dir / "boundaries.shp"
        prepare_shape_file(
            boundaries_input_file_path,
            [(bbox_polygon(1, 1, 3, 3), 1)],
        )
        # Green areas
        green_areas_input_file_path = work_dir / "green_areas.shp"
        prepare_shape_file(
            green_areas_input_file_path,
            [(bbox_polygon(0, 0, 2, 2), 1)],
        )
        # Green areas
        soil_classes_input_file_path = work_dir / "soil_classes.shp"
        prepare_shape_file(
            soil_classes_input_file_path,
            [(bbox_polygon(0, 0, 2, 2), 99)],
            soil_class_prop="soil_class",
        )

        # Mocks:
        # Boundaries storage file mocks
        boundaries_cloud_storage_buffer = StringIO()
        boundaries_cloud_storage_buffer.close = lambda: None  # Want to check content
        boundaries_blob = mock.MagicMock(spec=storage.Blob)
        boundaries_blob.open.return_value = boundaries_cloud_storage_buffer
        # Elevation header storage file mocks
        header_cloud_storage_buffer = StringIO()
        header_cloud_storage_buffer.close = lambda: None  # Want to check content
        header_blob = mock.MagicMock(spec=storage.Blob)
        header_blob.open.return_value = header_cloud_storage_buffer
        # Green areas storage file mocks
        green_areas_cloud_storage_buffer = StringIO()
        green_areas_cloud_storage_buffer.close = lambda: None  # Want to check content
        green_areas_blob = mock.MagicMock(spec=storage.Blob)
        green_areas_blob.open.return_value = green_areas_cloud_storage_buffer
        # Soil classes storage file mocks
        soil_classes_cloud_storage_buffer = StringIO()
        soil_classes_cloud_storage_buffer.close = lambda: None  # Want to check content
        soil_classes_blob = mock.MagicMock(spec=storage.Blob)
        soil_classes_blob.open.return_value = soil_classes_cloud_storage_buffer
        # Storage bucket
        study_area_bucket = mock.MagicMock(spec=storage.Bucket)
        elevation_blob = mock.MagicMock(spec=storage.Blob)
        study_area_bucket.blob.side_effect = [
            boundaries_blob,
            header_blob,
            elevation_blob,
            green_areas_blob,
            soil_classes_blob,
        ]

        prepared_inputs = study_area_transformers.prepare_and_upload_study_area_files(
            "TestArea1",
            elevation_input_file_path,
            boundaries_input_file_path,
            None,
            green_areas_input_file_path,
            soil_classes_input_file_path,
            "soil_class",
            work_dir,
            study_area_bucket,
        )

        with open(prepared_inputs.elevation_file_path, "rb") as input_file:
            elevation = elevation_readers.read_from_geotiff(input_file)
        assert elevation.header == geo_data.ElevationHeader(
            col_count=4,
            row_count=4,
            x_ll_corner=0.0,
            y_ll_corner=0.0,
            cell_size=1.0,
            nodata_value=0.0,
            crs=rasterio.CRS({"init": "EPSG:4326"}),
        )
        testing.assert_array_equal(
            elevation.data,
            [
                [5.0, 6.0, 7.0, 8.0],
                [10.0, 11.0, 12.0, 13.0],
                [15.0, 16.0, 17.0, 18.0],
                [20.0, 21.0, 22.0, 23.0],
            ],
        )
        assert len(prepared_inputs.boundaries_polygons) == 1
        assert prepared_inputs.boundaries_polygons[0][0].equals(
            bbox_polygon(1, 1, 3, 3)
        )
        assert len(prepared_inputs.green_areas_polygons) == 1
        assert prepared_inputs.green_areas_polygons[0][0].equals(
            bbox_polygon(1, 1, 2, 2)
        )
        assert len(prepared_inputs.soil_classes_polygons) == 1
        assert prepared_inputs.soil_classes_polygons[0][0].equals(
            bbox_polygon(1, 1, 2, 2)
        )
        assert prepared_inputs.soil_classes_polygons[0][1] == 99
        assert study_area_bucket.mock_calls == [
            mock.call.blob("TestArea1/boundaries.txt"),
            mock.call.blob("TestArea1/header.json"),
            mock.call.blob("TestArea1/elevation.tif"),
            mock.call.blob("TestArea1/green_areas.txt"),
            mock.call.blob("TestArea1/soil_classes.txt"),
        ]
        assert boundaries_blob.mock_calls == [mock.call.open("w")]
        assert header_blob.mock_calls == [mock.call.open("w")]
        assert elevation_blob.mock_calls == [
            mock.call.upload_from_filename(str(prepared_inputs.elevation_file_path))
        ]
        assert green_areas_blob.mock_calls == [mock.call.open("w")]
        assert soil_classes_blob.mock_calls == [mock.call.open("w")]
        assert (
            boundaries_cloud_storage_buffer.getvalue()
            == "1\n" + "5 1.0 1.0 3.0 3.0 1.0 1.0 3.0 3.0 1.0 1.0\n"
        )
        assert (
            header_cloud_storage_buffer.getvalue()
            == "{\n"
            + '    "col_count": 4,\n'
            + '    "row_count": 4,\n'
            + '    "x_ll_corner": 0.0,\n'
            + '    "y_ll_corner": 0.0,\n'
            + '    "cell_size": 1.0,\n'
            + '    "nodata_value": 0.0,\n'
            + '    "crs": "EPSG:4326"\n'
            + "}"
        )
        assert (
            green_areas_cloud_storage_buffer.getvalue()
            == "1\n" + "5 2.0 2.0 1.0 1.0 2.0 2.0 1.0 1.0 2.0 2.0\n"
        )
        assert (
            soil_classes_cloud_storage_buffer.getvalue()
            == "1\n" + "99 5 2.0 2.0 1.0 1.0 2.0 2.0 1.0 1.0 2.0 2.0\n"
        )


def test_prepare_and_upload_study_area_files_with_non_green_area_soil_class():
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = pathlib.Path(temp_dir)
        elevation_input_file_path = work_dir / "input_elevation.tif"
        prepare_elevation_data(
            elevation_input_file_path,
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0, 24.0],
            ],
        )
        # Green areas
        green_areas_input_file_path = work_dir / "green_areas.shp"
        prepare_shape_file(
            green_areas_input_file_path,
            [(bbox_polygon(0, 0, 2, 2), 1)],
        )
        # Green areas
        soil_classes_input_file_path = work_dir / "soil_classes.shp"
        prepare_shape_file(
            soil_classes_input_file_path,
            [(bbox_polygon(0, 0, 1, 5), 99)],  # This polygon will be marked as -9999
            soil_class_prop="soil_class",
        )

        # Mocks:
        # Elevation header storage file mocks
        header_cloud_storage_buffer = StringIO()
        header_cloud_storage_buffer.close = lambda: None  # Want to check content
        header_blob = mock.MagicMock(spec=storage.Blob)
        header_blob.open.return_value = header_cloud_storage_buffer
        # Green areas storage file mocks
        green_areas_cloud_storage_buffer = StringIO()
        green_areas_cloud_storage_buffer.close = lambda: None  # Want to check content
        green_areas_blob = mock.MagicMock(spec=storage.Blob)
        green_areas_blob.open.return_value = green_areas_cloud_storage_buffer
        # Soil classes storage file mocks
        soil_classes_cloud_storage_buffer = StringIO()
        soil_classes_cloud_storage_buffer.close = lambda: None  # Want to check content
        soil_classes_blob = mock.MagicMock(spec=storage.Blob)
        soil_classes_blob.open.return_value = soil_classes_cloud_storage_buffer
        # Storage bucket
        study_area_bucket = mock.MagicMock(spec=storage.Bucket)
        elevation_blob = mock.MagicMock(spec=storage.Blob)
        study_area_bucket.blob.side_effect = [
            header_blob,
            elevation_blob,
            green_areas_blob,
            soil_classes_blob,
        ]

        prepared_inputs = study_area_transformers.prepare_and_upload_study_area_files(
            "TestArea1",
            elevation_input_file_path,
            None,
            None,
            green_areas_input_file_path,
            soil_classes_input_file_path,
            "soil_class",
            work_dir,
            study_area_bucket,
            input_non_green_area_soil_classes={99},
        )

        with open(prepared_inputs.elevation_file_path, "rb") as input_file:
            elevation = elevation_readers.read_from_geotiff(input_file)
        assert elevation.header == geo_data.ElevationHeader(
            col_count=5,
            row_count=5,
            x_ll_corner=0.0,
            y_ll_corner=0.0,
            cell_size=1.0,
            nodata_value=0.0,
            crs=rasterio.CRS({"init": "EPSG:4326"}),
        )
        testing.assert_array_equal(
            elevation.data,
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0, 24.0],
            ],
        )
        assert len(prepared_inputs.green_areas_polygons) == 1
        assert prepared_inputs.green_areas_polygons[0][0].equals(
            bbox_polygon(0, 0, 2, 2)
        )
        assert len(prepared_inputs.soil_classes_polygons) == 1
        assert prepared_inputs.soil_classes_polygons[0][0].equals(
            bbox_polygon(0, 0, 1, 5)
        )
        assert prepared_inputs.soil_classes_polygons[0][1] == -9999  # default non-green
        assert study_area_bucket.mock_calls == [
            mock.call.blob("TestArea1/header.json"),
            mock.call.blob("TestArea1/elevation.tif"),
            mock.call.blob("TestArea1/green_areas.txt"),
            mock.call.blob("TestArea1/soil_classes.txt"),
        ]
        assert elevation_blob.mock_calls == [
            mock.call.upload_from_filename(str(prepared_inputs.elevation_file_path))
        ]
        assert green_areas_blob.mock_calls == [mock.call.open("w")]
        assert soil_classes_blob.mock_calls == [mock.call.open("w")]


def test_prepare_and_upload_citycat_input_files_no_boundaries_buildings_only():
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = pathlib.Path(temp_dir)

        # Input data:
        elevation_input_file_path = work_dir / "input_elevation.tif"
        prepare_elevation_data(
            elevation_input_file_path,
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0, 24.0],
            ],
        )
        buildings_polygons = [(bbox_polygon(2, 1, 4, 3), 1)]

        # Mocks:
        elevation_cloud_storage_buffer = StringIO()
        elevation_cloud_storage_buffer.close = lambda: None  # Want to check content
        elevation_blob = mock.MagicMock(spec=storage.Blob)
        elevation_blob.open.return_value = elevation_cloud_storage_buffer
        buildings_cloud_storage_buffer = StringIO()
        buildings_cloud_storage_buffer.close = lambda: None  # Want to check content
        buildings_blob = mock.MagicMock(spec=storage.Blob)
        buildings_blob.open.return_value = buildings_cloud_storage_buffer
        # Storage bucket
        citycat_bucket = mock.MagicMock(spec=storage.Bucket)
        citycat_bucket.blob.side_effect = [elevation_blob, buildings_blob]

        # Execute the code:
        study_area_transformers.prepare_and_upload_citycat_input_files(
            "TestArea1",
            study_area_transformers.PreparedInputData(
                elevation_file_path=elevation_input_file_path,
                boundaries_polygons=None,
                buildings_polygons=buildings_polygons,
                green_areas_polygons=None,
                soil_classes_polygons=None,
            ),
            work_dir,
            citycat_bucket,
        )

        assert citycat_bucket.mock_calls == [
            mock.call.blob("TestArea1/Domain_DEM.asc"),
            mock.call.blob("TestArea1/Buildings.txt"),
        ]
        assert elevation_cloud_storage_buffer.getvalue() == (
            "ncols 5\n"
            + "nrows 5\n"
            + "xllcorner 0.0\n"
            + "yllcorner 0.0\n"
            + "cellsize 1.0\n"
            + "NODATA_value -9999.0\n"
            + "-9999.0 1.0 2.0 3.0 4.0\n"
            + "5.0 6.0 7.0 8.0 9.0\n"
            + "10.0 11.0 12.0 13.0 14.0\n"
            + "15.0 16.0 17.0 18.0 19.0\n"
            + "20.0 21.0 22.0 23.0 24.0\n"
        )
        assert (
            buildings_cloud_storage_buffer.getvalue()
            == "1\n5 2.0 4.0 4.0 2.0 2.0 1.0 1.0 3.0 3.0 1.0\n"
        )


def test_prepare_and_upload_citycat_input_files_with_boundaries_and_green_areas():
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = pathlib.Path(temp_dir)

        # Input data:
        elevation_input_file_path = work_dir / "input_elevation.tif"
        prepare_elevation_data(
            elevation_input_file_path,
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0, 24.0],
            ],
        )
        boundaries_polygons = [
            (geometry.Polygon([(0, 2.5), (2.5, 5), (5, 2.5), (2.5, 0), (0, 2.5)]), 1)
        ]
        green_areas_polygons = [(bbox_polygon(0, 0, 5, 3), 1)]

        # Mocks:
        elevation_cloud_storage_buffer = StringIO()
        elevation_cloud_storage_buffer.close = lambda: None  # Want to check content
        elevation_blob = mock.MagicMock(spec=storage.Blob)
        elevation_blob.open.return_value = elevation_cloud_storage_buffer
        green_areas_cloud_storage_buffer = StringIO()
        green_areas_cloud_storage_buffer.close = lambda: None  # Want to check content
        green_areas_blob = mock.MagicMock(spec=storage.Blob)
        green_areas_blob.open.return_value = green_areas_cloud_storage_buffer
        # Storage bucket
        citycat_bucket = mock.MagicMock(spec=storage.Bucket)
        citycat_bucket.blob.side_effect = [elevation_blob, green_areas_blob]

        # Execute the code:
        study_area_transformers.prepare_and_upload_citycat_input_files(
            "TestArea1",
            study_area_transformers.PreparedInputData(
                elevation_file_path=elevation_input_file_path,
                boundaries_polygons=boundaries_polygons,
                buildings_polygons=None,
                green_areas_polygons=green_areas_polygons,
                soil_classes_polygons=None,
            ),
            work_dir,
            citycat_bucket,
            default_no_data_value=-1.0,
        )

        assert citycat_bucket.mock_calls == [
            mock.call.blob("TestArea1/Domain_DEM.asc"),
            mock.call.blob("TestArea1/GreenAreas.txt"),
        ]
        assert elevation_cloud_storage_buffer.getvalue() == (
            "ncols 5\n"
            + "nrows 5\n"
            + "xllcorner 0.0\n"
            + "yllcorner 0.0\n"
            + "cellsize 1.0\n"
            + "NODATA_value -1.0\n"
            + "-1.0 -1.0 2.0 -1.0 -1.0\n"
            + "-1.0 6.0 7.0 8.0 -1.0\n"
            + "10.0 11.0 12.0 13.0 14.0\n"
            + "-1.0 16.0 17.0 18.0 -1.0\n"
            + "-1.0 -1.0 22.0 -1.0 -1.0\n"
        )
        assert green_areas_cloud_storage_buffer.getvalue() == (
            "1\n6 4.5 5.0 2.5 0.0 0.5 4.5 3.0 2.5 0.0 2.5 3.0 3.0\n"
        )


def test_prepare_and_upload_citycat_input_files_with_boundaries_green_areas_and_soil():
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = pathlib.Path(temp_dir)

        # Input data:
        elevation_input_file_path = work_dir / "input_elevation.tif"
        prepare_elevation_data(
            elevation_input_file_path,
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0, 24.0],
            ],
        )
        boundaries_polygons = [
            (geometry.Polygon([(0, 2.5), (2.5, 5), (5, 2.5), (2.5, 0), (0, 2.5)]), 1)
        ]
        green_areas_polygons = [(bbox_polygon(0, 0, 5, 3), 1)]
        soil_classes_polygons = [
            (bbox_polygon(0, 0, 1, 5), -9999),  # Will be excluded from green areas
            (bbox_polygon(1, 0, 2.5, 5), 8),
            (bbox_polygon(2.5, 0, 4, 5), 9),
        ]

        # Mocks:
        elevation_cloud_storage_buffer = StringIO()
        elevation_cloud_storage_buffer.close = lambda: None  # Want to check content
        elevation_blob = mock.MagicMock(spec=storage.Blob)
        elevation_blob.open.return_value = elevation_cloud_storage_buffer
        green_areas_cloud_storage_buffer = StringIO()
        green_areas_cloud_storage_buffer.close = lambda: None  # Want to check content
        green_areas_blob = mock.MagicMock(spec=storage.Blob)
        green_areas_blob.open.return_value = green_areas_cloud_storage_buffer
        # Storage bucket
        citycat_bucket = mock.MagicMock(spec=storage.Bucket)
        citycat_bucket.blob.side_effect = [elevation_blob, green_areas_blob]

        # Execute the code:
        study_area_transformers.prepare_and_upload_citycat_input_files(
            "TestArea1",
            study_area_transformers.PreparedInputData(
                elevation_file_path=elevation_input_file_path,
                boundaries_polygons=boundaries_polygons,
                buildings_polygons=None,
                green_areas_polygons=green_areas_polygons,
                soil_classes_polygons=soil_classes_polygons,
            ),
            work_dir,
            citycat_bucket,
            default_no_data_value=-1.0,
        )

        assert citycat_bucket.mock_calls == [
            mock.call.blob("TestArea1/Domain_DEM.asc"),
            mock.call.blob("TestArea1/Spatial_GreenAreas.txt"),
        ]
        assert elevation_cloud_storage_buffer.getvalue() == (
            "ncols 5\n"
            + "nrows 5\n"
            + "xllcorner 0.0\n"
            + "yllcorner 0.0\n"
            + "cellsize 1.0\n"
            + "NODATA_value -1.0\n"
            + "-1.0 -1.0 2.0 -1.0 -1.0\n"
            + "-1.0 6.0 7.0 8.0 -1.0\n"
            + "10.0 11.0 12.0 13.0 14.0\n"
            + "-1.0 16.0 17.0 18.0 -1.0\n"
            + "-1.0 -1.0 22.0 -1.0 -1.0\n"
        )
        assert green_areas_cloud_storage_buffer.getvalue() == (
            "3\n"
            + "8 5 2.5 1.0 1.0 2.5 2.5 0.0 1.5 3.0 3.0 0.0\n"
            + "9 5 4.0 2.5 2.5 4.0 4.0 1.5 0.0 3.0 3.0 1.5\n"
            +
            # the following cell was added based on nearest neighbor approach
            "9 5 4.0 5.0 5.0 4.0 4.0 2.0 2.0 3.0 3.0 2.0\n"
        )

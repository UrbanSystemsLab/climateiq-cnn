import pathlib
import pytest
import tempfile
from typing import Tuple

import fiona
import numpy.testing as npt
import pyproj
from shapely import geometry

from study_area_uploader.readers import polygon_readers


def _polygon_to_points(polygon: geometry.Polygon) -> list[Tuple[float, float]]:
    xx, yy = polygon.exterior.coords.xy
    return list(zip(xx, yy))


def test_read_polygons_from_shape_file_with_transformation():
    x = 930986
    y = 129304
    input_points = [(x, y), (x + 1, y), (x, y + 1), (x, y)]
    source_crs = "EPSG:2263"
    target_crs = "EPSG:32618"
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = pathlib.Path(temp_dir) / "temp.shp"
        # Prepare temporary shape file
        schema = {"geometry": "Polygon", "properties": {}}
        with fiona.open(
            file_path,
            "w",
            crs=source_crs,
            driver="ESRI Shapefile",
            schema=schema,
        ) as output:
            output.write(
                {
                    "geometry": {"coordinates": [input_points], "type": "Polygon"},
                    "id": "1",
                    "properties": {},
                    "type": "Feature",
                }
            )
        polygon_masks = polygon_readers.read_polygons_from_shape_file(
            file_path, target_crs=target_crs
        )

    assert len(polygon_masks) == 1
    polygon, mask = polygon_masks[0]
    assert mask == 1  # No default mask since soil class is skipped

    # Check that output polygon has the same points as input
    target_xx, target_yy = polygon.exterior.coords.xy
    # Let's convert output coordinates back to source CRS for comparison
    orig_xx, orig_yy = pyproj.Transformer.from_crs(
        crs_from=target_crs, crs_to=source_crs, always_xy=True
    ).transform(target_xx, target_yy)
    output_points = list(zip(orig_xx, orig_yy))
    assert len(output_points) == 4
    npt.assert_almost_equal(output_points[0], input_points[3])
    npt.assert_almost_equal(output_points[1], input_points[2])
    npt.assert_almost_equal(output_points[2], input_points[1])
    npt.assert_almost_equal(output_points[3], input_points[0])


def test_read_polygons_from_shape_file_with_polygon_soil_class():
    x = 930986
    y = 129304
    input_points = [(x, y), (x + 1, y), (x, y + 1), (x, y)]
    crs = "EPSG:2263"
    soil_class_prop = "soil_class"
    soil_class = 9
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = pathlib.Path(temp_dir) / "temp.shp"
        # Prepare temporary shape file
        schema = {"geometry": "Polygon", "properties": {soil_class_prop: "int"}}
        with fiona.open(
            file_path,
            "w",
            crs=crs,
            driver="ESRI Shapefile",
            schema=schema,
        ) as output:
            output.write(
                {
                    "geometry": {"coordinates": [input_points], "type": "Polygon"},
                    "id": "1",
                    "properties": {soil_class_prop: soil_class},
                    "type": "Feature",
                }
            )
        polygon_masks = polygon_readers.read_polygons_from_shape_file(
            file_path, mask_value_feature_property=soil_class_prop
        )

    assert len(polygon_masks) == 1
    polygon, mask = polygon_masks[0]
    assert mask == soil_class

    # Check that output polygon has the same points as input
    assert polygon.equals(geometry.Polygon(input_points))


def test_read_polygons_from_shape_file_with_missing_polygon_soil_class():
    input_points = [(0, 0), (1, 0), (0, 1), (0, 0)]
    soil_class_prop = "soil_class"
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = pathlib.Path(temp_dir) / "temp.shp"
        # Prepare temporary shape file
        schema = {"geometry": "Polygon", "properties": {}}
        with fiona.open(
            file_path,
            "w",
            crs="EPSG:2263",
            driver="ESRI Shapefile",
            schema=schema,
        ) as output:
            output.write(
                {
                    "geometry": {"coordinates": [input_points], "type": "Polygon"},
                    "id": "1",
                    "properties": {},
                    "type": "Feature",
                }
            )
        with pytest.raises(ValueError) as exc:
            polygon_readers.read_polygons_from_shape_file(
                file_path, mask_value_feature_property=soil_class_prop
            )
        assert (
            str(exc.value)
            == "Mask value key 'soil_class' not found in feature properties {'FID': 0}"
        )


def test_read_polygons_from_shape_file_with_wrong_polygon_soil_class_type():
    input_points = [(0, 0), (1, 0), (0, 1), (0, 0)]
    soil_class_prop = "soil_class"
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = pathlib.Path(temp_dir) / "temp.shp"
        # Prepare temporary shape file
        schema = {"geometry": "Polygon", "properties": {soil_class_prop: "str"}}
        with fiona.open(
            file_path,
            "w",
            crs="EPSG:2263",
            driver="ESRI Shapefile",
            schema=schema,
        ) as output:
            output.write(
                {
                    "geometry": {"coordinates": [input_points], "type": "Polygon"},
                    "id": "1",
                    "properties": {soil_class_prop: "wrong"},
                    "type": "Feature",
                }
            )
        with pytest.raises(ValueError) as exc:
            polygon_readers.read_polygons_from_shape_file(
                file_path, mask_value_feature_property=soil_class_prop
            )
        assert (
            str(exc.value)
            == "Mask value 'wrong' for key 'soil_class' should be integer in"
            + " {'soil_class': 'wrong'}"
        )


def test_read_polygons_from_shape_file_several_entries():
    input_fragment1 = [(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
    input_fragment2 = [(10.0, 0.0), (11.0, 1.0), (11.0, 0.0), (10.0, 0.0)]
    input_fragment3 = [(20.0, 0.0), (21.0, 1.0), (21.0, 0.0), (20.0, 0.0)]
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = pathlib.Path(temp_dir) / "temp.shp"
        # Prepare temporary shape file
        schema = {"geometry": "Polygon", "properties": {}}
        with fiona.open(
            file_path,
            "w",
            crs="EPSG:2263",
            driver="ESRI Shapefile",
            schema=schema,
        ) as output:
            output.write(
                {
                    "geometry": {
                        "coordinates": [input_fragment1, input_fragment2],
                        "type": "Polygon",
                    },
                    "id": "1",
                    "properties": {},
                    "type": "Feature",
                }
            )
            output.write(
                {
                    "geometry": {"coordinates": [input_fragment3], "type": "Polygon"},
                    "id": "2",
                    "properties": {},
                    "type": "Feature",
                }
            )
        polygon_masks = polygon_readers.read_polygons_from_shape_file(file_path)

    assert len(polygon_masks) == 3
    assert polygon_masks[0][0].equals(geometry.Polygon(input_fragment1))
    assert polygon_masks[1][0].equals(geometry.Polygon(input_fragment2))
    assert polygon_masks[2][0].equals(geometry.Polygon(input_fragment3))


def test_read_polygons_from_shape_file_multipolygon():
    input_fragment1 = [(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
    input_fragment2 = [(10.0, 0.0), (11.0, 1.0), (11.0, 0.0), (10.0, 0.0)]
    input_fragment3 = [(20.0, 0.0), (21.0, 1.0), (21.0, 0.0), (20.0, 0.0)]
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = pathlib.Path(temp_dir) / "temp.shp"
        # Prepare temporary shape file
        schema = {"geometry": "MultiPolygon", "properties": {}}
        with fiona.open(
            file_path,
            "w",
            crs="EPSG:2263",
            driver="ESRI Shapefile",
            schema=schema,
        ) as output:
            output.write(
                {
                    "geometry": {
                        "coordinates": [
                            [input_fragment1, input_fragment2],
                            [input_fragment3],
                        ],
                        "type": "MultiPolygon",
                    },
                    "id": "1",
                    "properties": {},
                    "type": "Feature",
                }
            )
        polygon_masks = polygon_readers.read_polygons_from_shape_file(file_path)

    assert len(polygon_masks) == 3
    assert polygon_masks[0][0].equals(geometry.Polygon(input_fragment1))
    assert polygon_masks[1][0].equals(geometry.Polygon(input_fragment2))
    assert polygon_masks[2][0].equals(geometry.Polygon(input_fragment3))

import pathlib
import tempfile

import fiona
import numpy.testing as npt
import pyproj

from usl_lib.readers import polygon_readers


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
    source_crs = "EPSG:2263"
    soil_class_prop = "soil_class"
    soil_class = 9
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = pathlib.Path(temp_dir) / "temp.shp"
        # Prepare temporary shape file
        schema = {"geometry": "Polygon", "properties": {soil_class_prop: "int"}}
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
    xx, yy = polygon.exterior.coords.xy
    output_points = list(zip(xx, yy))
    assert len(output_points) == 4
    npt.assert_almost_equal(output_points[0], input_points[3])
    npt.assert_almost_equal(output_points[1], input_points[2])
    npt.assert_almost_equal(output_points[2], input_points[1])
    npt.assert_almost_equal(output_points[3], input_points[0])

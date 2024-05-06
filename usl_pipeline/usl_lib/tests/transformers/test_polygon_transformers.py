import pytest

from numpy import testing
from shapely import geometry

from usl_lib.shared import geo_data
from usl_lib.transformers import polygon_transformers


def test_rasterize_polygons_default_background():
    header = geo_data.ElevationHeader(
        col_count=7,
        row_count=4,
        x_ll_corner=0.0,
        y_ll_corner=0.0,
        cell_size=1.0,
        nodata_value=0.0,
    )
    p1 = geometry.Polygon([(1, 0), (3, 0), (3, 3), (1, 3), (1, 0)])
    p2 = geometry.Polygon([(4, 1), (7, 1), (7, 3), (4, 3), (4, 1)])
    p3 = geometry.Polygon([(3, 3), (4, 3), (4, 4), (3, 4), (3, 3)])
    raster = polygon_transformers.rasterize_polygons(
        header, [(p1, 1), (p2, 2), (p3, 3)])
    testing.assert_array_equal(
        raster,
        [
            [0, 0, 0, 3, 0, 0, 0],
            [0, 1, 1, 0, 2, 2, 2],
            [0, 1, 1, 0, 2, 2, 2],
            [0, 1, 1, 0, 0, 0, 0],
        ],
    )


def test_rasterize_polygons_custom_background():
    header = geo_data.ElevationHeader(
        col_count=3,
        row_count=3,
        x_ll_corner=0.0,
        y_ll_corner=0.0,
        cell_size=1.0,
        nodata_value=0.0,
    )
    p1 = geometry.Polygon([(1, 0), (3, 0), (3, 2), (1, 2), (1, 0)])
    raster = polygon_transformers.rasterize_polygons(
        header, [(p1, 1)], background_value=9
    )
    testing.assert_array_equal(
        raster,
        [
            [9, 9, 9],
            [9, 1, 1],
            [9, 1, 1],
        ],
    )


def test_rasterize_polygons_no_polygons():
    header = geo_data.ElevationHeader(
        col_count=2,
        row_count=2,
        x_ll_corner=0.0,
        y_ll_corner=0.0,
        cell_size=1.0,
        nodata_value=0.0,
    )
    raster = polygon_transformers.rasterize_polygons(header, [], background_value=9)
    testing.assert_array_equal(
        raster,
        [
            [9, 9],
            [9, 9],
        ],
    )


def test_rasterize_polygons_wrong_polygon_mask():
    header = geo_data.ElevationHeader(
        col_count=3,
        row_count=3,
        x_ll_corner=0.0,
        y_ll_corner=0.0,
        cell_size=1.0,
        nodata_value=0.0,
    )
    p1 = geometry.Polygon([(1, 0), (3, 0), (3, 2), (1, 2), (1, 0)])

    with pytest.raises(ValueError) as exc:
        polygon_transformers.rasterize_polygons(header, [(p1, 0)])
    assert str(exc.value) == "Polygons with background mask are not allowed: {}".format(
        (p1, 0)
    )


def test_get_bounding_box_for_boundaries():
    p1 = geometry.Polygon([(1, -10), (3, -10), (3, 4), (1, -10)])
    p2 = geometry.Polygon([(4, 1), (7, 1), (7, 13), (4, 1)])

    bounding_box_1 = polygon_transformers.get_bounding_box_for_boundaries([p1])
    assert bounding_box_1 == geo_data.BoundingBox.from_tuple((1, -10, 3, 4))

    bounding_box_2 = polygon_transformers.get_bounding_box_for_boundaries([p2])
    assert bounding_box_2 == geo_data.BoundingBox.from_tuple((4, 1, 7, 13))

    bounding_box_1_2 = polygon_transformers.get_bounding_box_for_boundaries([p1, p2])
    assert bounding_box_1_2 == geo_data.BoundingBox.from_tuple((1, -10, 7, 13))


def bbox_polygon(x1: float, y1: float, x2: float, y2: float):
    return geometry.Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)])


def test_crop_polygons_to_sub_area_simple_case():
    p1 = bbox_polygon(0, 20, 2, 22)
    p2 = bbox_polygon(8, 28, 10, 30)
    p3 = bbox_polygon(4, 24, 6, 26)
    p4 = bbox_polygon(-5, -5, 0, 0)
    polygon_masks = [(p1, 1), (p2, 2), (p3, 3), (p4, 4)]

    crop_output = list(
        polygon_transformers.crop_polygons_to_sub_area(
            polygon_masks, geo_data.BoundingBox.from_tuple((1, 21, 9, 29))
        )
    )

    assert len(crop_output) == 3  # Polygon p4 was ignored since it's outside

    assert crop_output[0][0].equals(bbox_polygon(1, 21, 2, 22))
    assert crop_output[0][1] == 1

    assert crop_output[1][0].equals(bbox_polygon(8, 28, 9, 29))
    assert crop_output[1][1] == 2

    assert crop_output[2] == (p3, 3)  # Got untouched


def test_crop_polygons_to_sub_area_multi_polygon_case():
    p1 = geometry.Polygon([(0, 0), (4, 2), (0, 4), (4, 6), (0, 8), (0, 0)])

    crop_output = list(
        polygon_transformers.crop_polygons_to_sub_area(
            [(p1, 1)], geo_data.BoundingBox.from_tuple((2, 0, 4, 8))
        )
    )

    assert len(crop_output) == 2
    assert crop_output[0][0].equals(geometry.Polygon([(2, 1), (2, 3), (4, 2), (2, 1)]))
    assert crop_output[1][0].equals(geometry.Polygon([(2, 5), (2, 7), (4, 6), (2, 5)]))


def test_crop_polygons_to_sub_area_empty_case():
    p1 = geometry.Polygon([(0, 3), (3, 3), (3, 0), (0, 3)])

    crop_output = list(
        polygon_transformers.crop_polygons_to_sub_area(
            [(p1, 1)], geo_data.BoundingBox.from_tuple((0, 0, 1, 1))
        )
    )

    assert len(crop_output) == 0

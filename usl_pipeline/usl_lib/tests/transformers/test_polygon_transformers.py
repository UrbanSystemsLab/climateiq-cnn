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
    raster = polygon_transformers.rasterize_polygons(header, [(p1, 1), (p2, 2)])
    testing.assert_array_equal(
        raster,
        [
            [0, 0, 0, 0, 0, 0, 0],
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
    assert bounding_box_1 == (1, -10, 3, 4)

    bounding_box_2 = polygon_transformers.get_bounding_box_for_boundaries([p2])
    assert bounding_box_2 == (4, 1, 7, 13)

    bounding_box_1_2 = polygon_transformers.get_bounding_box_for_boundaries([p1, p2])
    assert bounding_box_1_2 == (1, -10, 7, 13)

import numpy
from shapely import geometry

from usl_lib.shared import geo_data
from usl_lib.transformers import raster_transformers


def test_fill_in_soil_classes_missing_values_from_nearest_polygons_happy_path():
    header = geo_data.ElevationHeader(
        col_count=4,
        row_count=2,
        x_ll_corner=0.0,
        y_ll_corner=0.0,
        cell_size=1.0,
        nodata_value=0.0,
    )

    green_areas = numpy.array(
        [
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ]
    )
    soil_classes = numpy.array(
        [
            [0, 0, 5, 4],
            [3, 0, 1, 0],
        ]
    )
    soil_classes_polygon_masks = [
        (geometry.Polygon([(4, 0), (5, 0), (5, 1), (4, 1), (4, 0)]), 1),  # row:col=1:4
        (geometry.Polygon([(1, 2), (2, 2), (2, 3), (1, 3), (1, 2)]), 2),  # row:col=-1:1
        (geometry.Polygon([(4, 2), (5, 2), (5, 3), (4, 3), (4, 2)]), 3),  # row:col=-1:4
    ]

    corrections = \
        raster_transformers.fill_in_soil_classes_missing_values_from_nearest_polygons(
            header, green_areas, soil_classes, soil_classes_polygon_masks
        )
    assert corrections == [
        (geometry.Polygon([(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)]), 2),  # row:col=0:1
        (geometry.Polygon([(3, 0), (4, 0), (4, 1), (3, 1), (3, 0)]), 1),  # row:col=1:3
    ]


def test_fill_in_soil_classes_missing_values_from_nearest_polygons_no_neighbors():
    header = geo_data.ElevationHeader(
        col_count=1,
        row_count=1,
        x_ll_corner=0.0,
        y_ll_corner=0.0,
        cell_size=1.0,
        nodata_value=0.0,
    )

    corrections = \
        raster_transformers.fill_in_soil_classes_missing_values_from_nearest_polygons(
            header, numpy.array([[1]]), numpy.array([[0]]), []
        )
    assert corrections == []


def test_fill_in_soil_classes_missing_values_from_nearest_polygons_neighbor_over():
    header = geo_data.ElevationHeader(
        col_count=1,
        row_count=1,
        x_ll_corner=0.0,
        y_ll_corner=0.0,
        cell_size=1.0,
        nodata_value=0.0,
    )
    polygon = geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    corrections = \
        raster_transformers.fill_in_soil_classes_missing_values_from_nearest_polygons(
            header, numpy.array([[1]]), numpy.array([[0]]), [(polygon, 9)]
        )
    assert corrections == [(polygon, 9)]

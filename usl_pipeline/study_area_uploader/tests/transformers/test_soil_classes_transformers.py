from shapely import geometry

from study_area_uploader.transformers import soil_classes_transformers
from usl_lib.shared import geo_data


def bbox_polygon(x1: float, y1: float, x2: float, y2: float) -> geometry.Polygon:
    return geometry.Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)])


def test_transform_soil_classes_as_green_areas_happy_path():
    header = geo_data.ElevationHeader(
        col_count=4,
        row_count=2,
        x_ll_corner=0.0,
        y_ll_corner=0.0,
        cell_size=1.0,
        nodata_value=0.0,
    )

    green_areas_polygon_masks = [
        (bbox_polygon(1, 1, 3, 2), 1),  # Raster mask: [0, 1, 1, 0], (row=0)
        (bbox_polygon(2, 0, 4, 1), 1),  # Raster mask: [0, 0, 1, 1], (row=1)
    ]
    soil_classes_polygon_masks = [
        (bbox_polygon(1, 1, 2, 3), 10),  # This will be used to fill in row:col=1:2
        (bbox_polygon(4, 0, 5, 1), 11),  # This will be used to fill in row:col=1:3
        (bbox_polygon(2, 1, 3, 2), 12),  # This will be ignored as non-green
    ]

    corrections = soil_classes_transformers.transform_soil_classes_as_green_areas(
        header,
        green_areas_polygon_masks,
        soil_classes_polygon_masks,
        non_green_area_soil_classes={12},
    )
    assert len(corrections) == 3
    assert corrections[0][1] == 10
    assert corrections[0][0].equals(bbox_polygon(1, 1, 2, 2))  # trimmed from (1,1,2,3)
    assert corrections[1][1] == 10
    assert corrections[1][0].equals(bbox_polygon(2, 0, 3, 1))  # added from (1,1,3,3)
    assert corrections[2][1] == 11
    assert corrections[2][0].equals(bbox_polygon(3, 0, 4, 1))  # added from (4,0,5,1)

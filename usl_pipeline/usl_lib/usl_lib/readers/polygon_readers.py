import typing

import fiona
from shapely.geometry import Polygon

SHAPE_FILE_POLYGON_TYPE = "Polygon"


def read_polygons_from_shape_file(
    file_path: str,
    crs: str,
    mask_value_feature_property: typing.Optional[str] = None,
    skip_zero_mask_values: bool = True,
) -> list[(Polygon, int)]:
    layer = fiona.open(file_path, crs=crs)
    polygons = []
    for feature in layer:
        feature_type = feature.geometry.type
        if feature_type != SHAPE_FILE_POLYGON_TYPE:
            continue
        fragments = feature.geometry.coordinates
        mask_value = (
            1
            if mask_value_feature_property is None
            else fiona.model.to_dict(feature.properties)[mask_value_feature_property]
        )
        if mask_value == 0 and skip_zero_mask_values:
            continue
        for fragment in fragments:
            polygons.append((Polygon(fragment), mask_value))
    return polygons


def read_polygons_from_text_file(
    file: typing.TextIO,
    support_mask_values: bool = False,
) -> list[(Polygon, int)]:
    lines = file.readlines()

    # Number of polygon lines
    polygon_count = int(lines[0])

    polygon_masks = []
    for i in range(polygon_count):
        line = lines[1 + i]
        parts = line.split()
        mask_value = 1
        point_index_base = 0
        if support_mask_values:
            mask_value = int(parts[0])
            point_index_base = 1
        point_count = int(parts[point_index_base])
        points = []

        for j in range(point_count):
            x = float(parts[point_index_base + 1 + j])
            y = float(parts[point_index_base + 1 + point_count + j])
            points.append((x, y))

        polygon_masks.append((Polygon(points), mask_value))

    return polygon_masks

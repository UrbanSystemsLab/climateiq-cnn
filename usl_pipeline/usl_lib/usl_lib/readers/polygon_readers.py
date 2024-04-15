import typing
from typing import Tuple

from shapely import geometry


def read_polygons_from_text_file(
    file: typing.TextIO, support_mask_values: bool = False
) -> list[Tuple[geometry.Polygon, int]]:
    """Reads polygon information with optional mask values from text file.

    Args:
        file: File text stream to load from.
        support_mask_values: Optional indicator of the presence of polygon mask values
            in the first column in polygon lines. Default case is that mask column is
            not present (in this case mask value 1 will be assigned to every polygon).

    Returns:
        The list of tuples combining polygon with associated mask.
    """
    lines = file.readlines()

    # First line contains the number of polygon lines
    polygon_count = int(lines[0])
    polygons = []

    # The rest of lines have structure of white-space separated numeric values, where
    # first optional column has mask value (if this mode was requested by caller), then
    # the number of points in the polygon and after then all the x-coordinates of
    # polygon points followed by all the y-coordinates.
    for i in range(polygon_count):
        line = lines[1 + i]
        parts = line.split()
        mask_value = 1
        index_base = 0
        if support_mask_values:
            mask_value = int(parts[0])
            index_base = 1
        pnt_count = int(parts[index_base])
        points = []

        for j in range(0, pnt_count):
            x = float(parts[index_base + 1 + j])
            y = float(parts[index_base + 1 + pnt_count + j])
            points.append((x, y))

        polygons.append((geometry.Polygon(points), mask_value))

    return polygons

import typing
from typing import Generator, Tuple

from shapely import geometry


def read_polygons_from_text_file(
    file: typing.TextIO,
) -> Generator[Tuple[geometry.Polygon, int], None, None]:
    """Reads polygon information with optional mask values from text file.

    The algorithm detects if the first column in the file contains mask values that are
    assigned to each polygon. If not, default mask value 1 is used.

    Args:
        file: File text stream to load from.

    Returns:
        The generator of tuples combining polygon with associated mask to iterate over.
    """
    # First line contains the number of polygon lines
    polygon_count = int(file.readline())

    # The rest of lines have structure of white-space separated numeric values, where
    # first optional column has mask value (if this mode was requested by caller), then
    # the number of points in the polygon and after then all the x-coordinates of
    # polygon points followed by all the y-coordinates.
    for i in range(polygon_count):
        line = file.readline()
        parts = line.split()
        # In case the line has an even number of cells, it means that the first cell
        # contains the mask value.
        mask_value_is_present = len(parts) % 2 == 0
        mask_value = 1
        index_base = 0
        if mask_value_is_present:
            mask_value = int(parts[0])
            index_base = 1
        pnt_count = int(parts[index_base])
        points = []

        for j in range(0, pnt_count):
            x = float(parts[index_base + 1 + j])
            y = float(parts[index_base + 1 + pnt_count + j])
            points.append((x, y))

        yield geometry.Polygon(points), mask_value

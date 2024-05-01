from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from rasterio import CRS


@dataclass
class ElevationHeader:
    """Elevation header data.

    Elevation header data class keeping information needed to convert
    geo-spacial coordinates to raster cells and back.

    Args:
        col_count: Number of columns in a raster cell matrix.
        row_count: Number of rows in a raster cell matrix.
        x_ll_corner: X-coordinate of lower-left corner of a raster region.
        y_ll_corner: Y-coordinate of lower-left corner of a raster region.
        cell_size: Size of each raster cell.
        nodata_value: Special data value that indicates that cells having this
            value correspond to missing data.
        crs: Optional value defining Coordinate Reference System.
    """

    col_count: int
    row_count: int
    x_ll_corner: float
    y_ll_corner: float
    cell_size: float
    nodata_value: float
    crs: Optional[CRS] = None


@dataclass
class Elevation:
    header: ElevationHeader
    data: Optional[npt.NDArray[np.float64]] = None


"""The bounding box represented by a tuple (min-x, min-y, max-x, max-y).
"""
BoundingBox = Tuple[float, float, float, float]


def bounding_box_intersection(bbox1: BoundingBox, bbox2: BoundingBox) -> bool:
    """Checks if two bounding boxes have intersection.

    Each bounding box is represented by a tuple (min-x, min-y, max-x, max-y).

    Args:
        bbox1: Bounding box 1.
        bbox2: Bounding box 2.

    Returns:
        Indicator of the intersection of bounding boxes.
    """
    # Intersection check is done separately over each of X- and Y-axis. Along each
    # dimension, we require that minimum bound of one box doesn't happen to be greater
    # than maximum bound of the other box.
    horizontal_overlap = not (bbox1[0] > bbox2[2] or bbox2[0] > bbox1[2])
    vertical_overlap = not (bbox1[1] > bbox2[3] or bbox2[1] > bbox1[3])
    return horizontal_overlap and vertical_overlap


def bounding_box_nesting(outer_bbox: BoundingBox, inner_bbox: BoundingBox) -> bool:
    """Checks if inner bounding box is nested inside the outer one.

    Each bounding box is represented by a tuple (min-x, min-y, max-x, max-y).

    Args:
        outer_bbox: Bounding box that is checked to be the outer one.
        inner_bbox: Bounding box that is checked to be the inner one.

    Returns:
        Indicator of the nesting of the inner bounding box into the outer one.
    """
    # Nesting check is done separately over each of X- and Y-axis.
    horiz_nesting = outer_bbox[0] <= inner_bbox[0] and inner_bbox[2] <= outer_bbox[2]
    vert_nesting = outer_bbox[1] <= inner_bbox[1] and inner_bbox[3] <= outer_bbox[3]
    return horiz_nesting and vert_nesting

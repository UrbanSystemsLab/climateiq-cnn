from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from rasterio import CRS


@dataclass
class ElevationHeader:
    """Elevation header data class keeping information needed to convert
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
    crs: CRS = None


@dataclass
class Elevation:
    header: ElevationHeader
    data: npt.NDArray[np.float64] = None

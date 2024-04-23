from typing import Tuple

import numpy
import numpy.typing as npt
import rasterio
from rasterio import features
from shapely import geometry

from usl_lib.shared import geo_data


def rasterize_polygons(
    header: geo_data.ElevationHeader,
    polygons_with_masks: list[Tuple[geometry.Polygon, int]],
    background_value: int = 0,
) -> npt.NDArray[numpy.float64]:
    """Rasterize polygons to an integer raster matrix.

    Args:
        header: Coordinate system and raster grid information.
        polygons_with_masks: Polygons to rasterize with assigned mask values (mask
            values are used to draw and fill in polygon shape in the raster).
        background_value: Optional background mask used to fill in the raster before
            drawing polygons (default value is 0).

    Returns:
        Matrix with integer raster masks of the size corresponding to a number of rows
        and columns in the cell grid of the study area defined by the header.
    """
    for polygon_mask in polygons_with_masks:
        if polygon_mask[1] == background_value:
            raise ValueError(
                "Polygons with background mask are not allowed: {}".format(polygon_mask)
            )

    min_x = header.x_ll_corner
    min_y = header.y_ll_corner
    cell_size = header.cell_size
    raster_cols = header.col_count
    raster_rows = header.row_count

    if len(polygons_with_masks) == 0:
        return numpy.full((raster_rows, raster_cols), background_value, dtype=int)

    # This transformation is backward one mapping raster cells back to original
    # coordinates (original_x = min_x + raster_x * step, y is similar).
    transform = rasterio.Affine(cell_size, 0, min_x, 0, cell_size, min_y)

    raster_matrix = features.rasterize(
        polygons_with_masks,
        fill=background_value,
        out_shape=(raster_rows, raster_cols),
        transform=transform,
    )

    # Up-down flip is needed to correct the direction of Y-axis which should go from
    # the top to the bottom when raster is visualized.
    return numpy.flipud(raster_matrix)

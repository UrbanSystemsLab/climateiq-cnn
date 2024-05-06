from typing import Tuple

import numpy
import numpy.typing as npt
from shapely import geometry  # import Point, Polygon
import geopandas as gpd

from usl_lib.shared import geo_data


def fill_in_soil_classes_missing_values_from_nearest_polygons(
    elevation_header: geo_data.ElevationHeader,
    green_areas_raster: npt.NDArray[numpy.float64],
    soil_classes_raster: npt.NDArray[numpy.float64],
    soil_classes_polygon_masks: list[Tuple[geometry.Polygon, int]],
) -> list[Tuple[geometry.Polygon, int]]:
    """Generates 1-cell polygons with nearest-neighbor soil classes for missing cells.

    Args:
        elevation_header: Elevation header providing coordinate system and the study
            area region.
        green_areas_raster: Raster with 0/1 masks indicating the presence of green area
            for each raster cell.
        soil_classes_raster: Raster with integer soil classes assigned to each raster
            cell (0 means missing soil class).
        soil_classes_polygon_masks: List of tuples containing polygons with associated
            soil classes.

    Returns:
        The list of tuples containing polygons with associated soil classes for the
        cells in a raster with present green area but missing soil class.
    """
    if len(soil_classes_polygon_masks) == 0:
        return []

    masks = [polygon_mask[1] for polygon_mask in soil_classes_polygon_masks]
    d = {'geometry': [polygon_mask[0] for polygon_mask in soil_classes_polygon_masks]}
    gdf = gpd.GeoDataFrame(d)

    # Prepare coordinates left-upper corner of the study area
    cell_size = elevation_header.cell_size
    min_x = elevation_header.x_ll_corner
    max_y = elevation_header.y_ll_corner + elevation_header.row_count * cell_size

    # Make common 3-dimensional array where both 2d arrays are 2 layers and 3-rd
    # dimension is just 2-element array [<green-area-cell>, <soil-class-cell>]
    # corresponding to the same place in the raster.
    stacked_raster = numpy.dstack((green_areas_raster, soil_classes_raster))

    # Reduce 3-rd dimension to an indicator that green area is present but soil class is
    # 0 (missing value).
    missing_soil_classes = numpy.apply_along_axis(
        lambda gs: gs[0] == 1 and gs[1] == 0, 2, stacked_raster
    )
    soil_classes_corrections: list[Tuple[geometry.Polygon, int]] = []
    missing_rows, missing_cols = numpy.where(missing_soil_classes == 1)
    for (row, col) in zip(missing_rows.tolist(), missing_cols.tolist()):
        # Calculate corners for the cell with missing soil class
        x1 = min_x + col * cell_size
        x2 = x1 + cell_size
        y2 = max_y - row * cell_size
        y1 = y2 - cell_size
        # The middle point of the cell:
        cell_point = geometry.Point((x1 + x2) / 2, (y1 + y2) / 2)
        nearest_polygon_index = gdf.distance(cell_point).idxmin()
        if nearest_polygon_index >= 0:
            nearest_soil_class = masks[nearest_polygon_index]
            point_polygon = geometry.Polygon(
                [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
            )
            soil_classes_corrections.append((point_polygon, nearest_soil_class))

    return soil_classes_corrections

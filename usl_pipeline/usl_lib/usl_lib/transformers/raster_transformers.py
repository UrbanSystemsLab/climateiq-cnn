import logging
from typing import Dict, Tuple

import numpy
import numpy.typing as npt
from shapely import geometry
import geopandas as gpd

from usl_lib.shared import geo_data


def fill_in_soil_classes_missing_values_from_nearest_polygons(
    elevation_header: geo_data.ElevationHeader,
    green_areas_raster: npt.NDArray[numpy.int_],
    soil_classes_raster: npt.NDArray[numpy.int_],
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
    if not soil_classes_polygon_masks:
        return []
    logging.info("Doing gap-filling for raster cells with missing soil classes...")

    masks = [polygon_mask[1] for polygon_mask in soil_classes_polygon_masks]

    # Checking stats:
    mask_unique_values = set(masks)
    for mask_value in mask_unique_values:
        soil_classes_for_mask = numpy.where(
            (green_areas_raster == 1) & (soil_classes_raster == mask_value)
        )
        logging.info(
            "  - Stats: %s green area cells for soil class [%s] were detected",
            len(soil_classes_for_mask[0]),
            mask_value,
        )

    d = {"geometry": [polygon_mask[0] for polygon_mask in soil_classes_polygon_masks]}
    gdf = gpd.GeoDataFrame(d)

    # Prepare coordinates left-upper corner of the study area
    cell_size = elevation_header.cell_size
    min_x = elevation_header.x_ll_corner
    max_y = elevation_header.y_ll_corner + elevation_header.row_count * cell_size

    # The following condition is an indicator that green area is present (value 1) but
    # soil class is missing (value 0).
    missing_soil_classes = (green_areas_raster == 1) & (soil_classes_raster == 0)
    soil_classes_corrections: list[Tuple[geometry.Polygon, int]] = []
    missing_rows, missing_cols = numpy.where(missing_soil_classes)
    logging.info("  - %s missing soil class cells were detected...", len(missing_rows))
    processed_cells = 0
    added_soil_class_counts: Dict[int, int] = {}
    for row, col in zip(missing_rows, missing_cols):
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
            prev_count = added_soil_class_counts.get(nearest_soil_class, 0)
            added_soil_class_counts[nearest_soil_class] = prev_count + 1
        processed_cells = processed_cells + 1
        if processed_cells % 1000 == 0:
            logging.info("  - %s missing cells are filled in so far", processed_cells)

    logging.info("  - %s missing cells were filled in", processed_cells)
    logging.info(
        "  - filled-in cell counts by soil-class values: %s", added_soil_class_counts
    )

    return soil_classes_corrections

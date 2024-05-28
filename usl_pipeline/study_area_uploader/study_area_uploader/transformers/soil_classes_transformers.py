import logging
from typing import Tuple

from shapely import geometry

from usl_lib.shared import geo_data
from usl_lib.transformers import polygon_transformers, raster_transformers


def transform_soil_classes_as_green_areas(
    elevation_header: geo_data.ElevationHeader,
    green_areas_polygons: list[Tuple[geometry.Polygon, int]],
    soil_classes_polygons: list[Tuple[geometry.Polygon, int]],
) -> list[Tuple[geometry.Polygon, int]]:
    """Prepares soil classes data in a way that it can be used as green areas.

    More specifically original green area regions are used as a boundaries for soil
    classes data. Plus on top of the above intersection, for any cells from green area
    raster where soil classes information is missing, the value is assigned from the
    nearest neighbor soil class polygon.

    Args:
        elevation_header: The source of coordinate system and study area region info.
        green_areas_polygons: List of tuples with green area polygons.
        soil_classes_polygons: List of tuples with soil class polygons.

    Returns:
        Corrected soil classes that can be used as green areas.
    """
    logging.info("Starting reconstruction of green areas based on soil classes")
    logging.info(
        "Intersecting %s soil class polygons with green areas",
        len(soil_classes_polygons),
    )
    existing_soil_class_polygons = []
    for p in green_areas_polygons:
        existing_soil_class_polygons.extend(
            polygon_transformers.intersect(
                soil_classes_polygons,
                geometry.MultiPolygon([p[0]]),
            )
        )
    logging.info(
        "  - %s soil class polygons were produced as a result of intersection",
        len(existing_soil_class_polygons),
    )

    logging.info("Green areas raster is getting created...")
    green_areas_raster = polygon_transformers.rasterize_polygons(
        elevation_header, green_areas_polygons
    )
    logging.info("Soil classes raster is getting created...")
    soil_classes_raster = polygon_transformers.rasterize_polygons(
        elevation_header, soil_classes_polygons
    )

    logging.info("Soil classes raster analysis is started...")
    added_soil_class_cells = (
        raster_transformers.fill_in_soil_classes_missing_values_from_nearest_polygons(
            elevation_header,
            green_areas_raster,
            soil_classes_raster,
            soil_classes_polygons,
        )
    )
    logging.info("Green areas reconstruction based on soil classes is done.")

    return existing_soil_class_polygons + added_soil_class_cells

import logging
from typing import Tuple

from shapely import geometry

from usl_lib.shared import geo_data
from usl_lib.transformers import polygon_transformers, raster_transformers


def transform_soil_classes_as_green_areas(
    elevation_header: geo_data.ElevationHeader,
    green_areas_polygons: list[Tuple[geometry.Polygon, int]],
    soil_classes_polygons: list[Tuple[geometry.Polygon, int]],
    non_green_area_soil_classes: set[int] = set(),
) -> list[Tuple[geometry.Polygon, int]]:
    """Prepares soil classes data in a way that it can be used as green areas.

    More specifically original green area regions are used as a boundaries for soil
    classes data. Plus on top of the above intersection, for any cells from green area
    raster where soil classes information is missing, the value is assigned from the
    nearest neighbor soil class polygon.

    Here are formal steps of the algorithm:
      - Exclude from soil class polygons all parts of regions that are outside green
        areas, also filter out polygons with non-green soil classes, and report them as
        polygons to the output;
      - Classify soil classes for all raster cells from green areas that weren't covered
        by any soil classes (including non-green ones) using nearest neighbor approach
        looking for nearest non-green soil class polygons, and report them as 1-cell
        polygons to the output in addition to the previous item.

    Args:
        elevation_header: The source of coordinate system and study area region info.
        green_areas_polygons: List of tuples with green area polygons.
        soil_classes_polygons: List of tuples with soil class polygons.
        non_green_area_soil_classes: Optional set of soil class values that should be
            excluded from green areas.

    Returns:
        Corrected soil classes that can be used as green areas.
    """
    logging.info("Starting reconstruction of green areas based on soil classes")
    green_area_soil_classes_polygons = [
        p for p in soil_classes_polygons if p[1] not in non_green_area_soil_classes
    ]
    logging.info(
        "Intersecting %s soil class polygons with green areas",
        len(green_area_soil_classes_polygons),
    )
    existing_soil_class_polygons: list[Tuple[geometry.Polygon, int]] = []
    for p in green_areas_polygons:
        existing_soil_class_polygons.extend(
            polygon_transformers.intersect(
                green_area_soil_classes_polygons,
                geometry.MultiPolygon([p[0]]),
                skip_logging=True,
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
    # Raster for soil classes is made based on soil_classes_polygons that might include
    # non-green area polygons. This is to avoid creating cells inside the green spaces
    # with missing soil values, which would be interpolated, when we need to leave them
    # as their non-green-area soil type.
    soil_classes_raster = polygon_transformers.rasterize_polygons(
        elevation_header, soil_classes_polygons
    )

    logging.info("Soil classes raster analysis is started...")
    added_soil_class_cells = (
        raster_transformers.fill_in_soil_classes_missing_values_from_nearest_polygons(
            elevation_header,
            green_areas_raster,
            soil_classes_raster,
            green_area_soil_classes_polygons,
        )
    )
    logging.info("Green areas reconstruction based on soil classes is done.")

    return existing_soil_class_polygons + added_soil_class_cells

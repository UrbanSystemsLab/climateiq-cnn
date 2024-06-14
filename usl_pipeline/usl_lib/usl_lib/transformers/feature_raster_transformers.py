from typing import Callable, Optional, Tuple

import numpy
import numpy.typing as npt
from shapely import geometry

from usl_lib.shared import geo_data
from usl_lib.transformers import polygon_transformers, soil_classes_transformers

# Default soil class value that is recognized a non-green area.
DEFAULT_NON_GREEN_AREA_SOIL_CLASS = -9999


def _lookup_soil_class_infiltration_extract_property(
    soil_classes_raster: npt.NDArray[numpy.int_],
    soil_infiltration_configuration: geo_data.InfiltrationConfiguration,
    property_getter: Callable[[geo_data.InfiltrationParams], float],
) -> npt.NDArray[numpy.float32]:
    """For each cell in 2-d matrix looks up for infiltration property by soil class."""
    property_values_over_all_soil_classes = [
        property_getter(item) for item in soil_infiltration_configuration.items
    ]
    max_property_value = max(property_values_over_all_soil_classes)
    lookup_map = soil_infiltration_configuration.as_map()
    lookup_np_func = numpy.vectorize(
        lambda soil_class: (
            0.0
            if soil_class not in lookup_map
            else property_getter(lookup_map[soil_class])
        ),
        cache=True,
    )
    return lookup_np_func(soil_classes_raster) / max_property_value


def transform_to_feature_raster_layers(
    elevation: geo_data.Elevation,
    boundaries_polygons: Optional[list[Tuple[geometry.Polygon, int]]],
    buildings_polygons: list[Tuple[geometry.Polygon, int]],
    green_areas_polygons: list[Tuple[geometry.Polygon, int]],
    soil_classes_polygons: list[Tuple[geometry.Polygon, int]],
    soil_infiltration_configuration: geo_data.InfiltrationConfiguration,
) -> npt.NDArray[numpy.float32]:
    """Rasterize layers of feature matrix and sanitize the values.

    Args:
        elevation: The elevation data.
        boundaries_polygons: List of tuples with boundaries polygons.
        buildings_polygons: List of tuples with buildings polygons.
        green_areas_polygons: List of tuples with green area polygons.
        soil_classes_polygons: List of tuples with soil class polygons.
        soil_infiltration_configuration: Soil infiltration configuration which is used
            to look up infiltration properties for soil classes.

    Returns:
        3-dimensional matrix where 2 first dimensions are spacial (Y and X axes) and
        third one combines 8 feature layers:
         - elevation data,
         - elevation mask,
         - building footprint mask,
         - green area mask,
         - hydraulic conductivity,
         - wetting front suction head,
         - effective porosity,
         - effective saturation.
    """
    nodata_value = elevation.header.nodata_value
    if elevation.data is None:
        raise ValueError("Elevation data missing")
    elevation_raster = elevation.data.astype(dtype=numpy.float32)
    elevation_mask_raster = numpy.zeros(elevation_raster.shape, dtype=int)
    # setting mask raster to 1 for cells where elevation value is present (!=NODATA)
    elevation_mask_raster[elevation_raster != nodata_value] = 1
    empty_raster = numpy.zeros(elevation_raster.shape, dtype=numpy.float32)
    # If boundaries data is not present we ignore any boundaries-related corrections.
    if boundaries_polygons is not None:
        # If boundaries data is present but empty it means that the whole matrix should
        # be marked as NODATA.
        boundaries_raster = (
            empty_raster
            if len(boundaries_polygons) == 0
            else polygon_transformers.rasterize_polygons(
                elevation.header, boundaries_polygons
            )
        )
        # clearing elevation values and presence mask for cells outside boundaries
        elevation_raster[boundaries_raster == 0] = numpy.float32(nodata_value)
        elevation_mask_raster[boundaries_raster == 0] = 0

    buildings_raster = empty_raster
    if len(buildings_polygons) > 0:
        buildings_raster = polygon_transformers.rasterize_polygons(
            elevation.header, buildings_polygons
        ).astype(dtype=numpy.float32)
        buildings_raster[elevation_mask_raster == 0] = 0
    green_areas_mask_raster = empty_raster
    hydraulic_conductivity_raster = empty_raster
    wetting_front_suction_head_raster = empty_raster
    effective_porosity_raster = empty_raster
    effective_saturation_raster = empty_raster
    if len(green_areas_polygons) > 0 and len(soil_classes_polygons) > 0:
        corrected_soil_classes_polygons = (
            soil_classes_transformers.transform_soil_classes_as_green_areas(
                elevation.header,
                green_areas_polygons,
                soil_classes_polygons,
                non_green_area_soil_classes={DEFAULT_NON_GREEN_AREA_SOIL_CLASS},
            )
        )
        soil_classes_raster = polygon_transformers.rasterize_polygons(
            elevation.header, corrected_soil_classes_polygons
        )
        soil_classes_raster[elevation_mask_raster == 0] = 0
        green_areas_mask_raster[soil_classes_raster != 0] = 1
        hydraulic_conductivity_raster = (
            _lookup_soil_class_infiltration_extract_property(
                soil_classes_raster,
                soil_infiltration_configuration,
                lambda param: param.hydraulic_conductivity,
            )
        )
        wetting_front_suction_head_raster = (
            _lookup_soil_class_infiltration_extract_property(
                soil_classes_raster,
                soil_infiltration_configuration,
                lambda param: param.wetting_front_suction_head,
            )
        )
        effective_porosity_raster = _lookup_soil_class_infiltration_extract_property(
            soil_classes_raster,
            soil_infiltration_configuration,
            lambda param: param.effective_porosity,
        )
        effective_saturation_raster = _lookup_soil_class_infiltration_extract_property(
            soil_classes_raster,
            soil_infiltration_configuration,
            lambda param: param.effective_saturation,
        )

    return numpy.dstack(
        (
            elevation_raster.astype(dtype=numpy.float32),
            elevation_mask_raster.astype(dtype=numpy.float32),
            buildings_raster.astype(dtype=numpy.float32),
            green_areas_mask_raster.astype(dtype=numpy.float32),
            hydraulic_conductivity_raster,
            wetting_front_suction_head_raster,
            effective_porosity_raster,
            effective_saturation_raster,
        )
    )

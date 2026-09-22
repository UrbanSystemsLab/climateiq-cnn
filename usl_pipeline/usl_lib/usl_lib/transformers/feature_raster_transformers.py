from typing import Tuple

import numpy
import numpy.typing as npt
from shapely import geometry

from usl_lib.shared import geo_data
from usl_lib.storage import metastore
from usl_lib.transformers import polygon_transformers, soil_classes_transformers

# Default soil class value that is recognized a non-green area.
DEFAULT_NON_GREEN_AREA_SOIL_CLASS = -9999


def _build_soil_class_infiltration_layers(
    soil_classes_raster: npt.NDArray[numpy.int_],
    soil_infiltration_configuration: geo_data.InfiltrationConfiguration,
) -> npt.NDArray[numpy.float32]:
    """Converts 2d matrix to 3d one looking up infiltration properties by soil class."""
    infiltration_layers = numpy.zeros(
        soil_classes_raster.shape + (4,), dtype=numpy.float32
    )

    config_items = soil_infiltration_configuration.items
    max_conductivity = max(item.hydraulic_conductivity for item in config_items)
    max_wetting = max(item.wetting_front_suction_head for item in config_items)
    max_porosity = max(item.effective_porosity for item in config_items)
    max_saturation = max(item.effective_saturation for item in config_items)

    for config_item in config_items:
        infiltration_layers[soil_classes_raster == config_item.soil_class] = [
            config_item.hydraulic_conductivity / max_conductivity,
            config_item.wetting_front_suction_head / max_wetting,
            config_item.effective_porosity / max_porosity,
            config_item.effective_saturation / max_saturation,
        ]
    return infiltration_layers


def compute_slope_feature(elevation: geo_data.Elevation) -> npt.NDArray[numpy.float32]:
    """Computes the slope (in percent rise) from the DEM in elevation.data.

    Gradients are computed along x and y using the raster's cell size, and slope is
    the Euclidean norm of those gradients, expressed as a percentage. Extreme slope
    values (above the 99.5th percentile of valid cells) are capped to reduce the
    influence of DEM noise/artifacts on the normalisation range computed downstream.
    NoData cells are excluded from gradient/percentile computation and restored as
    NoData in the output.

    Note: numpy.gradient uses centered differences, so a cell adjacent to NoData
    produces a NaN gradient (and thus NaN slope) even though the cell's own
    elevation is valid -- this NaN would otherwise silently propagate into
    numpy.percentile (which returns NaN if any input is NaN), making the cap a
    no-op for any chunk where NoData borders valid terrain, which is most of
    them. nanpercentile plus a post-hoc NaN sweep close both gaps.

    Args:
        elevation: The elevation data.

    Returns:
        2-dimensional matrix (same shape as elevation.data) of slope values in percent
        rise, with NoData cells (and any gradient-adjacent NaNs) set to
        elevation.header.nodata_value.
    """
    nodata_value = elevation.header.nodata_value
    dem = elevation.data.astype(dtype=numpy.float32)
    mask = (dem == nodata_value) | numpy.isnan(dem)

    cell_size = elevation.header.cell_size
    dem_masked = numpy.where(mask, numpy.nan, dem)
    dz_dx = numpy.gradient(dem_masked, axis=1) / cell_size
    dz_dy = numpy.gradient(dem_masked, axis=0) / cell_size
    slope = numpy.sqrt(dz_dx**2 + dz_dy**2) * 100.0

    valid_slope_values = slope[~mask & ~numpy.isnan(slope)]
    if valid_slope_values.size > 0:
        cap = numpy.percentile(valid_slope_values, 99.5)
        slope = numpy.where(slope > cap, cap, slope)

    slope[mask | numpy.isnan(slope)] = nodata_value
    return slope.astype(dtype=numpy.float32)


def transform_to_feature_raster_layers(
    elevation: geo_data.Elevation,
    boundaries_polygons: list[Tuple[geometry.Polygon, int]] | None,
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
        third one combines 9 feature layers:
         - elevation data,
         - elevation mask,
         - building footprint mask,
         - green area mask,
         - hydraulic conductivity,
         - wetting front suction head,
         - effective porosity,
         - effective saturation,
         - slope (percent rise).
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
    infiltration_raster_layers = numpy.zeros(
        elevation_raster.shape + (4,), dtype=numpy.float32
    )
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
        infiltration_raster_layers = _build_soil_class_infiltration_layers(
            soil_classes_raster, soil_infiltration_configuration
        )

    # compute_slope_feature only knows about the raw DEM's own NoData cells --
    # it has no visibility into the boundary polygon applied above, so cells
    # outside the study area boundary (elevation_mask_raster == 0) would
    # otherwise carry a real, computed slope value while every other channel
    # correctly shows NoData there. Apply the same final validity mask
    # elevation/buildings/green-areas already use, for consistency.
    slope_raster = compute_slope_feature(elevation)
    slope_raster[elevation_mask_raster == 0] = numpy.float32(nodata_value)

    return numpy.dstack(
        (
            elevation_raster.astype(dtype=numpy.float32),
            elevation_mask_raster.astype(dtype=numpy.float32),
            buildings_raster.astype(dtype=numpy.float32),
            green_areas_mask_raster.astype(dtype=numpy.float32),
            infiltration_raster_layers,
            slope_raster.astype(dtype=numpy.float32),
        )
    )


def rescale_feature_matrix(
    feature_matrix: npt.NDArray,
    study_area_metadata: metastore.StudyArea,
) -> None:
    """Performs scaling of elevation (and slope, if present) in the feature matrix.

    Note: This logic updates feature_matrix values in place.

    Args:
        feature_matrix: Feature matrix to perform scaling for.
        study_area_metadata: Study Area metadata containing min/max elevation (and
            optionally slope) values that are used for scaling.
    """
    elevation_min = study_area_metadata.elevation_min
    elevation_max = study_area_metadata.elevation_max
    if elevation_min is None or elevation_max is None:
        raise ValueError("Elevation min/max values are not set in study area metadata")

    elevation_data = feature_matrix[:, :, 0]
    presence_mask = feature_matrix[:, :, 1]
    elevation_data[presence_mask == 1] = (
        elevation_data[presence_mask == 1] - elevation_min
    ) / (elevation_max - elevation_min)
    elevation_data[presence_mask != 1] = -1

    slope_min = study_area_metadata.slope_min
    slope_max = study_area_metadata.slope_max
    if (
        slope_min is not None
        and slope_max is not None
        and feature_matrix.shape[2] > 8
    ):
        slope_data = feature_matrix[:, :, 8]
        slope_data[presence_mask == 1] = (
            slope_data[presence_mask == 1] - slope_min
        ) / (slope_max - slope_min)
        slope_data[presence_mask != 1] = -1

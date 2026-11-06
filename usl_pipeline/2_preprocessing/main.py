"""Stage 2 Cloud Functions: re-exports the entry points."""

# Native libraries load in this order.
import netCDF4  # noqa: F401
import numpy  # noqa: F401
import xarray  # noqa: F401
import rasterio  # noqa: F401
from shapely import geometry  # noqa: F401
import wrf  # noqa: F401

from preprocessing.heat.features import build_heat_feature_matrix
from preprocessing.heat.labels import (
    build_wrf_label_matrix,
    build_wrf_label_matrix_http,
)
from preprocessing.heat.scenarios import write_heat_scenario_config_metadata
from preprocessing.heat.study_area_chunks import (
    build_and_upload_study_area_chunk,
    build_and_upload_study_area_chunk_http,
)
from preprocessing.feature_matrix import build_feature_matrix, build_feature_matrix_http
from preprocessing.feature_rescaling import rescale_feature_matrices
from preprocessing.flood_features import build_flood_feature_matrix
from preprocessing.flood_labels import process_citycat_outputs
from preprocessing.flood_scenarios import (
    delete_flood_scenario_metadata,
    write_flood_scenario_metadata_and_features,
)
from preprocessing.study_area_metadata import write_study_area_metadata

__all__ = [
    "build_and_upload_study_area_chunk",
    "build_and_upload_study_area_chunk_http",
    "build_feature_matrix",
    "build_feature_matrix_http",
    "build_flood_feature_matrix",
    "build_heat_feature_matrix",
    "build_wrf_label_matrix",
    "build_wrf_label_matrix_http",
    "delete_flood_scenario_metadata",
    "process_citycat_outputs",
    "rescale_feature_matrices",
    "write_flood_scenario_metadata_and_features",
    "write_heat_scenario_config_metadata",
    "write_study_area_metadata",
]

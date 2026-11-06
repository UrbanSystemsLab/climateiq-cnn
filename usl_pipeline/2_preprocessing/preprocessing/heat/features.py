"""Builds heat feature matrices from WPS outputs."""

import datetime
import logging
import pathlib
import re
from typing import IO, Tuple
from google.cloud import firestore
from google.cloud import storage
import functions_framework
import numpy
from numpy.typing import NDArray
import xarray
from usl_lib.shared import wps_data
from usl_lib.shared.function_runtime import (
    retry_and_report_errors as _retry_and_report_errors,
)
from usl_lib.storage import cloud_storage
from usl_lib.storage import file_names
from usl_lib.storage import metastore
from preprocessing.chunks import _write_as_npy, _write_chunk_metastore_error
from preprocessing.feature_metadata import FeatureMetadata


@functions_framework.cloud_event
@_retry_and_report_errors(
    lambda cloud_event, exc: _write_chunk_metastore_error(
        cloud_event.data["name"], str(exc)
    )
)
def build_heat_feature_matrix(cloud_event: functions_framework.CloudEvent) -> None:
    """Builds heat feature matrices when a WPS output file is uploaded."""
    chunk_path = cloud_event.data["name"]
    if not re.search(file_names.WPS_DOMAIN3_NC_REGEX, chunk_path):
        logging.debug("Skipping non-heat file %s", chunk_path)
        return

    _build_heat_feature_matrix(
        cloud_event.data["bucket"],
        chunk_path,
        cloud_storage.FEATURE_CHUNKS_BUCKET,
    )


def _build_heat_feature_matrix(
    bucket_name: str, chunk_path: str, output_bucket: str
) -> None:
    """Builds heat (WRF) feature matrices from one uploaded WPS output file."""
    storage_client = storage.Client()
    chunk_blob = storage_client.bucket(bucket_name).blob(chunk_path)

    with chunk_blob.open("rb") as chunk:
        feature_matrices, metadata = _build_wps_feature_matrices(chunk)
        # Write a separate file for each variable type
        # (spatial, spatiotemporal, lu_index).
        for var_type, feature_matrix in feature_matrices.items():
            feature_path = pathlib.PurePosixPath(chunk_path)
            feature_path_parent = feature_path.parent
            feature_file_name = (
                feature_path_parent / var_type.value / feature_path.name
            ).with_suffix(".npy")
            feature_blob = storage_client.bucket(output_bucket).blob(
                str(feature_file_name)
            )
            _write_as_npy(feature_blob, feature_matrix)
            _write_wps_chunk_metastore_entry(chunk_blob, feature_blob, metadata)


def _build_wps_feature_matrices(
    fd: IO[bytes],
) -> Tuple[dict[wps_data.VarType, NDArray], FeatureMetadata]:
    # Ignore type checker error - BytesIO inherits from expected type BufferedIOBase
    # https://shorturl.at/lk4om
    matrices = {}
    with xarray.open_dataset(fd, engine="h5netcdf") as ds:  # type: ignore
        # Dropp the existing 'Time' coordinate if present, to avoid conflict
        if "Time" in ds.coords:
            ds = ds.drop_vars("Time")

        # Assign the 'Time' coordinate
        if "Times" in ds:
            ds = ds.assign_coords(Time=ds.Times)

        # Derive non-native variables and assign to dataset
        ds = _compute_custom_wps_variables(ds)

        for var_type, vars in wps_data.ML_REQUIRED_VARS.items():
            features_components = [numpy.array([])] * len(vars)

            # Apply feature engineering and build features matrix
            for i, var in enumerate(vars):
                features_components[i] = _process_wps_feature(
                    feature=ds.data_vars[var.name],
                    var_config=wps_data.VAR_CONFIGS[var],
                )

            matrices[var_type] = numpy.dstack(features_components)

        # Get snapshot time
        snapshot_time = ds.Times.values[0].astype(str)

    return matrices, FeatureMetadata(
        time=datetime.datetime.fromisoformat(snapshot_time)
    )


def _compute_custom_wps_variables(dataset: xarray.Dataset) -> xarray.Dataset:
    """Computes additional, non-native variables for WPS datasets.

    Args:
      dataset: The xarray dataset containing all WPS data and its metadata.

    Returns:
      A new xarray dataset with newly derived variables.
    """
    dataset = _compute_wind_components(dataset)
    # TODO: compute solar time
    dataset = _compute_solar_time_components(dataset)

    return dataset


def _compute_wind_components(dataset: xarray.Dataset) -> xarray.Dataset:
    # WSPD (wind speed) at FNL level 0 (~10m)
    # WDIR_SIN (wind direction_sine component) at FNL level 0 (~10m)
    # WDIR_COS (wind direction_cosine component) at FNL level 0 (~10m)
    if all(var in dataset.keys() for var in ["UU", "VV"]):
        uu = dataset.data_vars["UU"]
        vv = dataset.data_vars["VV"]
        uu_centered = (uu[:, :, :, :-1] + uu[:, :, :, 1:]) / 2
        vv_centered = (vv[:, :, :-1, :] + vv[:, :, 1:, :]) / 2
        wind_speed = numpy.sqrt(uu_centered.values**2 + vv_centered.values**2)
        wind_direction = (
            270 - numpy.degrees(numpy.arctan2(vv_centered.values, uu_centered.values))
        ) % 360

        def _direction_to_sine(degrees):
            radians = (degrees / 360.0) * 2 * numpy.pi
            return numpy.sin(radians)

        def _direction_to_cosine(degrees):
            radians = (degrees / 360.0) * 2 * numpy.pi
            return numpy.cos(radians)

        wind_direction_sin = _direction_to_sine(wind_direction)
        wind_direction_cos = _direction_to_cosine(wind_direction)

        new_dims = ["Time", "num_metgrid_levels", "south_north", "west_east"]

        dataset = dataset.assign(
            WSPD=xarray.DataArray(wind_speed, coords=vv_centered.coords, dims=new_dims)
        )
        dataset = dataset.assign(
            WDIR_SIN=xarray.DataArray(
                wind_direction_sin, coords=vv_centered.coords, dims=new_dims
            )
        )
        dataset = dataset.assign(
            WDIR_COS=xarray.DataArray(
                wind_direction_cos, coords=vv_centered.coords, dims=new_dims
            )
        )

    return dataset


def _compute_solar_time_components(dataset: xarray.Dataset) -> xarray.Dataset:
    """Compute solar time components (sine and cosine) for the dataset's longitudes.

    Args:
    - dataset: The xarray dataset containing longitude, latitude, and time information.
    - use_gcp_time: A boolean flag indicating whether the Times are in Unix time format.

    Returns:
    - The dataset with added solar time sine and cosine components.
    """
    # Solar Time Computation for every Longitude of the City
    if all(var in dataset.keys() for var in ["XLONG_M", "XLAT_M", "Times"]):
        longitudes = dataset["XLONG_M"][0, :, :]  # Extract the longitudes
        times = dataset["Times"]  # Extract the time variable

        # Convert preformatted string-based times to nanosecond precision datetime
        times = xarray.DataArray(
            [
                numpy.datetime64("".join(t.astype(str)).replace("_", "T"), "ns")
                for t in times.values
            ],
            dims=["Time"],
        )

        # Extract hours and minutes in UTC
        utc_hours = times.dt.hour
        utc_minutes = times.dt.minute
        utc_hours_minutes = utc_hours + utc_minutes / 60.0

        # Function to calculate solar time from UTC and longitude
        def calculate_solar_time(utc_time, longitude):
            return (utc_time + longitude / 15) % 24

        # Calculate solar times for all longitudes
        solar_times = xarray.apply_ufunc(
            calculate_solar_time, utc_hours_minutes, longitudes, vectorize=True
        )

        # Functions to convert solar time to sine and cosine values
        def time_to_sine(hours):
            radians = (hours / 24.0) * 2 * numpy.pi
            return numpy.sin(radians)

        def time_to_cosine(hours):
            radians = (hours / 24.0) * 2 * numpy.pi
            return numpy.cos(radians)

        # Apply sine and cosine transformations
        solar_time_sin = xarray.apply_ufunc(time_to_sine, solar_times, vectorize=True)
        solar_time_cos = xarray.apply_ufunc(time_to_cosine, solar_times, vectorize=True)

        # Assign new variables for sine and cosine of solar time
        new_dims = ["Time", "south_north", "west_east"]
        dataset = dataset.assign(
            SOLAR_TIME_SIN=xarray.DataArray(
                solar_time_sin, coords=dataset["XLAT_M"].coords, dims=new_dims
            )
        )
        dataset = dataset.assign(
            SOLAR_TIME_COS=xarray.DataArray(
                solar_time_cos, coords=dataset["XLAT_M"].coords, dims=new_dims
            )
        )

    return dataset


def _process_wps_feature(
    feature: xarray.DataArray, var_config: wps_data.VarConfig
) -> NDArray:
    """Performs a series of data transforms on a WPS variable.

    Args:
      feature: The xarray.DataArray containing the feature, its dimensions,
      and metadata
      var_config: The dict entry to ML_REQUIRED_VARS_REPO from wps_data.py containing
      the metadata that will be used to determine what feature engineering processing
      should be applied

    Returns:
      A new numpy array with transforms applied according to rules
      for each variable defined in: https://shorturl.at/W6nzY
    """
    snapshot_time = feature.coords.get("Time")
    feature = feature.transpose("west_east", "south_north", ...)
    if "num_metgrid_levels" in feature.dims:
        feature = feature.isel(num_metgrid_levels=0)
    if "z-dimension0012" in feature.dims and snapshot_time:
        snapshot_datetime = datetime.datetime.strptime(
            snapshot_time.values[0].decode(), "%Y-%m-%d_%H:%M:%S"
        )
        feature = feature.isel({"z-dimension0012": snapshot_datetime.month - 1})
    feature = feature.isel(Time=0)

    feature_values = feature.values

    if var_config.get("unit") == wps_data.Unit.PERCENTAGE:
        feature_values = _convert_to_decimal(feature_values)

    scaling_config = var_config.get("scaling")
    if (
        scaling_config is not None
        and scaling_config.get("type") == wps_data.ScalingType.GLOBAL
    ):
        scaling_min = scaling_config.get("min")
        scaling_max = scaling_config.get("max")
        feature_values = _apply_minmax_scaler(feature_values, scaling_min, scaling_max)

    return feature_values


def _convert_to_decimal(x):
    return x / 100


def _apply_minmax_scaler(x, minx, maxx):
    # Clip values if they go out of bounds of the specified min and max
    x[x > maxx] = maxx
    x[x < minx] = minx

    return (x - minx) / (maxx - minx)


def _write_wps_chunk_metastore_entry(
    chunk_blob: storage.Blob, feature_blob: storage.Blob, metadata: FeatureMetadata
) -> None:
    """Updates the metastore with new information for the given wps chunks.

    Writes a Firestore entry for the new feature matrix chunk.

    Args:
      chunk_blob: The GCS blob containing the raw chunk.
      feature_blob: The GCS blob containing the feature tensor for the chunk.
      metadata: Additional metadata describing the features.
    """
    db = firestore.Client()

    # Pick out the study_area and file/chunk name from GCS path
    # File names can be in the form <study_area_name>/<file_name>
    # or <study_area_name>/<some_sub_folder>/<file_name>
    file_path = pathlib.PurePosixPath(chunk_blob.name)
    file_path_parts = file_path.parts
    study_area_name = file_path_parts[0]
    chunk_name = file_path.stem  # name of file w/o extension

    # Sanitize unsupported characters for firestore
    doc_id = chunk_name.replace(".", "_").replace(":", "_")

    metastore.StudyAreaTemporalChunk(
        id_=doc_id,
        raw_path=f"gs://{chunk_blob.bucket.name}/{chunk_blob.name}",
        feature_matrix_path=f"gs://{feature_blob.bucket.name}/{feature_blob.name}",
        time=metadata.time,
        # Remove any errors from previous failed retries which have now succeeded.
        error=firestore.DELETE_FIELD,
    ).merge(db, study_area_name)

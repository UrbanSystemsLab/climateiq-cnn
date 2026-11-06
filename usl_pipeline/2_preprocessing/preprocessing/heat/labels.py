"""Builds WRF label matrices."""

import datetime
import pathlib
import re
from typing import IO, Tuple
from google.cloud import firestore
from google.cloud import storage
import flask
import functions_framework
import netCDF4
import numpy
from numpy.typing import NDArray
import xarray
import wrf
from usl_lib.shared.function_runtime import error_to_response as _error_to_response
from usl_lib.shared.function_runtime import (
    retry_and_report_errors as _retry_and_report_errors,
)
from usl_lib.storage import cloud_storage
from usl_lib.storage import file_names
from usl_lib.storage import metastore
from preprocessing.chunks import _write_as_npy
from preprocessing.feature_metadata import FeatureMetadata


@functions_framework.cloud_event
@_retry_and_report_errors()
def build_wrf_label_matrix(cloud_event: functions_framework.CloudEvent) -> None:
    """Builds a label matrix when a set of simulation output files is uploaded.

    This function is triggered when files containing simulation data are uploaded to
    GCS. It produces a label matrix for the sim data and writes that label matrix to
    another GCS bucket for eventual use in model training and prediction.
    """
    _build_wrf_label_matrix(
        cloud_event.data["bucket"],
        cloud_event.data["name"],
        cloud_storage.LABEL_CHUNKS_BUCKET,
    )


@functions_framework.http
@_error_to_response
def build_wrf_label_matrix_http(request: flask.Request) -> flask.Response:
    """Builds a label matrix when a set of simulation output files is uploaded.

    This function is triggered when files containing simulation data are uploaded to
    GCS. It produces a label matrix for the sim data and writes that label matrix to
    another GCS bucket for eventual use in model training and prediction.
    """
    request_json = request.get_json()
    _build_wrf_label_matrix(
        request_json["bucket"],
        request_json["name"],
        cloud_storage.LABEL_CHUNKS_BUCKET,
    )
    return flask.jsonify({"message": "Label matrix built."})


def _build_wrf_label_matrix(
    bucket_name: str, chunk_name: str, output_bucket: str
) -> None:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    chunk_blob = bucket.blob(chunk_name)

    with chunk_blob.open("rb") as chunk:
        # Heat (WRF) - treat one WRF output file as one chunk
        if re.search(file_names.WRF_DOMAIN3_NC_REGEX, chunk_name):
            label_matrix, metadata = _process_wrf_label_and_metadata(chunk)

            # Current wrfout files don't include a .nc file extension, but handle both
            # cases just in case
            if chunk_name.endswith(".nc"):
                label_file_name = chunk_name.replace(".nc", ".npy")
            else:
                label_file_name = chunk_name + ".npy"

            label_blob = storage_client.bucket(output_bucket).blob(str(label_file_name))

            _write_as_npy(label_blob, label_matrix)

            # TODO: Can consider using a different file to trigger Simulation metastore
            # write to reduce duplicated metastore entry updates. Currently, we will
            # upsert Simulation entry for every wrfout.d03 file processed (even though
            # Simulation entry is the same for every wrfout chunk).
            _write_wrf_simulation_metastore_entry(chunk_blob, metadata)
            _write_wrf_label_chunk_metastore_entry(chunk_blob, label_blob, metadata)
        else:
            raise ValueError(f"Unexpected file {chunk_name}")


def _process_wrf_label_and_metadata(fd: IO[bytes]) -> Tuple[NDArray, FeatureMetadata]:
    nc_bytes = fd.read()
    label_components = []
    with netCDF4.Dataset("in_memory.nc", memory=nc_bytes) as nc:
        # Variable names are listed here according to the formatting
        # specified by wrf-python documentation
        # https://wrf-python.readthedocs.io/en/latest/user_api/generated/wrf.getvar.html#wrf-getvar
        vars_to_derive = ["rh2", "T2", "wspd_wdir10"]
        for var in vars_to_derive:
            if var == "wspd_wdir10":
                # TODO: Split wind dir to sin/cos components
                wspd10, wdir10 = wrf.getvar(nc, "wspd_wdir10")
                # Convert wind direction from degrees to radians
                wdir10_rad = numpy.deg2rad(wdir10)
                # Split wind direction into sin and cos components
                wind_dir_sin = numpy.sin(wdir10_rad)
                wind_dir_cos = numpy.cos(wdir10_rad)
                # Append wind speed and sin/cos of wind direction as labels
                label_components.append(numpy.swapaxes(wspd10, 0, 1))  # wind speed
                label_components.append(
                    numpy.swapaxes(wind_dir_sin, 0, 1)
                )  # sin(wind_dir)
                label_components.append(
                    numpy.swapaxes(wind_dir_cos, 0, 1)
                )  # cos(wind_dir)
            else:
                label = wrf.getvar(nc, var)
                # Ensure order of dimension axes are: (west_east, south_north)
                label_components.append(numpy.swapaxes(label, 0, 1))

        ds = xarray.open_dataset(xarray.backends.NetCDF4DataStore(nc))
        snapshot_time = ds.Times.values[0].decode("utf-8")

    labels_matrix = numpy.dstack(label_components)
    return labels_matrix, FeatureMetadata(
        time=datetime.datetime.fromisoformat(snapshot_time)
    )


def _write_wrf_simulation_metastore_entry(
    file_blob: storage.Blob, metadata: FeatureMetadata
) -> None:
    db = firestore.Client()

    # Pick out the study_area and file/chunk name from GCS path
    # File names should be in the form:
    # <study_area_name>/<some_sub_folder>/<file_name>
    file_path = pathlib.PurePosixPath(file_blob.name)
    file_path_parts = file_path.parts
    study_area_name = file_path_parts[0]
    config_path = _construct_heat_config_path(file_path, metadata)

    # Enter the simulation in our metastore.
    metastore.Simulation(
        gcs_prefix_uri=f"gs://{file_blob.bucket.name}/{study_area_name}",
        simulation_type=metastore.SimulationType.WRF,
        study_area=metastore.StudyArea.get_ref(db, study_area_name),
        configuration=metastore.HeatScenarioConfig.get_ref(db, config_path),
    ).set(db)


def _write_wrf_label_chunk_metastore_entry(
    chunk_blob: storage.Blob, label_blob: storage.Blob, metadata: FeatureMetadata
) -> None:
    """Updates the metastore with new information for the given wrf chunks.

    Args:
      chunk_blob: The GCS blob containing the raw chunk.
      label_blob: The GCS blob containing the label tensor for the chunk.
      metadata: Additional metadata describing the features.
    """
    db = firestore.Client()

    # Pick out the study_area and file/chunk name from GCS path
    # File names should be in the form:
    # <study_area_name>/<some_sub_folder>/<file_name>
    file_path = pathlib.PurePosixPath(chunk_blob.name)
    file_path_parts = file_path.parts
    study_area_name = file_path_parts[0]
    config_path = _construct_heat_config_path(file_path, metadata)

    metastore.SimulationLabelTemporalChunk(
        gcs_uri=f"gs://{label_blob.bucket.name}/{label_blob.name}",
        time=metadata.time,
    ).set(db, study_area_name, config_path)


def _construct_heat_config_path(
    file_path: pathlib.PurePosixPath, metadata: FeatureMetadata
) -> str:
    # Time should always be extracted as feature metadata
    # but needs to be optional field to allow for class reuse for flood.
    assert metadata.time is not None
    return f"{str(file_path.parent)}/Heat_Data_{metadata.time.year}.txt"

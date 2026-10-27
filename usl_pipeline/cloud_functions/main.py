import dataclasses
import datetime
import functools
import io
import logging
import os
import pathlib
import re
import sys
import tarfile
import tempfile
import time
import traceback
from typing import BinaryIO, Callable, IO, TextIO, Tuple
from google.api_core import exceptions
from google.cloud import error_reporting
from google.cloud import firestore
from google.cloud import storage
import flask
import functions_framework
import netCDF4
import numpy
from numpy.typing import NDArray
import xarray
import rasterio
from shapely import geometry
import wrf

from usl_lib.chunkers import raster_chunkers
from usl_lib.readers import simulation_readers
from usl_lib.readers import config_readers
from usl_lib.readers import elevation_readers, polygon_readers
from usl_lib.shared import geo_data, wps_data
from usl_lib.storage import cloud_storage
from usl_lib.storage import file_names
from usl_lib.storage import metastore
from usl_lib.transformers import feature_raster_transformers

# How long to accept cloud function invocations after the triggering
# event. It's up to GCP how frequently withing this period the cloud
# function is retried. This also needs to include lag for the length
# of time between when an event is triggered and when the cloud
# function actually runs, which can be a bit of time if there are a
# lot of triggers.
# Added
_MAX_RETRY_SECONDS = 60 * 60

# The vector is long enough to express rainfall every five minutes for up to three days.
_RAINFALL_VECTOR_LENGTH = (60 // 5) * 24 * 3

# Storage bucket file name extension that should trigger feature matrix rescaling.
_FEATURE_SCALING_TRIGGER_SUFFIX = ".scale_trigger"


@dataclasses.dataclass(slots=True)
class FeatureMetadata:
    """Additional information about the extracted features.

    Attributes:
        elevation_min: The lowest elevation height encountered.
        elevation_max: The highest elevation height encountered.
        slope_min: The lowest slope value encountered (percent rise).
        slope_max: The highest slope value encountered (percent rise).
    """

    elevation_min: float | None = None
    elevation_max: float | None = None
    slope_min: float | None = None
    slope_max: float | None = None
    chunk_size: int | None = None
    time: datetime.datetime | None = None


def _retry_and_report_errors(
    error_reporter_func: (
        Callable[[functions_framework.CloudEvent, Exception], None] | None
    ) = None,
) -> Callable[
    [Callable[[functions_framework.CloudEvent], None]],
    Callable[[functions_framework.CloudEvent], None],
]:
    """Adds error reporting and retry logic to a cloud function.

    If the decorated function raises an exception, the decorator will:
    - Log the exception to stderr, which will make the error appear in cloud function
      logs.
    - Log the exception to the GCP Error Reporter service.
    - If the caller supplies an `error_reporter_func` argument, calls that function with
      the cloud event and exception object, allowing callers to add additional custom
      error handling logic.

    If enough time has passed since the cloud function's initial invocation, then the
    decorated function will return rather than attempt to do work. This is to prevent
    retries of cloud functions from retrying indefinitely.

    Args:
      error_reporter_func: A function the caller may define to perform additional error
        handling. The funciont accepts a CloudEvent and Exception object. It may perform
        additional error logic, such as writing the error to the metastore.
    """

    def decorator(
        func: Callable[[functions_framework.CloudEvent], None],
    ) -> Callable[[functions_framework.CloudEvent], None]:
        """Decorator which adds error handling and retries to a cloud function."""

        @functools.wraps(func)
        def wrapper(cloud_event: functions_framework.CloudEvent) -> None:
            logging.basicConfig(level=logging.INFO)

            # Utilities like `gcloud storage cp` create temporary files for parallel
            # uploads and then combine the chunks in one final write. We want to avoid
            # the intermediary chunk writes.
            if cloud_event.data.get("name", "").startswith(
                "gcloud/tmp/parallel_composite_uploads"
            ):
                logging.debug("Skipping tmp upload file %s", cloud_event.data["name"])
                return

            # To handle retries, check the time created and return if the event is too
            # old. GCP retries until the function returns successfully, so the pattern
            # is to report exceptions to have the function retried and return normally
            # to stop retries.
            # https://cloud.google.com/functions/docs/bestpractices/retries
            event_time = datetime.datetime.fromisoformat(
                cloud_event.data["timeCreated"]
            )
            event_age = (
                datetime.datetime.now(datetime.timezone.utc) - event_time
            ).total_seconds()
            if event_age > _MAX_RETRY_SECONDS:
                logging.error(
                    "Dropped event id: %s, source: %s, name: %s after %s seconds",
                    cloud_event["id"],
                    cloud_event["source"],
                    cloud_event.data.get("name"),
                    _MAX_RETRY_SECONDS,
                )
                return

            logging.info(
                "Received event id: %s, source: %s, bucket: %s, name: %s",
                cloud_event["id"],
                cloud_event["source"],
                cloud_event.data.get("bucket"),
                cloud_event.data.get("name"),
            )

            # Catch exceptions and report them with GCP error reporting.
            try:
                func(cloud_event)
            except Exception as exc:  # noqa
                # Report the error in GCP Error Reporter.
                error_reporting.Client().report_exception()
                # Perform any custom error handling.
                if error_reporter_func is not None:
                    error_reporter_func(cloud_event, exc)
                # Raise the exception to indicate failure and allow GCP to retry.
                raise

        return wrapper

    return decorator


def _error_to_response(f):
    """Converts exceptions to an error 500 response with a stacktrace."""

    @functools.wraps(f)
    def decorated(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception:
            return flask.make_response(traceback.format_exc(limit=10), 500)

    return decorated


@functions_framework.cloud_event
@_retry_and_report_errors()
def build_and_upload_study_area_chunk(
    cloud_event: functions_framework.CloudEvent,
) -> None:
    return _build_and_upload_study_area_chunk(
        file_name=pathlib.PurePosixPath(cloud_event.data["name"]),
        bucket_name=cloud_event.data["bucket"],
    )


@functions_framework.http
@_error_to_response
def build_and_upload_study_area_chunk_http(
    request: flask.Request,
) -> flask.Response:
    request_json = request.get_json()
    _build_and_upload_study_area_chunk(
        file_name=pathlib.PurePosixPath(request_json["name"]),
        bucket_name=request_json["bucket"],
    )
    return flask.jsonify({"message", "Upload complete."})


def _build_and_upload_study_area_chunk(
    file_name: pathlib.PurePosixPath, bucket_name: str
):
    """Creates a study area chunk, uploads to bucket, and writes to metastore.

    This function is triggered when files containing raw geo data for a study area
    are uploaded to GCS. It will load the file into study area chunks bucket and
    write a new study area to metastore.
    """
    # For AtmoML, only 3rd domain (500m) output files will be used
    if re.search(file_names.WPS_DOMAIN3_NC_REGEX, file_name.name):
        storage_client = storage.Client()
        db = firestore.Client()

        bucket = storage_client.bucket(bucket_name)
        file_blob = bucket.blob(str(file_name))

        # For WRF, we can treat each snapshot output file as its own chunk. If
        # multiple snapshots must be processed together, we can choose to group
        # the files together in a TAR before uploading.
        with file_blob.open("rb") as fd:
            study_area_chunk_bucket = storage_client.bucket(
                cloud_storage.STUDY_AREA_CHUNKS_BUCKET
            )
            # TODO: Consider cheaper alt: `copy_blob()` here since file is a straight
            # copy
            study_area_chunk_bucket.blob(str(file_name)).upload_from_file(fd)

            # File names can be in the form <study_area_name>/<file_name>
            # or <study_area_name>/<some_sub_folder>/<file_name>
            study_area_name = file_name.parts[0]

            fd.seek(0)  # Seek to the beginning so the file can be read.
            nc_bytes = fd.read()
            # create a netCDF in-memory dataset from the bytes object.
            with netCDF4.Dataset("in_memory.nc", memory=nc_bytes) as ds:
                study_area = metastore.StudyArea(
                    name=study_area_name,
                    # Grid dimension will always be 200x200 for all cities
                    col_count=200,
                    row_count=200,
                    x_ll_corner=float(ds.getncattr("corner_lons")[0]),
                    y_ll_corner=float(ds.getncattr("corner_lats")[0]),
                    # https://www2.mmm.ucar.edu/wrf/users/namelist_best_prac_wps.html#dx_dy
                    cell_size=int(ds.getncattr("DX")),
                    # https://www2.mmm.ucar.edu/wrf/users/namelist_best_prac_wps.html#map_proj
                    crs=_get_crs_from_wps(ds),
                )
                study_area.set(db)


def _get_crs_from_wps(nc_dataset: netCDF4.Dataset) -> str:
    # The var used here doesn't matter - as long as it is a var that has coordinate
    # attributes
    wps_var = wrf.getvar(wrfin=nc_dataset, varname="XLAT_M")
    cart_proj = wrf.get_cartopy(wps_var)
    source_crs = cart_proj.source_crs.to_json_dict()["conversion"]["method"]["id"]
    crs = source_crs["authority"] + ":" + str(source_crs["code"])
    return crs


@functions_framework.cloud_event
@_retry_and_report_errors()
def write_heat_scenario_config_metadata(
    cloud_event: functions_framework.CloudEvent,
) -> None:
    """Writes metadata and features for uploaded heat scenario config.

    This function is triggered when files for heat configuration are uploaded to
    GCS. It writes an entry to the metastore for the configuration.
    """
    file_name = pathlib.PurePosixPath(cloud_event.data["name"])
    # For AtmoML, use `Heat_Data_` files to trigger HeatScenarioConfig write
    # to metastore
    if re.search(file_names.HEAT_CONFIG_TXT_REGEX, file_name.name):
        storage_client = storage.Client()
        db = firestore.Client()

        config_blob = storage_client.bucket(cloud_event.data["bucket"]).blob(
            cloud_event.data["name"]
        )
        with config_blob.open("rt") as txt_fd:
            config_dict = config_readers.read_key_value_pairs(txt_fd)

        metastore.HeatScenarioConfig(
            name=config_blob.name,
            # File names should be in the form:
            # <study_area>/<parent_config_name>/<file_name>
            # Ex: NYC_Heat/Summer_Config/Heat_Data_2012.txt
            parent_config_name=file_name.parent.name,
            gcs_uri=f"gs://{config_blob.bucket.name}/{config_blob.name}",
            simulation_year=int(config_dict["simulation_year"]),
            simulation_months=config_dict["simulation_months"],
            percentile=int(config_dict["percentile"]),
        ).set(db)


@functions_framework.cloud_event
@_retry_and_report_errors()
def write_study_area_metadata(cloud_event: functions_framework.CloudEvent) -> None:
    """Writes metadata for a study area on upload.

    This function is triggered when files containing raw geo data for an entire study
    area are uploaded to GCS. It writes an entry to the metastore for the new study
    area. It also calculates min & max values needed for feature scaling of the relevant
    files.
    """
    file_name = pathlib.PurePosixPath(cloud_event.data["name"])
    if file_name.name == file_names.HEADER_JSON:
        storage_client = storage.Client()
        db = firestore.Client()

        bucket = storage_client.bucket(cloud_event.data["bucket"])
        blob = bucket.blob(str(file_name))

        with blob.open("rb") as fd:
            # We're only reading the header, so reading the first MB is plenty.
            header = elevation_readers.read_header_from_json_file(fd)
        # File names should be in the form <study_area_name>/<file_name>
        study_area_name = file_name.parent.name

        metastore.StudyArea.delete_all_chunks(db, study_area_name)
        study_area = metastore.StudyArea(
            name=study_area_name,
            state=metastore.StudyAreaState.INIT,
            col_count=header.col_count,
            row_count=header.row_count,
            x_ll_corner=header.x_ll_corner,
            y_ll_corner=header.y_ll_corner,
            cell_size=header.cell_size,
            crs=header.crs.to_string() if header.crs is not None else None,
        )
        study_area.set(db)


@functions_framework.cloud_event
@_retry_and_report_errors()
def write_flood_scenario_metadata_and_features(
    cloud_event: functions_framework.CloudEvent,
) -> None:
    """Writes metadata and features for uploaded flood scenario config.

    This function is triggered when files for rainfall configuration are uploaded to
    GCS. It writes an entry to the metastore for the configuration and writes the
    rainfall configuration as a feature vector into the GCS features bucket.
    """
    file_name = pathlib.PurePosixPath(cloud_event.data["name"])
    # Distinguish between Rainfall_Data and CityCat_Config files, the latter of which we
    # don't need to record.
    if file_name.name.startswith("Rainfall_Data_"):
        storage_client = storage.Client()
        db = firestore.Client()

        config_blob = storage_client.bucket(cloud_event.data["bucket"]).blob(
            cloud_event.data["name"]
        )
        with config_blob.open("rt") as rain_fd:
            as_vector, length = _rain_config_as_vector(rain_fd)

        vector_blob = storage_client.bucket(cloud_storage.FEATURE_CHUNKS_BUCKET).blob(
            f"rainfall/{file_name.with_suffix('.npy')}"
        )
        _write_as_npy(vector_blob, as_vector)

        metastore.FloodScenarioConfig(
            name=config_blob.name,
            gcs_uri=f"gs://{config_blob.bucket.name}/{config_blob.name}",
            as_vector_gcs_uri=f"gs://{vector_blob.bucket.name}/{vector_blob.name}",
            rainfall_duration=length,
            # File names should be in the form <parent_config_name>/<file_name>
            parent_config_name=file_name.parent.name,
        ).set(db)


def _rain_config_as_vector(rain_fd: TextIO) -> Tuple[NDArray, int]:
    """Converts the rainfall config contents into a vector describing the rainfall.

    Args:
      rain_fd: The file containing the CityCAT rainfall configuration.

    Returns:
      A tuple of (rainfall, length) where `rainfall` contains
      the rainfall pattern as a numpy array, padded to an agreed-upon
      length, and `length` represent the un-padded number entries in
      the array (i.e. the length the array would be if it was not
      padded.)

    """
    entries = config_readers.read_rainfall_amounts(rain_fd)

    # The uploader should prevent configurations over this length.
    if len(entries) > _RAINFALL_VECTOR_LENGTH:
        raise ValueError(
            f"Rainfall configuration has unexpected length {len(entries)}. "
            f"Max allowed length is {_RAINFALL_VECTOR_LENGTH}."
        )

    return numpy.pad(
        numpy.array(entries, dtype=numpy.float32),
        (0, _RAINFALL_VECTOR_LENGTH - len(entries)),
    ), len(entries)


@functions_framework.cloud_event
@_retry_and_report_errors()
def delete_flood_scenario_metadata(cloud_event: functions_framework.CloudEvent) -> None:
    """Delete metadata for uploaded flood scenario config.

    This function is triggered when files for rainfall configuration are deleted from
    GCS and likewise removes them from the metastore.
    """
    file_name = pathlib.PurePosixPath(cloud_event.data["name"])
    # Distinguish between Rainfall_Data and CityCat_Config files, the latter of which we
    # don't need to record.
    if file_name.name.startswith("Rainfall_Data_"):
        db = firestore.Client()

        metastore.FloodScenarioConfig.delete(db, cloud_event.data["name"])


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


@functions_framework.cloud_event
@_retry_and_report_errors(
    lambda cloud_event, exc: _write_chunk_metastore_error(
        cloud_event.data["name"], str(exc)
    )
)
def build_feature_matrix(cloud_event: functions_framework.CloudEvent) -> None:
    """Builds a feature matrix when a set of geo files is uploaded.

    This function is triggered when files containing geo data are uploaded to
    GCS. It produces a feature matrix for the geo data and writes that feature matrix to
    another GCS bucket for eventual use in model training and prediction.
    """
    _build_feature_matrix(
        cloud_event.data["bucket"],
        cloud_event.data["name"],
        cloud_storage.FEATURE_CHUNKS_BUCKET,
    )


@functions_framework.cloud_event
@_retry_and_report_errors(
    lambda cloud_event, exc: _write_chunk_metastore_error(
        cloud_event.data["name"], str(exc)
    )
)
def build_flood_feature_matrix(cloud_event: functions_framework.CloudEvent) -> None:
    """Builds a flood feature matrix when a chunk archive is uploaded."""
    chunk_path = cloud_event.data["name"]
    if not chunk_path.endswith(".tar"):
        logging.debug("Skipping non-flood file %s", chunk_path)
        return

    _build_flood_feature_matrix(
        cloud_event.data["bucket"],
        chunk_path,
        cloud_storage.FEATURE_CHUNKS_BUCKET,
    )


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


@functions_framework.http
@_error_to_response
def build_feature_matrix_http(request: flask.Request) -> flask.Response:
    """Builds a feature matrix when a set of geo files is uploaded.

    This function is triggered when files containing geo data are uploaded to
    GCS. It produces a feature matrix for the geo data and writes that feature matrix to
    another GCS bucket for eventual use in model training and prediction.
    """
    request_json = request.get_json()
    _build_feature_matrix(
        request_json["bucket"],
        request_json["name"],
        cloud_storage.FEATURE_CHUNKS_BUCKET,
    )
    return flask.jsonify({"message": "Feature matrix built."})


def _build_flood_feature_matrix(
    bucket_name: str, chunk_path: str, output_bucket: str
) -> None:
    """Builds a flood (CityCat) feature matrix from an uploaded chunk archive."""
    storage_client = storage.Client()
    chunk_blob = storage_client.bucket(bucket_name).blob(chunk_path)

    with chunk_blob.open("rb") as chunk:
        study_area_name, chunk_name = _parse_chunk_path(chunk_path)
        feature_file_name = pathlib.PurePosixPath(chunk_path).with_suffix(".npy")
        feature_blob = storage_client.bucket(output_bucket).blob(str(feature_file_name))
        chunk_metadata = metastore.StudyAreaSpatialChunk.get_if_exists(
            firestore.Client(), study_area_name, chunk_name
        )
        # Let's check if the chunk metadata object is present and has the state
        # different from None (which means it's either FEATURE_MATRIX_PROCESSING
        # meaning that unscaled feature matrix is stored and this CF succeeded, or
        # it's FEATURE_MATRIX_READY and the downstream rescaling CF is also done).
        # If metadata object is not present it means we're in the first execution
        # attempt. None state means that we're in the retry and either this CF
        # crashed during previous execution attempt or it finished with an error.
        if chunk_metadata is not None and chunk_metadata.state is not None:
            logging.info(
                "Flood feature matrix for chunk %s was already generated",
                chunk_path,
            )
            return

        start_time = time.time()
        logging.info("Start generating flood feature matrix for chunk %s", chunk_path)
        metastore.StudyAreaSpatialChunk(
            id_=chunk_name,
            error=firestore.DELETE_FIELD,
        ).merge(firestore.Client(), study_area_name)
        feature_matrix, metadata, header = _build_flood_feature_matrix_from_archive(
            chunk
        )

        if feature_matrix is None or header is None:
            raise ValueError(f"Empty archive found in {chunk_blob}")

        # Updating min/max elevation in the study area metadata first before storing
        # feature matrix file that will trigger rescaling post-processing.
        _update_study_area_metastore_entry(chunk_blob, metadata)
        _write_as_npy(feature_blob, feature_matrix)
        logging.info(
            "Flood feature matrix file was generated for chunk %s in %s seconds",
            chunk_path,
            time.time() - start_time,
        )
        _write_flood_chunk_metastore_entry(chunk_blob, header)


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


def _build_feature_matrix(
    bucket_name: str, chunk_path: str, output_bucket: str
) -> None:
    """Dispatches a chunk to the flood or heat builder by file type."""
    if chunk_path.endswith(".tar"):
        _build_flood_feature_matrix(bucket_name, chunk_path, output_bucket)
    elif re.search(file_names.WPS_DOMAIN3_NC_REGEX, chunk_path):
        _build_heat_feature_matrix(bucket_name, chunk_path, output_bucket)
    else:
        raise ValueError(f"Unexpected file {chunk_path}")


@functions_framework.cloud_event
@_retry_and_report_errors()
def process_citycat_outputs(cloud_event: functions_framework.CloudEvent) -> None:
    """Creates training labels as CityCAT result files are uploaded.

    CityCAT creates one .rsl file for each time step in the simulation, representing
    simulated flood heights for that timestep. For each timestep's .rsl file:
    - Rasterize the output file and break it into chunks, as we do with features. The
      labels need to be chunked identically to the features.
    - Write each chunk as a numpy matrix for the timestep into paths like
      <x_chunk_index>_<y_chunk_index>/<timestep_number>.npy

    This same function will then be triggered for each chunk upload. For each chunk:
    - Look up the expected number of timesteps in the metastore associated with this
      simulation's rainfall configuration.
    - Check to see if corresponding chunks for the same <x_chunk_index> and
      <y_chunk_index> have been uploaded for all expected timesteps.
    - If chunks for all timesteps are present, collapse them and write the resulting
      combined tensor into the labels bucket.
    - If not, return and let another chunk upload trigger the above collapsing logic.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(cloud_event.data["bucket"])
    blob = bucket.blob(cloud_event.data["name"])
    blob_path = pathlib.PurePosixPath(cloud_event.data["name"])

    if blob_path.suffix == ".rsl":
        _write_city_cat_output_chunks(blob, blob_path)
    elif blob_path.suffix == ".npy":
        _collapse_city_cat_output_chunks(
            blob, blob_path, storage_client.bucket(cloud_storage.LABEL_CHUNKS_BUCKET)
        )
    else:
        logging.error("Unexpected file %s", blob_path)


def _write_city_cat_output_chunks(
    blob: storage.Blob, blob_path: pathlib.PurePosixPath, chunk_size: int = 1000
) -> None:
    """Writes each chunk of the CityCAT result file for further processing.

    Breaks the given CityCAT .rsl result file for one timestep into chunks and writes
    them into GCS as numpy arrays. Writing these arrays re-triggers the
    `process_citycat_outputs` cloud function, which will then call
    `_collapse_city_cat_output_chunks` to collapse the chunks for each timestep.

    See the documentation for process_citycat_outputs for more details.

    Args:
      blob: The blob containing the CityCAT .rsl result file.
      blob_path: The path to the blob within its bucket.
      chunk_size: The number of cells in each side of the chunk square.
    """
    # CityCAT output file paths are of the form:
    # <study_area_name>/<config/path>/R11_C1_T<x>_y>min.rsl
    study_area_name = blob_path.parts[0]
    config_path = str(pathlib.PurePosixPath(*blob_path.parts[1:-1]))
    logging.info(
        "Processing file %s for study area %s simulation config %s",
        blob,
        study_area_name,
        config_path,
    )

    # Enter the simulation in our metastore.
    db = firestore.Client()
    metastore.Simulation(
        gcs_prefix_uri=f"gs://{blob.bucket.name}/{blob_path.parent}",
        simulation_type=metastore.SimulationType.CITY_CAT,
        study_area=metastore.StudyArea.get_ref(db, study_area_name),
        configuration=metastore.FloodScenarioConfig.get_ref(db, config_path),
    ).set(db)

    # Retrieve geography information for the study area to rasterize CityCAT results.
    header = metastore.StudyArea.get(db, study_area_name).as_header()

    chunk_prefix = pathlib.PurePosixPath("timestep_parts") / blob_path.parent
    timestep = _timestep_number_from_rsl_path(blob_path.name)
    logging.info(
        "Writing chunks of %s for timestep %s to prefix %s",
        blob,
        timestep,
        chunk_prefix,
    )

    with blob.open("rt") as citycat_result:
        depths = simulation_readers.read_city_cat_result_as_raster(
            citycat_result, header
        )
    logging.info("%s loaded, now being chunked.", blob)

    for x_index, y_index, chunk in raster_chunkers.split_raster_into_chunks(
        chunk_size, depths
    ):
        # Write chunks to paths like <x_index>_<y_index>/<timestep>.npy so that chunks
        # at every timestep for the same location can be found under the same
        # <x_index>_<y_index>/ prefix
        chunk_name = chunk_prefix / f"{x_index}_{y_index}" / f"{timestep}.npy"
        with blob.bucket.blob(str(chunk_name)).open("wb") as chunk_file:
            numpy.save(chunk_file, chunk)

    logging.info("Processed all chunks for file %s", blob)


def _collapse_city_cat_output_chunks(
    blob: storage.Blob, blob_path: pathlib.PurePosixPath, labels_bucket: storage.Bucket
) -> None:
    """Collects chunks for all timesteps at the same region and writes them as labels.

    When a chunk file is uploaded, see if other chunks for the same region have been
    uploaded for all timesteps. If so, collapse them into a single 3D tensor and write
    them into the labels bucket.

    See the documentation for process_citycat_outputs for more details.

    Args:
      blob: The blob containing the chunk .npy file.
      blob_path: The path to the blob within its bucket.
      labels_bucket: The bucket in which to write the label tensor.
    """
    # Chunks are of the form
    # timestep_parts/<study_area_name>/<config/path>/<x_index>_<y_index>/<timestep>.npy
    study_area_name = blob_path.parts[1]
    # Note that the config name be multiple 'folders' e.g. config_name/rainfall_4.txt
    config_path = str(pathlib.PurePosixPath(*blob_path.parts[2:-2]))
    x_index, y_index = map(int, blob_path.parts[-2].split("_"))
    logging.info("Processing chunk %s for config %s", blob, config_path)

    # Retrieve the number of timesteps configured for this result file's simulation.
    db = firestore.Client()
    timesteps = metastore.FloodScenarioConfig.get(db, config_path).rainfall_duration

    # See if a file is present in GCS for each timestep.
    expected_names = set(
        str(blob_path.with_stem(str(timestep))) for timestep in range(timesteps)
    )
    timestep_blobs = list(blob.bucket.list_blobs(prefix=f"{blob_path.parent}/"))
    timestep_blob_names = set(b.name for b in timestep_blobs)
    if not expected_names == timestep_blob_names:
        logging.info(
            "Chunks for some timesteps missing, halting. Expected %s got %s.",
            expected_names,
            timestep_blob_names,
        )
        return

    # Sort the blobs by their timestep number so we assemble them in order.
    timestep_blobs.sort(key=lambda b: int(pathlib.PurePosixPath(b.name).stem))
    chunk_timestep_parts = []
    for ts_blob in timestep_blobs:
        try:
            with ts_blob.open("rb") as fd:
                chunk_timestep_parts.append(numpy.load(fd))
        except exceptions.NotFound:
            # Another invocation of this function has collapsed the chunks while this
            # invocation was running.
            logging.info("Timestep chunk deletion in process, halting.")
            return

    # Stack the matrices at each timestep together and write them to the labels bucket.
    full_chunk = numpy.dstack(chunk_timestep_parts)
    label_path = pathlib.PurePosixPath(*blob_path.parts[1:-1]).with_suffix(".npy")
    with labels_bucket.blob(str(label_path)).open("wb") as fd:
        numpy.save(fd, full_chunk)
        logging.info("Wrote full chunk after processing %s.", blob)

    # Delete the chunks now that we're done with them.
    for ts_blob in timestep_blobs:
        try:
            ts_blob.delete()
        except exceptions.NotFound:
            # Another invocation of this function has collapsed the chunks while this
            # invocation was running.
            continue

    study_area = metastore.StudyArea.get(db, study_area_name)
    metastore.SimulationLabelSpatialChunk(
        gcs_uri=f"gs://{labels_bucket.name}/{label_path}",
        x_index=x_index,
        y_index=y_index,
        dataset=metastore.SimulationLabelSpatialChunk.dataset_split(
            study_area, config_path, x_index, y_index
        ),
    ).set(db, study_area_name, config_path)

    logging.info(
        "Deleted timestep chunks %s after processing %s.", timestep_blobs, blob
    )


def _timestep_number_from_rsl_path(rsl_path: str) -> str:
    """Extracts the timestep number x from paths like R11_C1_T<x>_<y>min.rsl."""
    time_step_num = re.search(r"R\d+_C\d+_T(\d+)", rsl_path)
    if time_step_num is None:
        raise ValueError(
            f"Unexpected path {rsl_path} expected path of the form R<x>_C<y>_T<n>_"
        )

    return time_step_num.group(1)


def _read_polygons_from_byte_stream(
    stream: IO[bytes],
) -> list[Tuple[geometry.Polygon, int]]:
    """Reads polygons from textual file content provided in form of a byte stream."""
    return list(polygon_readers.read_polygons_from_text_file(io.TextIOWrapper(stream)))


def _build_flood_feature_matrix_from_archive(
    archive: BinaryIO,
) -> Tuple[NDArray | None, FeatureMetadata, geo_data.ElevationHeader | None]:
    """Builds a feature matrix for the given archive.

    Args:
      archive: The file containing the archive.

    Returns:
      A tuple of (array, metadata, elevation-header) for the feature matrix, metadata
      describing the feature matrix and elevation header describing coordinate grid. In
      case there are no files related to study area in the archive optional items in the
      returning tuple will be None.
    """
    metadata: FeatureMetadata = FeatureMetadata()
    elevation: geo_data.Elevation | None = None
    boundaries: list[Tuple[geometry.Polygon, int]] | None = None
    buildings: list[Tuple[geometry.Polygon, int]] | None = None
    green_areas: list[Tuple[geometry.Polygon, int]] | None = None
    soil_classes: list[Tuple[geometry.Polygon, int]] | None = None
    files_in_tar = []

    with tarfile.TarFile(fileobj=archive) as tar:
        for member in tar:
            fd = tar.extractfile(member)
            if fd is None:
                continue

            name = pathlib.PurePosixPath(member.name).name
            files_in_tar.append(name)
            # Handle flood model input files (CityCat)
            if name == file_names.ELEVATION_TIF:
                elevation, metadata = _read_elevation_features(fd)
            elif name == file_names.BOUNDARIES_TXT:
                boundaries = _read_polygons_from_byte_stream(fd)
            elif name == file_names.BUILDINGS_TXT:
                buildings = _read_polygons_from_byte_stream(fd)
            elif name == file_names.GREEN_AREAS_TXT:
                green_areas = _read_polygons_from_byte_stream(fd)
            elif name == file_names.SOIL_CLASSES_TXT:
                soil_classes = _read_polygons_from_byte_stream(fd)
            else:
                logging.warning(f"Unexpected member name: {name}")

    flood_files_present = [
        file is not None for file in (elevation, buildings, green_areas, soil_classes)
    ]
    if any(flood_files_present):
        if (
            elevation is None
            or buildings is None
            or green_areas is None
            or soil_classes is None
        ):
            raise ValueError(
                f"Some flood simulation data missing (see tar list: {files_in_tar})"
            )
        feature_matrix = feature_raster_transformers.transform_to_feature_raster_layers(
            elevation,
            boundaries,
            buildings,
            green_areas,
            soil_classes,
            geo_data.DEFAULT_INFILTRATION_CONFIGURATION,
        )

        slope_layer = feature_matrix[:, :, 8]
        nodata = elevation.header.nodata_value
        valid_slope = slope_layer[slope_layer != nodata]
        if valid_slope.size > 0:
            metadata.slope_min = float(valid_slope.min())
            metadata.slope_max = float(valid_slope.max())
        else:
            logging.warning(
                "No valid slope values found for study area (all cells nodata); "
                "leaving slope_min/slope_max unset rather than using the nodata "
                "sentinel, which would otherwise be treated as a real bound by "
                "rescale_feature_matrix and never be displaceable by a genuine "
                "value once persisted."
            )

        return feature_matrix, metadata, elevation.header

    return None, FeatureMetadata(), None


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


def _calculate_metadata_for_elevation(elevation: geo_data.Elevation) -> FeatureMetadata:
    """Calculates metadata for the elevation data."""
    elevation_data = elevation.data

    if elevation_data is None:
        raise ValueError("Elevation file unexpectedly empty.")

    # NODATA cells should be excluded from min/max calculation
    present_data = elevation_data[elevation_data != elevation.header.nodata_value]

    if present_data.size == 0:
        # All the cells in the study area have NODATA values.
        return FeatureMetadata()

    return FeatureMetadata(
        elevation_min=float(present_data.min()),
        elevation_max=float(present_data.max()),
        chunk_size=elevation.header.row_count,
    )


def _read_elevation_features(
    fd: IO[bytes],
) -> Tuple[geo_data.Elevation, FeatureMetadata]:
    """Reads the elevation file into a matrix and returns metadata for the matrix.

    Args:
      fd: The file containing elevation data.

    Returns:
      A tuple of (Elevation, metadata) for the elevation data and metadata describing
      the feature matrix.
    """
    elevation = elevation_readers.read_from_geotiff(rasterio.io.MemoryFile(fd.read()))

    return elevation, _calculate_metadata_for_elevation(elevation)


def _update_study_area_metastore_entry(
    chunk_blob: storage.Blob, metadata: FeatureMetadata
) -> None:
    """Updates the study area metadata with new information for a given chunk.

    Updates elevation_min and elevation_max values supplied in the metadata for the
    whole study area.

    Args:
      chunk_blob: The GCS blob containing the raw chunk archive.
      metadata: Additional metadata describing the features.
    """
    db = firestore.Client()
    study_area_name, _ = _parse_chunk_path(chunk_blob.name)

    if metadata.elevation_min is not None and metadata.elevation_max is not None:
        metastore.StudyArea.update_min_max_elevation(
            db,
            study_area_name,
            min_=metadata.elevation_min,
            max_=metadata.elevation_max,
        )
    if metadata.slope_min is not None and metadata.slope_max is not None:
        metastore.StudyArea.update_min_max_slope(
            db,
            study_area_name,
            min_=metadata.slope_min,
            max_=metadata.slope_max,
        )
    if metadata.chunk_size is not None:
        metastore.StudyArea.update_chunk_info(db, study_area_name, metadata.chunk_size)

        # Let's update study area state to CHUNKS_UPLOADED if it's still INIT and all
        # the chunks are present in metadata (in any state) meaning that every chunk
        # TAR-file was placed in the chunk bucket and did trigger processing event.
        study_area_metadata = metastore.StudyArea.get(db, study_area_name)
        if (
            study_area_metadata.state == metastore.StudyAreaState.INIT
            and study_area_metadata.chunk_x_count is not None
            and study_area_metadata.chunk_y_count is not None
        ):
            expected_chunk_count = (
                study_area_metadata.chunk_x_count * study_area_metadata.chunk_y_count
            )
            current_chunk_count = len(
                metastore.StudyArea.list_all_chunk_refs(db, study_area_name)
            )
            if current_chunk_count == expected_chunk_count:
                # All the chunk metadata objects were registered, let's update the state
                # of the study area.
                metastore.StudyArea.update_state(
                    db, study_area_name, metastore.StudyAreaState.CHUNKS_UPLOADED
                )


def _write_flood_chunk_metastore_entry(
    chunk_blob: storage.Blob,
    header: geo_data.ElevationHeader,
) -> None:
    """Updates the metastore with new information for the given flood chunks.

    Writes a Firestore entry for the new feature matrix chunk.

    Args:
        chunk_blob: The GCS blob containing the raw chunk archive.
        header: The elevation header describing coordinate grid of the chunk area.
    """
    db = firestore.Client()
    study_area_name, chunk_name = _parse_chunk_path(chunk_blob.name)
    # Match against the file name only. The regex is unanchored, so passing the
    # full path would let a study area name containing "<digits>_<digits>" supply
    # the indices instead of the chunk itself.
    x_index, y_index = _parse_spatial_chunk_indices_from_name(
        pathlib.PurePosixPath(chunk_blob.name).name
    )

    metastore.StudyAreaSpatialChunk(
        id_=chunk_name,
        state=metastore.StudyAreaChunkState.FEATURE_MATRIX_PROCESSING,
        raw_path=f"gs://{chunk_blob.bucket.name}/{chunk_blob.name}",
        needs_scaling=True,
        x_index=x_index,
        y_index=y_index,
        col_count=header.col_count,
        row_count=header.row_count,
        x_ll_corner=header.x_ll_corner,
        y_ll_corner=header.y_ll_corner,
        # Remove any errors from previous failed retries which have now succeeded.
        error=firestore.DELETE_FIELD,
    ).merge(db, study_area_name)


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


def _write_chunk_metastore_error(chunk_path: str, error_message: str) -> None:
    """Updates the metastore with an error message for this chunk.

    Args:
      chunk_path: The GCS path to the chunk being processed.
      error_message: The error message to write into Firestore.
    """
    db = firestore.Client()
    file_path = pathlib.PurePosixPath(chunk_path)
    study_area_name = file_path.parts[0]
    doc_id = file_path.stem.replace(".", "_").replace(":", "_")
    metastore.StudyAreaChunk(id_=doc_id, error=error_message).merge(db, study_area_name)


def _parse_chunk_path(chunk_path: str) -> Tuple[str, str]:
    """Parses a GCS chunk path into (study_area, chunk) components.

    Args:
      chunk_path: The GCS path of a chunk to parse.

    Returns:
      Given a GCS path in the form "study_area_name/chunk_name" returns a tuple
      (study_area_name, chunk_name)
    """
    as_path = pathlib.PurePosixPath(chunk_path)
    return str(as_path.parent), as_path.stem


def _parse_spatial_chunk_indices_from_name(chunk_name: str) -> Tuple[int, int]:
    """Extract the x, y indices from a name like "chunk_x_y.tar"."""
    indices = re.search(r"(\d+)_(\d+)", chunk_name)
    if indices is None:
        raise ValueError(
            f"Unexpected chunk name {chunk_name} does not contain chunk indices"
        )
    return int(indices.group(1)), int(indices.group(2))


def _write_as_npy(blob: storage.Blob, array: NDArray) -> None:
    """Writes `array` into the GCS `blob` in the numpy binary array format.

    Args:
      blob: The blob to write the array into.
      array: The array to upload.
    """
    npy_file = io.BytesIO()
    numpy.save(npy_file, array)
    # Seek to the beginning so the file can be read.
    npy_file.flush()
    npy_file.seek(0)
    blob.upload_from_file(npy_file)


def _start_feature_rescaling_if_ready(feature_bucket: storage.Bucket, blob_path: str):
    """Creates scaled version for every chunk file in the storage bucket parent folder.

    Scaling is done in two stages: (1) appearance of each unscaled feature matrix file
    (with name following "chunk_{x}_{y}.npy" pattern) triggers the logic checking if all
    the chunks are processed and min/max elevation values aggregated in study are
    metadata are updated to their final values; if yes, empty trigger files (with names
    following "chunk_{x}_{y}.scale_trigger" pattern) are placed next to each unscaled
    feature matrix file in order to trigger separate processing for each matrix; then
    (2) each trigger file triggers parallel run of this cloud function scaling unscaled
    feature matrix file producing scaled version of it as output (file name follows
    "scaled_chunk_{x}_{y}.npy" pattern); metadata of a given chunk is updated to reflect
    that scaling is done; unscaled input file together with trigger file for this matrix
    are deleted at the end.

    Args:
        feature_bucket: Storage bucket with feature matrices.
        blob_path: Path pointing to one of unscaled feature matrix files with the name
            following the "chunk_*.npy" pattern or one of trigger files with the name
            following the "chunk_*.scale_trigger" pattern. Other files are ignored.
    """
    study_area_name, chunk_name = _parse_chunk_path(blob_path)
    db = firestore.Client()

    if not chunk_name.startswith("chunk_"):
        logging.info("Skipping %s, which is not an unscaled feature matrix", blob_path)
        return

    study_area = metastore.StudyArea.get(db, study_area_name)

    if blob_path.endswith(".npy"):
        if study_area.chunk_x_count is None or study_area.chunk_y_count is None:
            raise ValueError(
                f"[Feature Rescaler] Study area {study_area_name} has no chunk counts"
                + " in metadata"
            )

        prefix_blobs = feature_bucket.list_blobs(prefix=f"{study_area_name}/chunk_")
        # Let's exclude any files that don't have ".npy" file extension
        feature_blobs = [blob for blob in prefix_blobs if blob.name.endswith(".npy")]

        chunk_count = study_area.chunk_x_count * study_area.chunk_y_count
        if chunk_count != len(feature_blobs):
            # Not all the chunks are ready, so we're just waiting for the rest of the
            # files to be written.
            logging.info(
                "[Feature Rescaler] Study area %s has chunk counts in metadata (%s)"
                + " different from number of stored feature chunks (%s)",
                study_area_name,
                chunk_count,
                len(feature_blobs),
            )
            return

        # We're at the final step when all unscaled feature matrices are ready
        logging.info(
            "[Feature Rescaler] Study area %s has all the unscaled feature matrices"
            + "stored, count matches to what's registered in metadata (%s)",
            study_area_name,
            chunk_count,
        )
        metastore.StudyArea.update_state(
            db, study_area_name, metastore.StudyAreaState.FEATURE_MATRICES_CREATED
        )
        for feature_blob in feature_blobs:
            _, blob_chunk_name = _parse_chunk_path(feature_blob.name)
            trigger_blob = feature_bucket.blob(
                f"{study_area_name}/{blob_chunk_name}{_FEATURE_SCALING_TRIGGER_SUFFIX}"
            )
            trigger_blob.upload_from_string(data="")  # Empty file content
    elif blob_path.endswith(_FEATURE_SCALING_TRIGGER_SUFFIX):
        chunk_metadata = metastore.StudyAreaSpatialChunk.get_if_exists(
            db, study_area_name, chunk_name
        )
        if chunk_metadata is None:
            logging.info(
                f"Chunk metadata is not registered for {study_area_name}/{chunk_name}"
            )
            return

        if not chunk_metadata.needs_scaling:
            logging.info(f"Chunk {study_area_name}/{chunk_name} doesn't need scaling")
            return

        logging.info(
            "[Feature Rescaler] Rescaling feature matrix %s/%s.npy",
            study_area_name,
            chunk_name,
        )
        feature_blob = feature_bucket.blob(f"{study_area_name}/{chunk_name}.npy")
        try:
            with feature_blob.open("rb") as feature_matrix_input_fd:
                feature_matrix = numpy.load(feature_matrix_input_fd)
        except exceptions.NotFound:
            # Another invocation of this function has rescaled the feature matrix while
            # this invocation was running.
            logging.info("Feature matrix scaling in process, halting.")
            return

        feature_raster_transformers.rescale_feature_matrix(feature_matrix, study_area)
        output_blob = feature_bucket.blob(f"{study_area_name}/scaled_{chunk_name}.npy")
        with output_blob.open("wb") as feature_matrix_output_fd:
            numpy.save(feature_matrix_output_fd, feature_matrix)

        scaled_feature_matrix_path = (
            f"gs://{feature_bucket.name}/{study_area_name}/scaled_{chunk_name}.npy"
        )
        metastore.StudyAreaSpatialChunk.update_scaling_done(
            db, study_area_name, chunk_name, scaled_feature_matrix_path
        )

        # Deleting unscaled matrix, skip Not Found errors for idempotency.
        try:
            feature_blob.delete()
        except exceptions.NotFound:
            pass

        # Deleting trigger file, skip Not Found errors for idempotency.
        try:
            feature_bucket.blob(blob_path).delete()
        except exceptions.NotFound:
            pass

        # Let's check if all the chunks are rescaled and the whole study area can be
        # switched to finalized state.
        ref_list = metastore.StudyArea.list_all_chunk_refs(db, study_area_name)
        chunk_list = [metastore.StudyAreaSpatialChunk.from_ref(ref) for ref in ref_list]
        if all(
            chunk.state == metastore.StudyAreaChunkState.FEATURE_MATRIX_READY
            for chunk in chunk_list
        ):
            metastore.StudyArea.update_state(
                db, study_area_name, metastore.StudyAreaState.RESCALING_DONE
            )


@functions_framework.cloud_event
@_retry_and_report_errors()
def rescale_feature_matrices(cloud_event: functions_framework.CloudEvent) -> None:
    """Starts rescaling process once all unscaled feature matrix files are uploaded.

    Output files with scaled feature matrices will be placed to the same folder where
    unscaled files (following "chunk_*" name pattern) are stored but will get "scaled_"
    prefix. Rescaling only starts when metadata of study area (corresponding to the
    parent folder name) contains chunks counters giving exactly the same number of
    chunks as are currently stored in the bucket folder. This should happen when the
    last feature chunk is stored.

    Args:
        cloud_event: Cloud event pointing to one of unscaled feature matrix files with
            name following the "chunk_*" pattern. Other files are ignored.
    """
    if re.search(file_names.WPS_DOMAIN3_NC_REGEX, cloud_event.data["name"]):
        logging.info("Skipping WRF file  %s", cloud_event.data["name"])
        return

    if cloud_event.data["name"].startswith("rainfall/"):
        logging.info("Skipping rainfall vector %s", cloud_event.data["name"])
        return

    _start_feature_rescaling_if_ready(
        storage.Client().bucket(cloud_event.data["bucket"]), cloud_event.data["name"]
    )


# ---------------------------------------------------------------------------
# H3 pipeline
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(__file__))

try:
    from cli_unified_pipeline import (  # noqa: E402
        process_admin_levels,
        process_h3_tilesets,
        build_h3_breadcrumb_index,
    )
    from cli_add_admin_breadcrumbs import (  # noqa: E402
        add_parent_fields_point_within,
    )
    from cli_preprocess_tiff_to_h3csv import preprocess_city  # noqa: E402
except ImportError:
    pass

_MAX_RETRY_SECONDS = 60 * 60

# These are constant across all cities — no config file needed.
SCENARIO_MAPPING = {
    "Rainfall_Data_1.txt": "flood_depth_1y",
    "Rainfall_Data_2.txt": "flood_depth_5y",
    "Rainfall_Data_3.txt": "flood_depth_10y",
    "Rainfall_Data_4.txt": "flood_depth_25y",
    "Rainfall_Data_5.txt": "flood_depth_50y",
    "Rainfall_Data_6.txt": "flood_depth_100y",
    "Rainfall_Data_7.txt": "flood_depth_200y",
    "Rainfall_Data_8.txt": "flood_depth_500y",
    "Rainfall_Data_9.txt": "flood_depth_1000y",
}

VALUE_COLS = list(SCENARIO_MAPPING.values())

H3_HEX_RESOLUTIONS = [6, 7, 8, 9, 10, 11, 12]

ADMIN_BOUNDARY_FILES = {
    "level_6": "counties.parquet",
    "level_8": "places.parquet",
    "level_10": "tracts.parquet",
}

FILTER_THRESHOLD = 0.001

EXPECTED_SCENARIO_COUNT = 9

OUTPUT_BUCKET = "climateiq-h3-tiles"
MAX_COMBINED_BYTES = 20 * 1024 * 1024 * 1024

LEVEL_PATH_MAP = {
    "6": "counties",
    "8": "places",
    "10": "tracts",
    "106": "res06",
    "107": "res07",
    "108": "res08",
    "109": "res09",
    "110": "res10",
    "111": "res11",
    "112": "res12",
}

WATER_POLYGONS_PREFIX = "water_polygons"
JRC_TILE_URL = (
    "https://storage.googleapis.com/global-surface-water"
    "/downloads2021/occurrence/occurrence_{tile}v1_4_2021.tif"
)
JRC_OCCURRENCE_THRESHOLD = 80


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------


def _download_from_gcs(bucket_name, source_blob, dest_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob)
    blob.download_to_filename(dest_path)
    logging.info("Downloaded gs://%s/%s", bucket_name, source_blob)


def _upload_to_gcs(bucket_name, source_path, dest_blob):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_blob)
    blob.upload_from_filename(source_path)
    logging.info("Uploaded to gs://%s/%s", bucket_name, dest_blob)


def _download_directory_from_gcs(bucket_name, prefix, dest_dir):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    os.makedirs(dest_dir, exist_ok=True)
    for blob in blobs:
        if not blob.name.endswith("/"):
            relative_path = blob.name[len(prefix) :].lstrip("/")
            dest_path = os.path.join(dest_dir, relative_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            blob.download_to_filename(dest_path)
            logging.info("Downloaded %s", blob.name)


# ---------------------------------------------------------------------------
# Derive city info from the predictions bucket path
# ---------------------------------------------------------------------------


def _parse_city_folder(city_folder):
    """Derive city name and config prefix from folder name.

    Pattern: '{CityName}_Predictions' or '{CityName}_Prediction'
    e.g. 'NYC_Predictions' -> city_name='NYC', config_prefix='NYC'
         'Chicago_Predictions' -> city_name='Chicago', config_prefix='Chicago'
    """
    m = re.match(r"^(.+?)_Predictions?(?:_.*)?$", city_folder)
    if m:
        city_name = m.group(1)
    else:
        city_name = city_folder
    config_prefix = city_name
    mosaic_filename = f"{city_folder}_mosaic_peak_wgs84.tif"
    return city_name, config_prefix, mosaic_filename


def _build_city_config(city_name, city_folder, predictions_bucket):
    """Build a city config dict from just the folder name — no YAML needed."""
    config_prefix = city_name
    mosaic_filename = f"{city_folder}_mosaic_peak_wgs84.tif"
    return {
        "name": city_name,
        "predictions": {
            "bucket": predictions_bucket,
            "city_folder": city_folder,
            "config_prefix": config_prefix,
            "mosaic_filename": mosaic_filename,
        },
    }


def _derive_admin_fields(city_config, tiff_bbox, admin_boundaries):
    """Derive state_fips, county_geoids, and place_name from the TIF bbox.

    Uses spatial intersection against the Census parquet files so no
    per-city config is needed.
    """
    import geopandas as gpd
    from shapely.geometry import box

    if tiff_bbox is None:
        logging.warning("No tiff_bbox — skipping admin field derivation")
        return city_config

    bbox_geom = box(*tiff_bbox)

    counties_path = admin_boundaries.get("level_6")
    if counties_path and os.path.exists(counties_path):
        counties = gpd.read_parquet(counties_path)
        counties = counties.to_crs("EPSG:4326")
        hits = counties[counties.intersects(bbox_geom)]
        if not hits.empty:
            state_fips = hits["STATEFP"].mode().iloc[0]
            county_geoids = sorted(hits["GEOID"].unique().tolist())
            city_config["state_fips"] = state_fips
            city_config["county_geoids"] = county_geoids
            logging.info(
                "Derived state_fips=%s, county_geoids=%s", state_fips, county_geoids
            )

    places_path = admin_boundaries.get("level_8")
    if places_path and os.path.exists(places_path):
        places = gpd.read_parquet(places_path)
        places = places.to_crs("EPSG:4326")
        hits = places[places.intersects(bbox_geom)]
        if not hits.empty:
            areas = hits.geometry.area
            largest = hits.loc[areas.idxmax()]
            city_config["place_name"] = largest["NAME"]
            if "STATEFP" in largest.index:
                city_config["state_fips"] = largest["STATEFP"]
                logging.info(
                    "Derived state_fips=%s from place %s",
                    largest["STATEFP"],
                    largest["NAME"],
                )
            logging.info("Derived place_name=%s", city_config["place_name"])

    return city_config


def _build_pipeline_config(work_dir, pipeline_bucket, cities):
    """Build the full pipeline config dict without any YAML file."""
    data_dir = os.path.join(work_dir, "data")
    _download_directory_from_gcs(pipeline_bucket, "admin_boundaries/", data_dir)

    return {
        "output_dir": os.path.join(work_dir, "output"),
        "h3_col": "cell_code",
        "value_cols": VALUE_COLS,
        "filter_threshold": FILTER_THRESHOLD,
        "h3_hex_resolutions": H3_HEX_RESOLUTIONS,
        "scenario_mapping": SCENARIO_MAPPING,
        "admin_boundaries": {
            "level_6": os.path.join(data_dir, ADMIN_BOUNDARY_FILES["level_6"]),
            "level_8": os.path.join(data_dir, ADMIN_BOUNDARY_FILES["level_8"]),
            "level_10": os.path.join(data_dir, ADMIN_BOUNDARY_FILES["level_10"]),
        },
        "cities": cities,
    }


# ---------------------------------------------------------------------------
# Breadcrumbs, title-casing, and post-processing
# ---------------------------------------------------------------------------

_SMALL_WORDS = {
    "of",
    "the",
    "and",
    "in",
    "on",
    "at",
    "to",
    "for",
    "de",
    "del",
    "la",
    "las",
    "los",
}
_NAME_CORRECTIONS = {
    "Dekalb": "DeKalb",
    "Mcdonald": "McDonald",
    "Mchenry": "McHenry",
    "Mclean": "McLean",
    "Mckinley": "McKinley",
    "Desoto": "DeSoto",
    "Dupage": "DuPage",
    "Lasalle": "LaSalle",
}


def _smart_title(s):
    if not isinstance(s, str):
        return s
    words = s.title().split()
    for i in range(1, len(words)):
        if words[i].lower() in _SMALL_WORDS:
            words[i] = words[i].lower()
    result = " ".join(words)
    for wrong, right in _NAME_CORRECTIONS.items():
        result = result.replace(wrong, right)
    return result


def _fix_bbox(v):
    if isinstance(v, str) and v.startswith("["):
        try:
            parts = v.strip("[]").replace(",", " ").split()
            return [float(x) for x in parts if x]
        except ValueError:
            return v
    if hasattr(v, "__iter__") and not isinstance(v, str):
        return list(v)
    return v


def _title_case_gdf(gdf):
    if "name" in gdf.columns:
        gdf["name"] = gdf["name"].apply(_smart_title)
    for col in gdf.columns:
        if col.endswith("_localname"):
            gdf[col] = gdf[col].apply(_smart_title)
    if "place_name" in gdf.columns:
        gdf["place_name"] = gdf["place_name"].apply(_smart_title)
    if "county_name" in gdf.columns:
        gdf["county_name"] = gdf["county_name"].apply(_smart_title)
    if "bbox" in gdf.columns:
        gdf["bbox"] = gdf["bbox"].apply(_fix_bbox)
    return gdf


STATE_FIPS_TO_INFO = {
    "04": {"name": "Arizona", "osm_id": "162018", "abbr": "AZ"},
    "06": {"name": "California", "osm_id": "165475", "abbr": "CA"},
    "10": {"name": "Delaware", "osm_id": "162110", "abbr": "DE"},
    "12": {"name": "Florida", "osm_id": "162050", "abbr": "FL"},
    "13": {"name": "Georgia", "osm_id": "161957", "abbr": "GA"},
    "17": {"name": "Illinois", "osm_id": "122586", "abbr": "IL"},
    "22": {"name": "Louisiana", "osm_id": "224922", "abbr": "LA"},
    "24": {"name": "Maryland", "osm_id": "162112", "abbr": "MD"},
    "34": {"name": "New Jersey", "osm_id": "224951", "abbr": "NJ"},
    "36": {"name": "New York", "osm_id": "61320", "abbr": "NY"},
    "42": {"name": "Pennsylvania", "osm_id": "162109", "abbr": "PA"},
    "48": {"name": "Texas", "osm_id": "114690", "abbr": "TX"},
}


def _city_output_folder(city_name, state_fips=None):
    """Build output folder name: USA_{STATE}_{CityPascalCase}.

    Falls back to USA_{CityName} if state_fips is unknown.
    """
    pascal = city_name.replace(" ", "")
    if state_fips:
        abbr = STATE_FIPS_TO_INFO.get(str(state_fips), {}).get("abbr")
        if abbr:
            return f"USA_{abbr}_{pascal}"
    return f"USA_{pascal}"

USA_BOUNDARY_ID = "148838"
USA_NAME = "United States of America"


def _build_city_breadcrumbs(city_config):
    """Build breadcrumb dict for a city from its config.

    Returns dict with admin_level_N_parent_boundary_id/localname keys.
    """
    crumbs = {
        "admin_level_2_parent_boundary_id": USA_BOUNDARY_ID,
        "admin_level_2_parent_localname": USA_NAME,
    }
    state_fips = city_config.get("state_fips")
    if state_fips:
        info = STATE_FIPS_TO_INFO.get(str(state_fips), {})
        crumbs["admin_level_4_parent_boundary_id"] = info.get("osm_id", str(state_fips))
        crumbs["admin_level_4_parent_localname"] = info.get("name", "")
    place_name = city_config.get("place_name")
    if place_name:
        crumbs["admin_level_8_parent_localname"] = place_name
    return crumbs


def _add_breadcrumbs_to_outputs(output_dir, config_dict=None):
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from shapely.ops import unary_union

    logging.info("Adding breadcrumbs to output files...")

    level_6_path = os.path.join(output_dir, "admin_level_6_all_cities.geojson")
    level_8_path = os.path.join(output_dir, "admin_level_8_all_cities.geojson")
    level_10_path = os.path.join(output_dir, "admin_level_10_all_cities.geojson")

    files_exist = {
        6: os.path.exists(level_6_path),
        8: os.path.exists(level_8_path),
        10: os.path.exists(level_10_path),
    }

    if not any(files_exist.values()):
        logging.info("No admin level files found for breadcrumbs")
        return

    gdfs = {}
    if files_exist[6]:
        gdfs[6] = gpd.read_file(level_6_path)
    if files_exist[8]:
        gdfs[8] = gpd.read_file(level_8_path)
    if files_exist[10]:
        gdfs[10] = gpd.read_file(level_10_path)

    if 6 in gdfs:
        gdfs[6]["is_city"] = False
    if 8 in gdfs:
        place_name = None
        if config_dict and "cities" in config_dict:
            for c in config_dict["cities"]:
                pn = c.get("place_name")
                if pn:
                    place_name = pn.lower()
                    break
        if place_name and "name" in gdfs[8].columns:
            gdfs[8]["is_city"] = gdfs[8]["name"].str.lower().str.strip() == place_name.lower().strip()
        else:
            gdfs[8]["is_city"] = True
    if 10 in gdfs:
        gdfs[10]["is_city"] = False

    for lvl, gdf in gdfs.items():
        _title_case_gdf(gdf)

    for lvl, gdf in gdfs.items():
        gdf["admin_level_2_parent_boundary_id"] = USA_BOUNDARY_ID
        gdf["admin_level_2_parent_localname"] = USA_NAME

    for lvl, gdf in gdfs.items():
        if "STATEFP" in gdf.columns:
            gdf["admin_level_4_parent_boundary_id"] = gdf["STATEFP"].map(
                lambda fp: STATE_FIPS_TO_INFO.get(str(fp), {}).get("osm_id", str(fp))
            )
            gdf["admin_level_4_parent_localname"] = gdf["STATEFP"].map(
                lambda fp: STATE_FIPS_TO_INFO.get(str(fp), {}).get("name", "")
            )

    if 8 in gdfs and 6 in gdfs:
        gdfs[8] = add_parent_fields_point_within(gdfs[8], gdfs[6], parent_level=6)

    if 10 in gdfs and 8 in gdfs:
        gdfs[10] = add_parent_fields_point_within(gdfs[10], gdfs[8], parent_level=8)
        if "admin_level_8_parent_localname" in gdfs[10].columns:
            col8 = gdfs[10]["admin_level_8_parent_localname"]
            gdfs[10]["admin_level_8_parent_localname"] = col8.where(
                ~(col8.isna() | (col8.astype(str) == "<NA>")), ""
            )
            gdfs[10]["place_name"] = gdfs[10]["admin_level_8_parent_localname"]
        if "admin_level_8_parent_boundary_id" in gdfs[10].columns:
            col8id = gdfs[10]["admin_level_8_parent_boundary_id"]
            gdfs[10]["admin_level_8_parent_boundary_id"] = col8id.where(
                ~(col8id.isna() | (col8id.astype(str) == "<NA>")), ""
            )

    if 10 in gdfs and 6 in gdfs:
        gdfs[10] = add_parent_fields_point_within(gdfs[10], gdfs[6], parent_level=6)
        # Fallback: fill <NA> county breadcrumbs from FIPS codes
        gdf10 = gdfs[10]
        county_col = "admin_level_6_parent_localname"
        county_id_col = "admin_level_6_parent_boundary_id"
        if county_col in gdf10.columns and "STATEFP" in gdf10.columns and "COUNTYFP" in gdf10.columns:
            missing = gdf10[county_col].isna() | (gdf10[county_col].astype(str) == "<NA>")
            if missing.any() and 6 in gdfs:
                county_lookup = dict(
                    zip(gdfs[6]["GEOID"].astype(str), gdfs[6]["name"].astype(str))
                ) if "GEOID" in gdfs[6].columns and "name" in gdfs[6].columns else {}
                for idx in gdf10.index[missing]:
                    geoid = str(gdf10.at[idx, "STATEFP"]) + str(gdf10.at[idx, "COUNTYFP"])
                    if geoid in county_lookup:
                        gdf10.at[idx, county_id_col] = geoid
                        gdf10.at[idx, county_col] = county_lookup[geoid]
        if county_col in gdf10.columns:
            gdf10["county_name"] = gdf10[county_col]

    for lvl, gdf in gdfs.items():
        _title_case_gdf(gdf)

    if 8 in gdfs:
        gdfs[8].to_file(level_8_path, driver="GeoJSON")
    if 10 in gdfs:
        gdfs[10].to_file(level_10_path, driver="GeoJSON")
    if 6 in gdfs:
        gdfs[6].to_file(level_6_path, driver="GeoJSON")

    if 6 in gdfs:
        nyc_geoids = {"36061", "36047", "36081", "36005", "36085"}
        gdf6 = gdfs[6]
        id_col = "boundary_id" if "boundary_id" in gdf6.columns else "id"
        nyc_mask = gdf6[id_col].astype(str).isin(nyc_geoids)
        nyc_counties = gdf6[nyc_mask]

        if len(nyc_counties) > 0:
            merged_geom = unary_union(nyc_counties.geometry)
            rec = {
                "admin_level": 4,
                "id": "61320",
                "boundary_id": "61320",
                "name": "New York",
                "is_city": True,
                "admin_level_2_parent_boundary_id": USA_BOUNDARY_ID,
                "admin_level_2_parent_localname": USA_NAME,
                "admin_level_4_parent_boundary_id": "61320",
                "admin_level_4_parent_localname": "New York",
                "city": "NYC",
                "STATEFP": "36",
            }
            numeric_cols = [
                c
                for c in nyc_counties.columns
                if c
                not in (
                    "geometry",
                    id_col,
                    "name",
                    "city",
                    "STATEFP",
                    "COUNTYFP",
                    "GEOID",
                    "admin_level",
                    "is_city",
                )
                and nyc_counties[c].dtype in ("float64", "int64", "float32")
            ]
            for col in numeric_cols:
                vals = nyc_counties[col].dropna().values
                rec[col] = round(float(np.mean(vals)), 6) if len(vals) > 0 else 0.0

            level_4_gdf = gpd.GeoDataFrame(
                [rec], geometry=[merged_geom], crs="EPSG:4326"
            )
            level_4_path = os.path.join(output_dir, "admin_level_4_all_cities.geojson")
            level_4_gdf.to_file(level_4_path, driver="GeoJSON")
            logging.info("Created admin_level_4 (NYC aggregation)")

    if 10 in gdfs:
        city_breadcrumbs = {}
        city_best_score = {}
        for _, row in gdfs[10].iterrows():
            city = row.get("city")
            if not city:
                continue
            score = 0
            candidate = {}
            for lvl in (2, 4, 6, 8):
                for suffix in ("_parent_boundary_id", "_parent_localname"):
                    key = f"admin_level_{lvl}{suffix}"
                    val = row.get(key)
                    if pd.notna(val) and str(val) not in (
                        "<NA>",
                        "<Na>",
                        "None",
                        "nan",
                    ):
                        candidate[key] = val
                        score += 1
            if score > city_best_score.get(city, -1):
                city_breadcrumbs[city] = candidate
                city_best_score[city] = score

        # H3 hex files get breadcrumbs injected at write time via
        # h3_breadcrumbs passed to process_h3_tilesets — no need to
        # re-read multi-GB files here.

    logging.info("Breadcrumbs added successfully")


# ---------------------------------------------------------------------------
# Upload per-city + rebuild all_cities from per-city parts
# ---------------------------------------------------------------------------


def _merge_and_upload_outputs(bucket_name, output_prefix, output_dir, cities_in_run):
    """Upload per-city files, then compose all_cities via GCS server-side compose.

    For each admin level:
    1. Upload per-city GeoJSON.
    2. Create a features-only blob (raw features, no FeatureCollection wrapper).
    3. Bootstrap features-only blobs for other cities from their per-city files.
    4. GCS compose: header + city1 + comma + city2 + … + footer → all_cities.

    The compose is server-side — no data streams through the function, so it
    works for files of any size in seconds.
    """
    import glob as _glob
    import re as _re

    output_files = _glob.glob(os.path.join(output_dir, "*.geojson"))
    if not output_files:
        logging.info("No output files to upload")
        return

    def _level_key(path):
        m = _re.search(r"admin_level_(\d+)_", os.path.basename(path))
        return int(m.group(1)) if m else 9999

    output_files = sorted(output_files, key=_level_key)

    client = storage.Client()
    bucket_obj = client.bucket(bucket_name)
    city_name = next(iter(cities_in_run))
    FEATURES_RAW_PREFIX = f"{output_prefix}features_raw"
    CHUNK = 256 * 1024 * 1024

    for local_path in output_files:
        fname = os.path.basename(local_path)
        per_city_dest = f"{output_prefix}by_city/{city_name}/{fname}"
        _upload_to_gcs(bucket_name, local_path, per_city_dest)
        logging.info("Uploaded per-city: %s", per_city_dest)

    def _strip_fc_wrapper(data):
        """Return the content between [ and ] in a FeatureCollection."""
        start = data.find(b"[")
        end = data.rfind(b"]")
        if start == -1 or end == -1 or end <= start:
            return b""
        return data[start + 1 : end]

    def _ensure_features_raw(city, level_str, bucket_o):
        """Create features_raw blob from per-city file if it doesn't exist."""
        raw_path = f"{FEATURES_RAW_PREFIX}/{city}/level_{level_str}.bin"
        raw_blob = bucket_o.blob(raw_path)
        if raw_blob.exists():
            return raw_blob

        pc_path = (
            f"{output_prefix}by_city/{city}/admin_level_{level_str}_all_cities.geojson"
        )
        pc_blob = bucket_o.blob(pc_path)
        if not pc_blob.exists():
            return None

        pc_blob.reload()
        pc_size = pc_blob.size or 0

        if pc_size < 500 * 1024 * 1024:
            content = pc_blob.download_as_bytes()
            features_raw = _strip_fc_wrapper(content)
        else:
            raw_blob.chunk_size = CHUNK
            pc_blob.chunk_size = CHUNK
            with pc_blob.open("rb") as inp, raw_blob.open("wb") as out:
                header_done = False
                leftover = b""
                while True:
                    block = inp.read(CHUNK)
                    if not block:
                        if leftover:
                            end = leftover.rfind(b"]")
                            if end > 0:
                                out.write(leftover[:end])
                        break
                    if not header_done:
                        combined = leftover + block
                        bracket = combined.find(b"[")
                        if bracket == -1:
                            leftover = combined[-100:]
                            continue
                        header_done = True
                        leftover = combined[bracket + 1 :]
                        continue
                    if leftover:
                        out.write(leftover)
                    leftover = block
            return raw_blob

        if features_raw:
            raw_blob.upload_from_string(features_raw)
            return raw_blob
        return None

    for local_path in output_files:
        fname = os.path.basename(local_path)
        level_match = _re.search(r"admin_level_(\d+)_", fname)
        if not level_match:
            continue
        level_str = level_match.group(1)
        all_cities_path = f"{output_prefix}{fname}"

        try:
            with open(local_path, "rb") as f:
                content = f.read()
            features_raw = _strip_fc_wrapper(content)

            raw_path = f"{FEATURES_RAW_PREFIX}/{city_name}/level_{level_str}.bin"
            raw_blob = bucket_obj.blob(raw_path)
            raw_blob.upload_from_string(features_raw)
            logging.info(
                "Uploaded features_raw for %s level %s (%d bytes)",
                city_name,
                level_str,
                len(features_raw),
            )

            by_city_prefix = f"{output_prefix}by_city/"
            city_dirs = set()
            blobs_iter = client.list_blobs(
                bucket_name, prefix=by_city_prefix, delimiter="/"
            )
            for _ in blobs_iter:
                pass
            for prefix_str in blobs_iter.prefixes:
                c = prefix_str.replace(by_city_prefix, "").rstrip("/")
                if c:
                    city_dirs.add(c)

            logging.info(
                "Found %d cities for level %s compose: %s",
                len(city_dirs),
                level_str,
                sorted(city_dirs),
            )

            city_raw_blobs = []
            for c in sorted(city_dirs):
                if c == city_name:
                    city_raw_blobs.append((c, raw_blob))
                    continue
                rb = _ensure_features_raw(c, level_str, bucket_obj)
                if rb is not None:
                    city_raw_blobs.append((c, rb))

            if not city_raw_blobs:
                logging.warning("No features_raw blobs found for level %s", level_str)
                continue

            header_blob = bucket_obj.blob(f"{FEATURES_RAW_PREFIX}/_header.bin")
            header_blob.upload_from_string(b'{"type":"FeatureCollection","features":[')

            comma_blob = bucket_obj.blob(f"{FEATURES_RAW_PREFIX}/_comma.bin")
            comma_blob.upload_from_string(b",")

            footer_blob = bucket_obj.blob(f"{FEATURES_RAW_PREFIX}/_footer.bin")
            footer_blob.upload_from_string(b"]}")

            sources = [header_blob]
            for i, (c, rb) in enumerate(city_raw_blobs):
                if i > 0:
                    sources.append(comma_blob)
                sources.append(rb)
            sources.append(footer_blob)

            logging.info(
                "Composing %s from %d sources (%d cities)",
                fname,
                len(sources),
                len(city_raw_blobs),
            )

            dest_blob = bucket_obj.blob(all_cities_path)
            dest_blob.compose(sources)

            logging.info("  Composed %s successfully", fname)

        except Exception as e:
            logging.error("Failed to compose %s: %s", fname, e)


# ---------------------------------------------------------------------------
# Publish clean-named output to the public tile bucket
# ---------------------------------------------------------------------------


def _level_to_path(level_str):
    path = LEVEL_PATH_MAP.get(level_str)
    if path:
        return path
    lvl = int(level_str)
    if lvl > 100:
        return f"res{lvl - 100:02d}"
    return f"level{lvl}"


def _gcs_compose(dest_blob, sources):
    """Compose with cascading to handle the 32-source GCS limit."""
    MAX_COMPOSE = 32
    while len(sources) > MAX_COMPOSE:
        batches = [
            sources[i : i + MAX_COMPOSE] for i in range(0, len(sources), MAX_COMPOSE)
        ]
        intermediate = []
        for bi, batch in enumerate(batches):
            if len(batch) == 1:
                intermediate.append(batch[0])
                continue
            tmp = dest_blob.bucket.blob(f"{dest_blob.name}__tmp{bi}")
            tmp.compose(batch)
            intermediate.append(tmp)
        sources = intermediate
    dest_blob.compose(sources)


def _publish_to_output_bucket(
    pipeline_bucket_name, output_dir, city_name, output_bucket=None,
    state_fips=None,
):
    """Copy per-city and combined files to the output bucket with clean names.

    Per-city:  us/{city}/admin/counties.geojson
    Combined:  us/combined/hex/res12.geojson  (split if >20 GB)
    """
    import glob as _glob
    import re as _re

    output_bucket = output_bucket or os.environ.get("OUTPUT_BUCKET", OUTPUT_BUCKET)

    output_files = _glob.glob(os.path.join(output_dir, "*.geojson"))
    if not output_files:
        return

    client = storage.Client()
    out_bucket = client.bucket(output_bucket)
    pipe_bucket = client.bucket(pipeline_bucket_name)
    city_folder = _city_output_folder(city_name, state_fips)

    for local_path in output_files:
        fname = os.path.basename(local_path)
        m = _re.search(r"admin_level_(\d+)_", fname)
        if not m:
            continue
        level_str = m.group(1)
        rel_path = _level_to_path(level_str)

        dest_path = f"{city_folder}/{rel_path}.geojson"
        blob = out_bucket.blob(dest_path)
        blob.upload_from_filename(local_path)
        logging.info("Published per-city: gs://%s/%s", output_bucket, dest_path)

    pipe_prefix = "output/features_raw"
    by_city_prefix = "output/by_city/"

    city_dirs = set()
    blobs_iter = client.list_blobs(
        pipeline_bucket_name, prefix=by_city_prefix, delimiter="/"
    )
    for _ in blobs_iter:
        pass
    for prefix_str in blobs_iter.prefixes:
        c = prefix_str.replace(by_city_prefix, "").rstrip("/")
        if c:
            city_dirs.add(c)

    if not city_dirs:
        return

    header_blob = out_bucket.blob("_compose/header.bin")
    header_blob.upload_from_string(b'{"type":"FeatureCollection","features":[')
    comma_blob = out_bucket.blob("_compose/comma.bin")
    comma_blob.upload_from_string(b",")
    footer_blob = out_bucket.blob("_compose/footer.bin")
    footer_blob.upload_from_string(b"]}")

    for local_path in output_files:
        fname = os.path.basename(local_path)
        m = _re.search(r"admin_level_(\d+)_", fname)
        if not m:
            continue
        level_str = m.group(1)
        rel_path = _level_to_path(level_str)

        try:
            city_raw_entries = []
            for c in sorted(city_dirs):
                raw_src = f"{pipe_prefix}/{c}/level_{level_str}.bin"
                src_blob = pipe_bucket.blob(raw_src)
                if not src_blob.exists():
                    continue
                src_blob.reload()
                size = src_blob.size or 0
                dest_raw = f"_compose/{c}/level_{level_str}.bin"
                pipe_bucket.copy_blob(src_blob, out_bucket, dest_raw)
                city_raw_entries.append((c, out_bucket.blob(dest_raw), size))

            if not city_raw_entries:
                logging.warning("No raw blobs for level %s to publish", level_str)
                continue

            total_raw = sum(s for _, _, s in city_raw_entries)

            if total_raw <= MAX_COMBINED_BYTES:
                sources = [header_blob]
                for i, (c, rb, _) in enumerate(city_raw_entries):
                    if i > 0:
                        sources.append(comma_blob)
                    sources.append(rb)
                sources.append(footer_blob)

                dest = f"USA_Combined/{rel_path}.geojson"
                dest_blob = out_bucket.blob(dest)
                _gcs_compose(dest_blob, sources)
                logging.info(
                    "Published combined: gs://%s/%s (%d cities, %.1f GB)",
                    output_bucket,
                    dest,
                    len(city_raw_entries),
                    total_raw / (1024**3),
                )
            else:
                parts = []
                current_part = []
                current_size = 0
                for entry in city_raw_entries:
                    c, rb, size = entry
                    if current_part and current_size + size > MAX_COMBINED_BYTES:
                        parts.append(current_part)
                        current_part = []
                        current_size = 0
                    current_part.append(entry)
                    current_size += size
                if current_part:
                    parts.append(current_part)

                for pi, part_entries in enumerate(parts, 1):
                    sources = [header_blob]
                    for i, (c, rb, _) in enumerate(part_entries):
                        if i > 0:
                            sources.append(comma_blob)
                        sources.append(rb)
                    sources.append(footer_blob)

                    dest = f"USA_Combined/{rel_path}_part{pi}.geojson"
                    dest_blob = out_bucket.blob(dest)
                    _gcs_compose(dest_blob, sources)
                    part_cities = [c for c, _, _ in part_entries]
                    logging.info(
                        "Published combined part %d/%d: gs://%s/%s (cities: %s)",
                        pi,
                        len(parts),
                        output_bucket,
                        dest,
                        part_cities,
                    )

        except Exception as e:
            logging.error("Failed to publish level %s: %s", level_str, e)

    try:
        for blob in client.list_blobs(output_bucket, prefix="_compose/"):
            blob.delete()
        for blob in client.list_blobs(output_bucket, prefix="USA_Combined/"):
            if "__tmp" in blob.name:
                blob.delete()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


def _run_pipeline(
    config_dict, work_dir, run_mode="full", on_level_done=None, h3_breadcrumbs=None
):
    config_dict["output_dir"] = os.path.join(work_dir, "output")
    os.makedirs(config_dict["output_dir"], exist_ok=True)

    logging.info(
        "Starting pipeline with %d cities, mode=%s",
        len(config_dict["cities"]),
        run_mode,
    )

    admin_outputs = {}
    h3_outputs = {}

    if run_mode in ("full", "admin_only"):
        admin_outputs = process_admin_levels(config_dict, on_level_done=on_level_done)
        logging.info("Admin boundary tilesets created: %d", len(admin_outputs))

    h3_breadcrumb_lookup = None
    if run_mode in ("full", "h3_only"):
        level_10_path = os.path.join(
            config_dict["output_dir"], "admin_level_10_all_cities.geojson"
        )
        h3_breadcrumb_lookup = build_h3_breadcrumb_index(level_10_path)
        if h3_breadcrumb_lookup:
            logging.info("Built H3 spatial breadcrumb index from %s", level_10_path)

        h3_outputs = process_h3_tilesets(
            config_dict,
            max_workers=4,
            on_resolution_done=on_level_done,
            h3_breadcrumbs=h3_breadcrumbs,
            h3_breadcrumb_lookup=h3_breadcrumb_lookup,
        )
        logging.info("H3 tilesets created: %d", len(h3_outputs))

    all_outputs = {**admin_outputs, **h3_outputs}
    total_size_mb = sum(
        os.path.getsize(fp) / (1024 * 1024)
        for fp in all_outputs.values()
        if os.path.exists(fp)
    )
    return {
        "status": "success",
        "tilesets_created": len(all_outputs),
        "total_size_mb": round(total_size_mb, 2),
        "output_files": list(all_outputs.values()),
        "admin_levels": list(admin_outputs.keys()),
        "h3_resolutions": list(h3_outputs.keys()),
    }


# ---------------------------------------------------------------------------
# JRC water polygon: auto-generate at pipeline time
# ---------------------------------------------------------------------------


def _get_jrc_tile_name(left_lon, top_lat):
    lon_dir = "E" if left_lon >= 0 else "W"
    lat_dir = "N" if top_lat >= 0 else "S"
    return f"{int(abs(left_lon))}{lon_dir}_{int(abs(top_lat))}{lat_dir}"


def _get_jrc_tiles_for_bbox(bbox):
    import math

    min_lon, min_lat, max_lon, max_lat = bbox
    start_lon = int(math.floor(min_lon / 10) * 10)
    end_lon = int(math.floor(max_lon / 10) * 10)
    start_top = int(math.ceil(min_lat / 10) * 10)
    end_top = int(math.ceil(max_lat / 10) * 10)
    tiles = []
    for left in range(start_lon, end_lon + 10, 10):
        for top in range(start_top, end_top + 10, 10):
            tiles.append(_get_jrc_tile_name(left, top))
    return tiles


def _generate_water_polygon(tiff_bbox, work_dir):
    import requests
    import rasterio
    from rasterio.features import shapes as rasterio_shapes
    from rasterio.mask import mask as rasterio_mask
    from rasterio.merge import merge as rasterio_merge
    import numpy as np
    from shapely.geometry import shape, box, MultiPolygon, Polygon
    from shapely.ops import unary_union
    from shapely.validation import make_valid

    buffer = 0.05
    bbox = (
        tiff_bbox[0] - buffer,
        tiff_bbox[1] - buffer,
        tiff_bbox[2] + buffer,
        tiff_bbox[3] + buffer,
    )

    tiles = _get_jrc_tiles_for_bbox(bbox)
    logging.info("JRC tiles needed: %s", tiles)

    tile_paths = []
    for tile_name in tiles:
        url = JRC_TILE_URL.format(tile=tile_name)
        local_path = os.path.join(work_dir, f"jrc_{tile_name}.tif")
        logging.info("Downloading JRC tile %s...", tile_name)
        resp = requests.get(url, stream=True, timeout=300)
        if resp.status_code == 404:
            logging.info("JRC tile %s not found (no data)", tile_name)
            continue
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        tile_paths.append(local_path)
        logging.info(
            "Downloaded JRC tile %s (%.1f MB)",
            tile_name,
            os.path.getsize(local_path) / 1e6,
        )

    if not tile_paths:
        logging.warning("No JRC tiles found — skipping water polygon")
        return None

    datasets = [rasterio.open(p) for p in tile_paths]
    if len(datasets) == 1:
        merged_data = datasets[0].read(1)
        merged_transform = datasets[0].transform
        merged_crs = datasets[0].crs
        merged_shape = merged_data.shape
    else:
        merged_data, merged_transform = rasterio_merge(datasets)
        merged_data = merged_data[0]
        merged_crs = datasets[0].crs
        merged_shape = merged_data.shape
    for ds in datasets:
        ds.close()

    bbox_geom = box(*bbox)
    merged_path = os.path.join(work_dir, "jrc_merged.tif")
    with rasterio.open(
        merged_path,
        "w",
        driver="GTiff",
        height=merged_shape[0],
        width=merged_shape[1],
        count=1,
        dtype=merged_data.dtype,
        crs=merged_crs,
        transform=merged_transform,
    ) as dst:
        dst.write(merged_data, 1)
    del merged_data

    with rasterio.open(merged_path) as src:
        cropped, crop_transform = rasterio_mask(src, [bbox_geom], crop=True, nodata=0)
        cropped = cropped[0]
    os.unlink(merged_path)

    water_mask = (cropped >= JRC_OCCURRENCE_THRESHOLD).astype(np.uint8)
    water_count = int(water_mask.sum())
    logging.info(
        "JRC water pixels (>=%d%% occurrence): %d",
        JRC_OCCURRENCE_THRESHOLD,
        water_count,
    )
    del cropped

    if water_count == 0:
        logging.info("No permanent water found — skipping water polygon")
        return None

    polygons = []
    for geom_dict, value in rasterio_shapes(water_mask, transform=crop_transform):
        if value == 1:
            poly = shape(geom_dict)
            if poly.is_valid and not poly.is_empty:
                polygons.append(poly)
    del water_mask

    if not polygons:
        return None

    logging.info("Merging %d water polygons...", len(polygons))
    water_union = unary_union(polygons)
    water_union = make_valid(water_union)
    del polygons

    if water_union.geom_type == "Polygon":
        water_union = MultiPolygon([water_union])
    elif water_union.geom_type == "GeometryCollection":
        poly_parts = [
            g for g in water_union.geoms if isinstance(g, (Polygon, MultiPolygon))
        ]
        water_union = unary_union(poly_parts) if poly_parts else None
        if water_union and water_union.geom_type == "Polygon":
            water_union = MultiPolygon([water_union])

    if water_union is None or water_union.is_empty:
        return None

    for p in tile_paths:
        if os.path.exists(p):
            os.unlink(p)

    logging.info(
        "Water polygon: %s, bounds=%s", water_union.geom_type, water_union.bounds
    )
    return water_union


def _ensure_water_polygon(city_name, tiff_bbox, pipeline_bucket, work_dir):
    import pickle

    pkl_name = f"{city_name}_coastal_water.pkl"
    gcs_path = f"{WATER_POLYGONS_PREFIX}/{pkl_name}"
    local_pkl = os.path.join(work_dir, pkl_name)

    client = storage.Client()
    bucket_obj = client.bucket(pipeline_bucket)
    blob = bucket_obj.blob(gcs_path)

    if blob.exists():
        logging.info("Water polygon cache hit: %s", gcs_path)
        blob.download_to_filename(local_pkl)
        return local_pkl

    if tiff_bbox is None:
        logging.warning("No tiff_bbox — cannot generate water polygon")
        return None

    logging.info("Generating JRC water polygon for %s...", city_name)
    water_poly = _generate_water_polygon(tiff_bbox, work_dir)
    if water_poly is None:
        return None

    with open(local_pkl, "wb") as f:
        pickle.dump(water_poly, f)

    blob.upload_from_filename(local_pkl)
    logging.info("Uploaded water polygon to gs://%s/%s", pipeline_bucket, gcs_path)
    return local_pkl


# ---------------------------------------------------------------------------
# Core processing: takes a city_folder name and runs end to end
# ---------------------------------------------------------------------------


def _process_city_from_predictions(
    pipeline_bucket, predictions_bucket, city_folder, work_dir
):
    """Fully automatic: derive everything from the city_folder name."""
    city_name, config_prefix, mosaic_filename = _parse_city_folder(city_folder)

    city_config = _build_city_config(city_name, city_folder, predictions_bucket)
    config_dict = _build_pipeline_config(work_dir, pipeline_bucket, [city_config])

    logging.info(
        "Processing %s (folder=%s, predictions=%s)",
        city_name,
        city_folder,
        predictions_bucket,
    )

    csv_cache_prefix = "h3_csv_cache"
    cache_blob = f"{csv_cache_prefix}/{city_name}_max.csv"
    bbox_blob = f"{csv_cache_prefix}/{city_name}_bbox.txt"
    client = storage.Client()
    bucket_obj = client.bucket(pipeline_bucket)

    if bucket_obj.blob(cache_blob).exists():
        logging.info("Cache hit: downloading %s", cache_blob)
        local_h3_csv = os.path.join(work_dir, f"{city_name}_max.csv")
        _download_from_gcs(pipeline_bucket, cache_blob, local_h3_csv)

        tiff_bbox = None
        if bucket_obj.blob(bbox_blob).exists():
            local_bbox = os.path.join(work_dir, f"{city_name}_bbox.txt")
            _download_from_gcs(pipeline_bucket, bbox_blob, local_bbox)
            with open(local_bbox) as f:
                parts = f.read().strip().split(",")
                tiff_bbox = tuple(float(x) for x in parts)
    else:
        logging.info("Preprocessing tiffs for %s...", city_name)
        local_h3_csv, tiff_bbox = preprocess_city(
            city_config=city_config,
            scenario_mapping=SCENARIO_MAPPING,
            work_dir=work_dir,
            n_workers=4,
            intermediates_bucket=pipeline_bucket,
            intermediates_prefix="intermediates",
        )
        _upload_to_gcs(pipeline_bucket, local_h3_csv, cache_blob)
        if tiff_bbox is not None:
            local_bbox = os.path.join(work_dir, f"{city_name}_bbox.txt")
            with open(local_bbox, "w") as f:
                f.write(",".join(str(x) for x in tiff_bbox))
            _upload_to_gcs(pipeline_bucket, local_bbox, bbox_blob)

    city_config["tiff_bbox"] = tiff_bbox
    city_config["h3_csv"] = local_h3_csv

    _derive_admin_fields(city_config, tiff_bbox, config_dict["admin_boundaries"])

    water_pkl = _ensure_water_polygon(city_name, tiff_bbox, pipeline_bucket, work_dir)
    if water_pkl:
        city_config["water_polygon_path"] = water_pkl

    os.makedirs(config_dict["output_dir"], exist_ok=True)

    output_dir = os.path.join(work_dir, "output")
    uploaded_levels = []
    state_fips = city_config.get("state_fips")
    city_output_dir = _city_output_folder(city_name, state_fips)

    def _on_level_done(level, output_path):
        """Track completed levels; breadcrumbs are added per level."""
        try:
            _add_breadcrumbs_to_outputs(
                os.path.dirname(output_path), config_dict=config_dict
            )
            uploaded_levels.append(str(level if level < 100 else level))
        except Exception as e:
            logging.error("Failed to process level %s: %s", level, e)

    h3_breadcrumbs = {city_name: _build_city_breadcrumbs(city_config)}

    results = _run_pipeline(
        config_dict,
        work_dir,
        run_mode="full",
        on_level_done=_on_level_done,
        h3_breadcrumbs=h3_breadcrumbs,
    )

    _merge_and_upload_outputs(pipeline_bucket, "output/", output_dir, {city_name})

    out_bucket = os.environ.get("OUTPUT_BUCKET", OUTPUT_BUCKET)

    _publish_combined_outputs(output_dir, city_output_dir, out_bucket,
                              pipeline_bucket, state_fips=state_fips)

    if local_h3_csv and os.path.exists(local_h3_csv):
        client = storage.Client()
        bucket_obj = client.bucket(out_bucket)
        csv_dest = f"{city_output_dir}/h3_sampled.csv"
        bucket_obj.blob(csv_dest).upload_from_filename(local_h3_csv)
        logging.info("Published H3 CSV: gs://%s/%s", out_bucket, csv_dest)

    results["city"] = city_name
    results["gcs_output"] = f"gs://{pipeline_bucket}/output/"
    results["output_bucket"] = out_bucket
    results["city_output_folder"] = city_output_dir

    _sync_to_mongodb(output_dir, city_name)

    return results


# ---------------------------------------------------------------------------
# Combined output publishing
# ---------------------------------------------------------------------------


def _combine_geojson_files(file_paths):
    """Merge multiple GeoJSON FeatureCollections into one."""
    import json as _json
    all_features = []
    for path in file_paths:
        try:
            with open(path) as f:
                fc = _json.load(f)
            all_features.extend(fc.get("features", []))
        except Exception as e:
            logging.error("Failed to read %s for combining: %s", path, e)
    return {"type": "FeatureCollection", "features": all_features}


def _publish_combined_outputs(
    output_dir, city_folder, output_bucket, pipeline_bucket, state_fips=None,
):
    """Publish combined GeoJSON files to the output bucket.

    Per-city:   {city_folder}/h3_{city_folder}.geojson  (H3 res 6-12 combined)
    National:   USA_Combined/admin.geojson (all cities' admin levels 6+8+10, + level 4 NYC)
    """
    import glob as _glob
    import json as _json
    import re as _re

    output_files = _glob.glob(os.path.join(output_dir, "*.geojson"))
    if not output_files:
        return

    h3_files = []
    admin_files = []
    for local_path in sorted(output_files):
        fname = os.path.basename(local_path)
        m = _re.search(r"admin_level_(\d+)_", fname)
        if not m:
            continue
        level = int(m.group(1))
        if level > 100:
            h3_files.append(local_path)
        else:
            admin_files.append(local_path)

    client = storage.Client()
    out_bucket = client.bucket(output_bucket)

    if h3_files:
        combined_h3 = _combine_geojson_files(h3_files)
        h3_fname = f"h3_{city_folder}.geojson"
        h3_local = os.path.join(output_dir, h3_fname)
        with open(h3_local, "w") as f:
            _json.dump(combined_h3, f)
        dest = f"{city_folder}/{h3_fname}"
        out_bucket.blob(dest).upload_from_filename(h3_local)
        logging.info(
            "Published combined H3: gs://%s/%s (%d features)",
            output_bucket, dest, len(combined_h3["features"]),
        )

    _publish_national_admin(admin_files, output_bucket)


def _publish_national_admin(local_admin_files, output_bucket):
    """Merge this city's admin files into USA_Combined/admin.geojson.

    Reads existing admin.geojson from GCS (other cities), adds this city's
    admin features, and re-uploads the combined file.
    """
    import json as _json

    if not local_admin_files:
        return

    client = storage.Client()
    out_bucket = client.bucket(output_bucket)

    all_features = []
    existing_blob = out_bucket.blob("USA_Combined/admin.geojson")
    if existing_blob.exists():
        try:
            existing = _json.loads(existing_blob.download_as_text())
            all_features.extend(existing.get("features", []))
            logging.info(
                "Read existing national admin: %d features", len(all_features)
            )
        except Exception as e:
            logging.error("Failed to read existing admin.geojson: %s", e)

    new_features = _combine_geojson_files(local_admin_files).get("features", [])

    new_cities = {
        f.get("properties", {}).get("city") for f in new_features if f.get("properties")
    }
    if new_cities:
        all_features = [
            f for f in all_features
            if f.get("properties", {}).get("city") not in new_cities
        ]

    all_features.extend(new_features)

    if not all_features:
        logging.info("No admin features for national merge")
        return

    combined = {"type": "FeatureCollection", "features": all_features}
    existing_blob.upload_from_string(
        _json.dumps(combined), content_type="application/json"
    )
    logging.info(
        "Published national admin: gs://%s/USA_Combined/admin.geojson (%d features)",
        output_bucket, len(all_features),
    )


# ---------------------------------------------------------------------------
# MongoDB sync
# ---------------------------------------------------------------------------

MONGO_ADMIN_COLLECTION = "admin_level"
MONGO_H3_COLLECTION = "h3"


def _get_mongo_uri():
    uri = os.environ.get("MONGODB_URI")
    if uri:
        return uri
    secret_name = os.environ.get(
        "MONGODB_SECRET", "projects/climateiq-test/secrets/mongodb-uri/versions/latest"
    )
    try:
        from google.cloud import secretmanager
        sm = secretmanager.SecretManagerServiceClient()
        resp = sm.access_secret_version(request={"name": secret_name})
        return resp.payload.data.decode("UTF-8")
    except Exception as e:
        logging.warning("Could not read MongoDB secret: %s", e)
        return None


def _sync_to_mongodb(output_dir, city_name):
    """Push combined GeoJSON to MongoDB Atlas.

    Reads the combined h3_{city_folder}.geojson → h3 collection
    Reads all admin-level files (4/6/8/10) → admin_level collection
    Upserts by (city, admin_level, boundary_id/cell_code).
    """
    import json as _json
    import glob as _glob
    import re as _re

    mongo_uri = _get_mongo_uri()
    if not mongo_uri:
        logging.info("MongoDB URI not configured — skipping sync")
        return

    try:
        from pymongo import MongoClient, ReplaceOne
    except ImportError:
        logging.warning("pymongo not installed — skipping MongoDB sync")
        return

    try:
        client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=30000,
            socketTimeoutMS=300000,
            connectTimeoutMS=30000,
            retryWrites=True,
        )
        db = client[os.environ.get("MONGODB_DATABASE", "climateiq_data")]
    except Exception as e:
        logging.error("MongoDB connection failed: %s", e)
        return

    h3_files = [f for f in os.listdir(output_dir) if f.startswith("h3_") and f.endswith(".geojson")]
    if h3_files:
        h3_path = os.path.join(output_dir, h3_files[0])
        _mongo_replace_city_file(db[MONGO_H3_COLLECTION], h3_path, city_name, _json)
    else:
        logging.info("No combined H3 file found — skipping H3 MongoDB sync")

    admin_features = []
    for local_path in sorted(_glob.glob(os.path.join(output_dir, "*.geojson"))):
        fname = os.path.basename(local_path)
        m = _re.search(r"admin_level_(\d+)_", fname)
        if not m:
            continue
        level = int(m.group(1))
        if level > 100:
            continue
        try:
            with open(local_path) as f:
                fc = _json.load(f)
            feats = fc.get("features", [])
            if level == 8:
                feats = [f for f in feats if f.get("properties", {}).get("is_city") is True]
            admin_features.extend(feats)
        except Exception as e:
            logging.error("Failed to read %s: %s", fname, e)

    if admin_features:
        docs = [
            {"type": "Feature", "properties": f.get("properties", {}), "geometry": f.get("geometry")}
            for f in admin_features
        ]
        db[MONGO_ADMIN_COLLECTION].delete_many({"properties.city": city_name})
        _mongo_batch_insert(db[MONGO_ADMIN_COLLECTION], docs, city_name)

    try:
        client.close()
    except Exception:
        pass


def _mongo_replace_city_file(collection, file_path, city_name, _json):
    """Replace a city's docs: remove old city data, bulk insert new from GeoJSON file."""
    try:
        with open(file_path) as f:
            fc = _json.load(f)
    except Exception as e:
        logging.error("Failed to read %s: %s", file_path, e)
        return
    features = fc.get("features", [])
    if not features:
        return
    docs = [
        {"type": "Feature", "properties": f.get("properties", {}), "geometry": f.get("geometry")}
        for f in features
    ]
    del fc, features
    result = collection.delete_many({"properties.city": city_name})
    logging.info("MongoDB %s: removed %d old %s docs", collection.name, result.deleted_count, city_name)
    _mongo_batch_insert(collection, docs, city_name)


def _mongo_batch_insert(collection, docs, city_name, batch_size=5000):
    """Insert docs in batches of 5000 with retry."""
    import time as _time
    total = len(docs)
    done = 0
    for i in range(0, total, batch_size):
        batch = docs[i : i + batch_size]
        for attempt in range(5):
            try:
                collection.insert_many(batch, ordered=False)
                break
            except Exception as e:
                if attempt >= 4:
                    logging.error("MongoDB insert failed after 5 retries at %d/%d: %s", done, total, e)
                    raise
                _time.sleep(min(30, 5 * (attempt + 1)))
        done += len(batch)
        if done % 50000 < batch_size:
            logging.info("MongoDB %s %s: %d/%d", collection.name, city_name, done, total)
    logging.info("MongoDB %s %s: DONE %d docs", collection.name, city_name, total)


# ---------------------------------------------------------------------------
# HTTP entry point
# ---------------------------------------------------------------------------


@functions_framework.http
@_error_to_response
def process_h3_pipeline(request: flask.Request):  # type: ignore[return]
    """Process a city. Requires 'city_folder' and 'predictions_bucket' in JSON body.

    Example: {"city_folder": "NYC_Predictions",
    "predictions_bucket": "test-climateiq-predictions"}
    """
    request_json = request.get_json(silent=True) or {}

    pipeline_bucket = request_json.get("bucket") or os.environ.get("GCS_BUCKET")
    if not pipeline_bucket:
        return (
            flask.jsonify(
                {
                    "error": "GCS_BUCKET environment variable"
                    " or 'bucket' parameter required"
                }
            ),
            400,
        )

    city_folder = request_json.get("city_folder")
    predictions_bucket = request_json.get("predictions_bucket")

    if not city_folder or not predictions_bucket:
        return (
            flask.jsonify(
                {"error": "'city_folder' and 'predictions_bucket' are required"}
            ),
            400,
        )

    logging.info(
        "Starting H3 pipeline — city_folder=%s predictions=%s",
        city_folder,
        predictions_bucket,
    )

    with tempfile.TemporaryDirectory() as work_dir:
        results = _process_city_from_predictions(
            pipeline_bucket, predictions_bucket, city_folder, work_dir
        )
        logging.info("Pipeline completed successfully for %s", results.get("city"))
        return flask.jsonify(results)


# ---------------------------------------------------------------------------
# Cloud Storage event trigger entry point
# ---------------------------------------------------------------------------


@functions_framework.cloud_event
@_retry_and_report_errors()
def process_h3_pipeline_on_tiff_upload(
    cloud_event: functions_framework.CloudEvent,
) -> None:
    """Auto-triggers when a mosaic TIF is uploaded to the predictions bucket.

    Detects city from the path, checks if all 9 scenarios are present,
    and kicks off the full pipeline. No config file needed.
    """
    data = cloud_event.data
    src_bucket_name = data["bucket"]
    file_name = data["name"]

    logging.info("GCS finalize: gs://%s/%s", src_bucket_name, file_name)

    if not file_name.startswith("flood_predictions/"):
        return

    parts = file_name.split("/")
    if len(parts) < 4:
        return

    city_folder = parts[1]
    mosaic_filename = parts[-1]

    if not mosaic_filename.lower().endswith(".tif"):
        return
    if "chunk" in mosaic_filename.lower():
        return

    city_name, config_prefix, expected_mosaic = _parse_city_folder(city_folder)

    logging.info("Mosaic TIF detected: city=%s folder=%s", city_name, city_folder)

    pipeline_bucket = os.environ.get("GCS_BUCKET")
    if not pipeline_bucket:
        logging.error("GCS_BUCKET environment variable is required")
        return

    client = storage.Client()
    src_bucket_obj = client.bucket(src_bucket_name)

    present = []
    missing = []
    for i in range(1, EXPECTED_SCENARIO_COUNT + 1):
        scenario_file = f"Rainfall_Data_{i}.txt"
        blob_path = (
            f"flood_predictions/{city_folder}/"
            f"{config_prefix}%2F{scenario_file}/{expected_mosaic}"
        )
        if src_bucket_obj.blob(blob_path).exists():
            present.append(scenario_file)
        else:
            missing.append(scenario_file)

    logging.info(
        "%s: %d/%d scenarios ready", city_name, len(present), EXPECTED_SCENARIO_COUNT
    )
    if missing:
        logging.info("Still waiting for: %s", missing)
        return

    pipeline_bucket_obj = client.bucket(pipeline_bucket)
    cache_blob = f"h3_csv_cache/{city_name}_max.csv"
    if pipeline_bucket_obj.blob(cache_blob).exists():
        logging.info("%s: CSV cache already exists — skipping", city_name)
        return

    logging.info(
        "%s: All %d scenarios ready — starting pipeline!",
        city_name,
        EXPECTED_SCENARIO_COUNT,
    )

    with tempfile.TemporaryDirectory() as work_dir:
        results = _process_city_from_predictions(
            pipeline_bucket, src_bucket_name, city_folder, work_dir
        )
        logging.info("Pipeline result for %s: %s", city_name, results.get("status"))

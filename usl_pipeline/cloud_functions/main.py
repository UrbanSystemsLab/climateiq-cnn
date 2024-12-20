import dataclasses
import datetime
import functools
import io
import pathlib
import logging
import re
import tarfile
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
    """

    elevation_min: float | None = None
    elevation_max: float | None = None
    chunk_size: int | None = None
    time: datetime.datetime | None = None


def _retry_and_report_errors(
    error_reporter_func: (
        Callable[[functions_framework.CloudEvent, Exception], None] | None
    ) = None
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
        func: Callable[[functions_framework.CloudEvent], None]
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


def _build_feature_matrix(
    bucket_name: str, chunk_path: str, output_bucket: str
) -> None:
    """Builds a feature matrix when a set of geo files is uploaded."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    chunk_blob = bucket.blob(chunk_path)

    with chunk_blob.open("rb") as chunk:
        study_area_name, chunk_name = _parse_chunk_path(chunk_path)
        # Flood (CityCat)
        if chunk_path.endswith(".tar"):
            feature_file_name = pathlib.PurePosixPath(chunk_path).with_suffix(".npy")
            feature_blob = storage_client.bucket(output_bucket).blob(
                str(feature_file_name)
            )
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
            logging.info(
                "Start generating flood feature matrix for chunk %s", chunk_path
            )
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

        # Heat (WRF) - treat one WPS outout file as one chunk
        elif re.search(file_names.WPS_DOMAIN3_NC_REGEX, chunk_path):
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
    x_index, y_index = _parse_spatial_chunk_indices_from_name(chunk_blob.name)

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
    study_area_name, chunk_name = _parse_chunk_path(chunk_path)
    metastore.StudyAreaChunk(id_=chunk_name, error=error_message).merge(
        db, study_area_name
    )


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
    # Let's look up chunk metadata and check if scaling is needed
    db = firestore.Client()
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

    _start_feature_rescaling_if_ready(
        storage.Client().bucket(cloud_event.data["bucket"]), cloud_event.data["name"]
    )

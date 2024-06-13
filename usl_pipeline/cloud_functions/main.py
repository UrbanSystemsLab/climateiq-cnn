import dataclasses
import datetime
import functools
import io
import pathlib
import logging
import re
import tarfile
from typing import BinaryIO, Callable, IO, Optional, TextIO, Tuple

from google.api_core import exceptions
from google.cloud import error_reporting
from google.cloud import firestore
from google.cloud import storage
import functions_framework
import numpy
from numpy.typing import NDArray
import xarray
import rasterio

from usl_lib.chunkers import raster_chunkers
from usl_lib.readers import simulation_readers
from usl_lib.readers import config_readers
from usl_lib.readers import elevation_readers
import usl_lib.shared.wps_data as wps_data
from usl_lib.storage import cloud_storage
from usl_lib.storage import file_names
from usl_lib.storage import metastore

# How long to accept cloud function invocations after the triggering
# event. It's up to GCP how frequently withing this period the cloud
# function is retried. This also needs to include lag for the length
# of time between when an event is triggered and when the cloud
# function actually runs, which can be a bit of time if there are a
# lot of triggers.
_MAX_RETRY_SECONDS = 60 * 60

# The vector is long enough to express rainfall every five minutes for up to three days.
_RAINFALL_VECTOR_LENGTH = (60 // 5) * 24 * 3


@dataclasses.dataclass(slots=True)
class FeatureMetadata:
    """Additional information about the extracted features.

    Attributes:
        elevation_min: The lowest elevation height encountered.
        elevation_max: The highest elevation height encountered.
    """

    elevation_min: Optional[float] = None
    elevation_max: Optional[float] = None


def _retry_and_report_errors(
    error_reporter_func: Optional[
        Callable[[functions_framework.CloudEvent, Exception], None]
    ] = None
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
    if file_name.name == file_names.ELEVATION_TIF:
        storage_client = storage.Client()
        db = firestore.Client()

        bucket = storage_client.bucket(cloud_event.data["bucket"])
        blob = bucket.blob(str(file_name))

        with blob.open("rb") as fd:
            # We're only reading the header, so reading the first MB is plenty.
            header = elevation_readers.read_from_geotiff(
                rasterio.io.MemoryFile(fd.read(1048576)), header_only=True
            ).header
        # File names should be in the form <study_area_name>/<file_name>
        study_area_name = file_name.parent.name

        study_area = metastore.StudyArea(
            name=study_area_name,
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
@_retry_and_report_errors(
    lambda cloud_event, exc: _write_chunk_metastore_error(
        cloud_event.data["name"], str(exc)
    )
)
def build_feature_matrix(cloud_event: functions_framework.CloudEvent) -> None:
    """Builds a feature matrix when an archive of geo files is uploaded.

    This function is triggered when archive files containing geo data are uploaded to
    GCS. It produces a feature matrix for the geo data and writes that feature matrix to
    another GCS bucket for eventual use in model training and prediction.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(cloud_event.data["bucket"])
    chunk_name = cloud_event.data["name"]
    chunk_blob = bucket.blob(chunk_name)

    with chunk_blob.open("rb") as archive:
        feature_matrix, metadata = _build_feature_matrix_from_archive(archive)
        if feature_matrix is None:
            raise ValueError(f"Empty archive found in {chunk_blob}")

    feature_file_name = pathlib.PurePosixPath(chunk_name).with_suffix(".npy")
    feature_blob = storage_client.bucket(cloud_storage.FEATURE_CHUNKS_BUCKET).blob(
        str(feature_file_name)
    )

    _write_as_npy(feature_blob, feature_matrix)
    _write_metastore_entry(chunk_blob, feature_blob, metadata)


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
    x_index, y_index = blob_path.parts[-2].split("_")
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

    metastore.SimulationLabelChunk(
        gcs_uri=f"gs://{labels_bucket.name}/{label_path}",
        x_index=int(x_index),
        y_index=int(y_index),
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


def _build_feature_matrix_from_archive(
    archive: BinaryIO,
) -> Tuple[Optional[NDArray], FeatureMetadata]:
    """Builds a feature matrix for the given archive.

    Args:
      archive: The file containing the archive.

    Returns:
      A tuple of (array, metadata) for the feature matrix and
      metadata describing the feature matrix.
    """
    with tarfile.TarFile(fileobj=archive) as tar:
        for member in tar:
            fd = tar.extractfile(member)
            if fd is None:
                continue

            name = pathlib.PurePosixPath(member.name).name
            # TODO: Group logic branch by path (per hazard model)
            if name == file_names.ELEVATION_TIF:
                return _read_elevation_features(fd)
            # Handle heat model input files (WPS)
            elif name.startswith("met_em") and name.endswith(".nc"):
                return _build_wps_feature_matrix(fd)
            # TODO: handle additional archive members.
            else:
                logging.warning(f"Unexpected member name: {name}")

    return None, FeatureMetadata()


def _build_wps_feature_matrix(fd: IO[bytes]) -> Tuple[NDArray, FeatureMetadata]:
    # Ignore type checker error - BytesIO inherits from expected type BufferedIOBase
    # https://shorturl.at/lk4om
    with xarray.open_dataset(fd) as ds:  # type: ignore
        features_components = []
        for var_name in wps_data.ML_REQUIRED_VARS_REPO.keys():
            var_config = wps_data.ML_REQUIRED_VARS_REPO[var_name]
            feature = _process_wps_feature(
                feature=ds.data_vars[var_name], var_config=var_config
            )
            features_components.append(feature)

        features_matrix = numpy.dstack(features_components)

    # TODO: Write to metastore
    return features_matrix, FeatureMetadata()


def _process_wps_feature(feature: xarray.DataArray, var_config: dict) -> NDArray:
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
    # Drop time axis
    feature = feature.isel(Time=0)

    # Ensure order of dimension axes are: (west_east, south_north, <other spatial dims>)
    # to stay consistent with rest of data pipeline
    feature = feature.transpose("west_east", "south_north", ...)

    # FNL-derived var - extract to only first level
    if "num_metgrid_levels" in feature.dims:
        feature = feature.isel(num_metgrid_levels=0)

    feature_values = feature.values

    # Convert percentage-based units to decimal
    if var_config.get("unit") == wps_data.Unit.PERCENTAGE:
        feature_values = _convert_to_decimal(feature_values)

    # If var config requires global scaling, normalize feature values
    scaling_config = var_config.get("scaling")
    if (
        scaling_config is not None
        and scaling_config.get("type") == wps_data.ScalingType.GLOBAL
    ):
        min = scaling_config.get("min")
        max = scaling_config.get("max")
        feature_values = _apply_minmax_scaler(feature_values, min, max)

    return feature_values


def _convert_to_decimal(x):
    return x / 100


def _apply_minmax_scaler(x, minx, maxx):
    # Clip values if they go out of bounds of the specified min and max
    x[x > maxx] = maxx
    x[x < minx] = minx

    return (x - minx) / (maxx - minx)


def _read_elevation_features(fd: IO[bytes]) -> Tuple[NDArray, FeatureMetadata]:
    """Reads the elevation file into a matrix and returns metadata for the matrix.

    Args:
      fd: The file containing elevation data.

    Returns:
      A tuple of (array, metadata) for the elevation matrix and
      metadata describing the feature matrix.
    """
    elevation = elevation_readers.read_from_geotiff(
        rasterio.io.MemoryFile(fd.read())
    ).data

    if elevation is None:
        raise ValueError("Elevation file unexpectedly empty.")

    metadata = FeatureMetadata(
        elevation_min=float(elevation.min()),
        elevation_max=float(elevation.max()),
    )

    return elevation, metadata


def _write_metastore_entry(
    chunk_blob: storage.Blob, feature_blob: storage.Blob, metadata: FeatureMetadata
) -> None:
    """Updates the metastore with new information for the given chunks.

    Writes a Firestore entry for the new feature matrix chunk. Updates elvation_min and
    elevation_max values supplied in the metadata for the whole study area.

    Args:
      chunk_blob: The GCS blob containing the raw chunk archive.
      feature_blob: The GCS blob containing the feature tensor for the chunk.
      metadata: Additional metadata describing the features.
    """
    db = firestore.Client()
    study_area_name, chunk_name = _parse_chunk_path(chunk_blob.name)

    metastore.StudyAreaChunk(
        id_=chunk_name,
        archive_path=f"gs://{chunk_blob.bucket.name}/{chunk_blob.name}",
        feature_matrix_path=f"gs://{feature_blob.bucket.name}/{feature_blob.name}",
        # Remove any errors from previous failed retries which have now succeeded.
        error=firestore.DELETE_FIELD,
    ).merge(db, study_area_name)

    if metadata.elevation_min is not None and metadata.elevation_max is not None:
        metastore.StudyArea.update_min_max_elevation(
            db,
            study_area_name,
            min_=metadata.elevation_min,
            max_=metadata.elevation_max,
        )


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

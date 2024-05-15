import dataclasses
import datetime
import functools
import io
import pathlib
import logging
import tarfile
from typing import BinaryIO, Callable, IO, Optional, TextIO, Tuple

from google.cloud import error_reporting
from google.cloud import firestore
from google.cloud import storage
import functions_framework
import numpy
from numpy.typing import NDArray
import xarray
import rasterio

from usl_lib.readers import config_readers
from usl_lib.readers import elevation_readers
from usl_lib.storage import cloud_storage
from usl_lib.storage import file_names
from usl_lib.storage import metastore

_MAX_RETRY_SECONDS = 60 * 2

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
            logging.basicConfig(level=logging.WARN)

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
                    "Dropped event id %s source %s after %s seconds",
                    cloud_event["id"],
                    cloud_event["source"],
                    _MAX_RETRY_SECONDS,
                )
                return

            # Catch exceptions and report them with GCP error reporting.
            try:
                func(cloud_event)
            except Exception as exc:  # noqa
                # Log the exception to display in Cloud Function logs.
                logging.exception(exc)
                # Report the error in GCP Error Reporter.
                error_reporting.Client().report_exception()
                # Perform any custom error handling.
                if error_reporter_func is not None:
                    error_reporter_func(cloud_event, exc)

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
            gcs_path=f"gs://{config_blob.bucket.name}/{config_blob.name}",
            as_vector_gcs_path=f"gs://{vector_blob.bucket.name}/{vector_blob.name}",
            num_rainfall_entries=length,
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

        metastore.FloodScenarioConfig.delete(
            db, f"gs://{cloud_event.data['bucket']}/{cloud_event.data['name']}"
        )


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
                return _read_wps_features(fd)
            # TODO: handle additional archive members.
            else:
                logging.warning(f"Unexpected member name: {name}")

    return None, FeatureMetadata()


def _read_wps_features(fd: IO[bytes]) -> Tuple[NDArray, FeatureMetadata]:
    # Ignore type checker error - BytesIO inherits from expected type BufferedIOBase
    # https://shorturl.at/lk4om
    with xarray.open_dataset(fd) as ds:  # type: ignore
        feature = ds.data_vars["SNOALB"].values
        # Drop time axis - each WPS output file will correspond to single time value
        feature = numpy.squeeze(feature, axis=0)

    # TODO: Write to metastore
    return feature, FeatureMetadata()


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

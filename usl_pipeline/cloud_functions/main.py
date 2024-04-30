import dataclasses
import datetime
import io
import pathlib
import logging
import tarfile
from typing import BinaryIO, IO, Optional, Tuple

from google.cloud import error_reporting
from google.cloud import firestore
from google.cloud import storage
import functions_framework
import numpy
from numpy.typing import NDArray
import rasterio

from usl_lib.readers import elevation_readers
from usl_lib.storage import cloud_storage
from usl_lib.storage import metastore

_MAX_RETRY_SECONDS = 20


@dataclasses.dataclass(slots=True)
class FeatureMetadata:
    """Additional information about the extracted features.

    Attributes:
        elevation_min: The lowest elevation height encountered.
        elevation_max: The highest elevation height encountered.
    """

    elevation_min: Optional[float] = None
    elevation_max: Optional[float] = None


@functions_framework.cloud_event
def build_feature_matrix(cloud_event: functions_framework.CloudEvent) -> None:
    """Builds a feature matrix when an archive of geo files is uploaded.

    This function is triggered when archive files containing geo data are uploaded to
    GCP. It produces a feature matrix for the geo data and writes that feature matrix to
    another GCP bucket for eventual use in model training and prediction.
    """
    logging.basicConfig(level=logging.WARN)

    # To handle retries, check the time created and return if the event is too old.
    # GCP retries until the function returns successfully, so the pattern is to report
    # exceptions to have the function retried and return normally to stop retries.
    # https://cloud.google.com/functions/docs/bestpractices/retries
    event_time = datetime.datetime.fromisoformat(cloud_event.data["timeCreated"])
    event_age = (
        datetime.datetime.now(datetime.timezone.utc) - event_time
    ).total_seconds()
    if event_age > _MAX_RETRY_SECONDS:
        logging.error(
            "Dropped %s after %s seconds", cloud_event.data["id"], _MAX_RETRY_SECONDS
        )
        return

    # Catch exceptions and report them with GCP error reporting.
    try:
        _build_feature_matrix(cloud_event)
    except Exception as exc:  # noqa
        # Log the exception to display in Cloud Function logs.
        logging.exception(exc)
        # Report the error in GCP Error Reporter.
        error_reporting.Client().report_exception()
        # Write the error to the metastore.
        _write_chunk_metastore_error(cloud_event.data["name"], str(exc))


def _build_feature_matrix(cloud_event: functions_framework.CloudEvent) -> None:
    """Builds a feature matrix when an archive of geo files is uploaded."""
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

    # Write the feature matrix in .npy format to GCS.
    matrix_file = io.BytesIO()
    numpy.save(matrix_file, feature_matrix)
    # Seek to the beginning so the file can be read.
    matrix_file.flush()
    matrix_file.seek(0)
    feature_blob.upload_from_file(matrix_file)

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
            if name == "elevation.tif":
                return _read_elevation_features(fd)
            # TODO: handle additional archive members.
            else:
                logging.warning(f"Unexpected member name: {name}")

    return None, FeatureMetadata()


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

import io
import pathlib
import logging
import tarfile
from typing import BinaryIO, Optional

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


@functions_framework.cloud_event
def build_feature_matrix(cloud_event: functions_framework.CloudEvent) -> None:
    """Builds a feature matrix when an archive of geo files is uploaded.

    This function is triggered when archive files containing geo data are uploaded to
    GCP. It produces a feature matrix for the geo data and writes that feature matrix to
    another GCP bucket for eventual use in model training and prediction.
    """
    logging.basicConfig(level=logging.WARN)
    # Catch exceptions and report them with GCP error reporting.
    try:
        _build_feature_matrix(cloud_event)
    except:  # noqa
        error_reporting.Client().report_exception()
        raise


def _build_feature_matrix(cloud_event: functions_framework.CloudEvent) -> None:
    """Builds a feature matrix when an archive of geo files is uploaded."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(cloud_event.data["bucket"])
    chunk_name = cloud_event.data["name"]
    chunk_blob = bucket.blob(chunk_name)

    with chunk_blob.open("rb") as archive:
        feature_matrix = _build_feature_matrix_from_archive(archive)
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

    _write_metastore_entry(chunk_blob, feature_blob)


def _build_feature_matrix_from_archive(
    archive: BinaryIO,
) -> Optional[NDArray[numpy.float64]]:
    """Builds a feature matrix for the given archive."""
    with tarfile.TarFile(fileobj=archive) as tar:
        for member in tar:
            fd = tar.extractfile(member)
            if fd is None:
                continue

            name = pathlib.PurePosixPath(member.name).name
            if name == "elevation.tif":
                elevation = elevation_readers.read_from_geotiff(
                    rasterio.io.MemoryFile(fd.read())
                )
                return elevation.data
            # TODO: handle additional archive members.
            else:
                logging.warning(f"Unexpected member name: {name}")

    return None


def _write_metastore_entry(
    chunk_blob: storage.Blob, feature_blob: storage.Blob
) -> None:
    """Writes a Firestore entry for the new feature matrix chunk."""
    db = firestore.Client()

    # The GCS paths should be in the form study_area_name/chunk_name
    as_path = pathlib.PurePosixPath(chunk_blob.name)
    study_area_name = str(as_path.parent)
    chunk_name = as_path.stem

    metastore.StudyAreaChunk(
        id_=chunk_name,
        archive_path=f"gs://{chunk_blob.bucket.name}/{chunk_blob.name}",
        feature_matrix_path=f"gs://{feature_blob.bucket.name}/{feature_blob.name}",
    ).merge(db, study_area_name)

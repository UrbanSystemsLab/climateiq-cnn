import io
import pathlib
import logging
import tarfile
import traceback
from typing import BinaryIO

from google.cloud import error_reporting
from google.cloud import firestore
from google.cloud import storage
import functions_framework
import numpy
import rasterio

FEATURE_BUCKET_NAME = "climateiq-map-feature-chunks"


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
        error_message = traceback.format_exc().replace("\n", "  ")
        logging.error(error_message)
        raise


def _build_feature_matrix(cloud_event: functions_framework.CloudEvent) -> None:
    """Builds a feature matrix when an archive of geo files is uploaded."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(cloud_event.data["bucket"])
    blob = bucket.blob(cloud_event.data["name"])

    with blob.open("rb") as archive:
        feature_matrix = _build_feature_matrix_from_archive(archive)
        if feature_matrix is None:
            raise ValueError(f"Empty archive found in {blob}")

    feature_file_name = pathlib.PurePosixPath(cloud_event.data["name"]).with_suffix(
        ".npy"
    )
    feature_blob = storage_client.bucket(FEATURE_BUCKET_NAME).blob(
        str(feature_file_name)
    )

    # Write the feature matrix in .npy format to GCS.
    matrix_file = io.BytesIO()
    numpy.save(matrix_file, feature_matrix)
    # Seek to the beginning so the file can be read.
    matrix_file.seek(0)
    feature_blob.upload_from_file(matrix_file)

    _write_feature_metastore_entry(feature_file_name)


def _build_feature_matrix_from_archive(archive: BinaryIO) -> numpy.matrix | None:
    """Builds a feature matrix for the given archive."""
    with tarfile.TarFile(fileobj=archive) as tar:
        for member in tar:
            name = pathlib.PurePosixPath(member.name).name
            if name == "elevation.tiff":
                fd = tar.extractfile(member)
                with rasterio.open(rasterio.io.MemoryFile(fd.read())) as raster:
                    return raster.read(1)
            # TODO: handle additional archive members.
            else:
                logging.warning(f"Unexpected member name: {name}")


def _write_feature_metastore_entry(feature_file_name: pathlib.Path) -> None:
    """Writes a Firestore entry for the new feature matrix chunk."""
    db = firestore.Client()
    # The GCS paths should be in the form map_name/chunk_name.npy
    map_name = str(feature_file_name.parent)
    chunk_name = feature_file_name.stem
    chunk_ref = (
        db.collection("maps")
        .document(map_name)
        .collection("chunks")
        .document(chunk_name)
    )
    chunk_ref.set(
        {"feature_matrix_path": f"gcs://{FEATURE_BUCKET_NAME}/{feature_file_name}"},
        merge=True,
    )

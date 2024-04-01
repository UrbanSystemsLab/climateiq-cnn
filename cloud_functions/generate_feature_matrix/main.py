import io
import pathlib
import logging
import tarfile
from typing import BinaryIO

from google.cloud import error_reporting
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

    feature_file_name = str(
        pathlib.PurePosixPath(cloud_event.data["name"]).with_suffix(".npy")
    )
    feature_blob = storage_client.bucket(FEATURE_BUCKET_NAME).blob(feature_file_name)

    # Write the feature matrix in .npy format to GCS.
    matrix_file = io.BytesIO()
    numpy.save(matrix_file, feature_matrix)
    # Seek to the beginning so the file can be read.
    matrix_file.seek(0)
    feature_blob.upload_from_file(matrix_file)


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

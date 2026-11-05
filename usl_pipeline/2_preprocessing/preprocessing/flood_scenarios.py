"""Rainfall scenario records and vectors."""

import pathlib
from typing import TextIO, Tuple
from google.cloud import firestore
from google.cloud import storage
import functions_framework
import numpy
from numpy.typing import NDArray
from usl_lib.readers import config_readers
from usl_lib.shared.function_runtime import (
    retry_and_report_errors as _retry_and_report_errors,
)
from usl_lib.storage import cloud_storage
from usl_lib.storage import metastore
from preprocessing.chunks import _write_as_npy


# The vector is long enough to express rainfall every five minutes for up to three days.
_RAINFALL_VECTOR_LENGTH = (60 // 5) * 24 * 3


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

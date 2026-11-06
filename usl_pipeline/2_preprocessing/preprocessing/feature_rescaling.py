"""Rescales feature matrices once a study area's chunks land."""

import logging
import re
from google.api_core import exceptions
from google.cloud import firestore
from google.cloud import storage
import functions_framework
import numpy
from usl_lib.shared.function_runtime import (
    retry_and_report_errors as _retry_and_report_errors,
)
from usl_lib.storage import file_names
from usl_lib.storage import metastore
from usl_lib.transformers import feature_raster_transformers
from preprocessing.chunks import _parse_chunk_path


# Storage bucket file name extension that should trigger feature matrix rescaling.
_FEATURE_SCALING_TRIGGER_SUFFIX = ".scale_trigger"


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

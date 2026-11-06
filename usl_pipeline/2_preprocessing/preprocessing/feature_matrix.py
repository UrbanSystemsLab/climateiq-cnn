"""Routes a chunk to the flood or heat feature builder."""

import re
import flask
import functions_framework
from usl_lib.shared.function_runtime import error_to_response as _error_to_response
from usl_lib.shared.function_runtime import (
    retry_and_report_errors as _retry_and_report_errors,
)
from usl_lib.storage import cloud_storage
from usl_lib.storage import file_names
from preprocessing.chunks import _write_chunk_metastore_error
from preprocessing.flood_features import _build_flood_feature_matrix
from preprocessing.heat.features import _build_heat_feature_matrix


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
    """Dispatches a chunk to the flood or heat builder by file type."""
    if chunk_path.endswith(".tar"):
        _build_flood_feature_matrix(bucket_name, chunk_path, output_bucket)
    elif re.search(file_names.WPS_DOMAIN3_NC_REGEX, chunk_path):
        _build_heat_feature_matrix(bucket_name, chunk_path, output_bucket)
    else:
        raise ValueError(f"Unexpected file {chunk_path}")

"""Cloud Function entry points for stage 4."""

import dataclasses
import datetime
import json
import logging
import os

from google.auth.transport import requests as auth_requests
from google.cloud import storage
import flask
import functions_framework
import google.auth

from usl_lib.shared.function_runtime import error_to_response as _error_to_response
from usl_lib.shared.function_runtime import (
    retry_and_report_errors as _retry_and_report_errors,
)

from postprocessing import tile_publish
from postprocessing import util_io
from postprocessing.city_config import (
    EXPECTED_SCENARIO_COUNT,
    _parse_city_folder,
)

H3_PIPELINE_JOB = "h3-pipeline-job"
H3_PIPELINE_JOB_REGION = "us-central1"


@functions_framework.http
@_error_to_response
def process_h3_pipeline(request: flask.Request):  # type: ignore[return]
    """Starts a city's run of the h3-pipeline-job Cloud Run job.

    Requires 'city_folder' and 'predictions_bucket' in the JSON body.
    Example: {"city_folder": "NYC_Predictions",
    "predictions_bucket": "test-climateiq-predictions"}

    Returns {"status": "started", "execution"} without waiting for the run.
    """
    request_json = request.get_json(silent=True) or {}

    pipeline_bucket = request_json.get("bucket") or os.environ.get("GCS_BUCKET")
    if not pipeline_bucket:
        return (
            flask.jsonify(
                {
                    "error": "GCS_BUCKET environment variable"
                    " or 'bucket' parameter required"
                }
            ),
            400,
        )

    city_folder = request_json.get("city_folder")
    predictions_bucket = request_json.get("predictions_bucket")

    if not city_folder or not predictions_bucket:
        return (
            flask.jsonify(
                {"error": "'city_folder' and 'predictions_bucket' are required"}
            ),
            400,
        )

    execution = _start_city_job(pipeline_bucket, predictions_bucket, city_folder)
    return flask.jsonify(
        {"status": "started", "city_folder": city_folder, "execution": execution}
    )


@functions_framework.cloud_event
@_retry_and_report_errors()
def process_h3_pipeline_on_tiff_upload(
    cloud_event: functions_framework.CloudEvent,
) -> None:
    """Auto-triggers when a mosaic TIF is uploaded to the predictions bucket.

    Detects city from the path, checks if all 9 scenarios are present,
    and starts the h3-pipeline-job Cloud Run job. No config file needed.
    """
    data = cloud_event.data
    src_bucket_name = data["bucket"]
    file_name = data["name"]

    logging.info("GCS finalize: gs://%s/%s", src_bucket_name, file_name)

    if not file_name.startswith("flood_predictions/"):
        return

    parts = file_name.split("/")
    if len(parts) < 4:
        return

    city_folder = parts[1]
    mosaic_filename = parts[-1]

    if not mosaic_filename.lower().endswith(".tif"):
        return
    if "chunk" in mosaic_filename.lower():
        return

    city_name, config_prefix, expected_mosaic = _parse_city_folder(city_folder)

    logging.info("Mosaic TIF detected: city=%s folder=%s", city_name, city_folder)

    pipeline_bucket = os.environ.get("GCS_BUCKET")
    if not pipeline_bucket:
        logging.error("GCS_BUCKET environment variable is required")
        return

    client = storage.Client()
    src_bucket_obj = client.bucket(src_bucket_name)

    present = []
    missing = []
    for i in range(1, EXPECTED_SCENARIO_COUNT + 1):
        scenario_file = f"Rainfall_Data_{i}.txt"
        blob_path = (
            f"flood_predictions/{city_folder}/"
            f"{config_prefix}%2F{scenario_file}/{expected_mosaic}"
        )
        if src_bucket_obj.blob(blob_path).exists():
            present.append(scenario_file)
        else:
            missing.append(scenario_file)

    logging.info(
        "%s: %d/%d scenarios ready", city_name, len(present), EXPECTED_SCENARIO_COUNT
    )
    if missing:
        logging.info("Still waiting for: %s", missing)
        return

    pipeline_bucket_obj = client.bucket(pipeline_bucket)
    cache_blob = f"h3_csv_cache/{city_name}_max.csv"
    if pipeline_bucket_obj.blob(cache_blob).exists():
        logging.info("%s: CSV cache already exists — skipping", city_name)
        return

    logging.info(
        "%s: All %d scenarios ready — starting the pipeline job",
        city_name,
        EXPECTED_SCENARIO_COUNT,
    )
    _start_city_job(pipeline_bucket, src_bucket_name, city_folder)


_TILE_PUBLISH_ACTIONS = ("plan", "convert", "upload", "publish")


@functions_framework.http
@_error_to_response
def publish_tiles(request: flask.Request):  # type: ignore[return]
    """Publishes an object in the H3 tiles bucket as a tileset, one step per call.

    The h3-tile-publish workflow, which the mapbox-release GitHub workflow runs,
    calls this for a city's <ISO3>_<STATE>_<City>/h3_<ISO3>_<STATE>_<City>.geojson
    or USA_Combined/admin.geojson: "plan", "convert", "upload", then "publish".

    JSON body:
        action: "plan", "convert", "upload" or "publish".
        name: Uploaded object, e.g. "USA_MO_KansasCity/h3_USA_MO_KansasCity.geojson".
        bucket: Defaults to the GCS_BUCKET environment variable.
        force: "plan" only. Publish even if this content already was.

    Every action returns {"status": "skipped", "reason"} when the object feeds
    no tileset or TILE_TARGETS is unset. "plan" also skips an object whose MD5
    or generation is already published, and otherwise returns
    {"status": "ready", "tileset", "source_md5"}.
    "convert" stages the object as line-delimited GeoJSON, "upload" sends it to
    each tile server target, and "publish" publishes the tileset there and
    records the results in tile_sources/<tileset>/publish.json.
    """
    request_json = request.get_json(silent=True) or {}

    bucket_name = request_json.get("bucket") or os.environ.get("GCS_BUCKET")
    if not bucket_name:
        return _bad_request("GCS_BUCKET environment variable or 'bucket' required")

    action = request_json.get("action")
    if action not in _TILE_PUBLISH_ACTIONS:
        return _bad_request(f"'action' must be one of {_TILE_PUBLISH_ACTIONS}")

    object_name = request_json.get("name")
    if not object_name or not isinstance(object_name, str):
        return _bad_request("'name' must name an object in the bucket")

    tileset = tile_publish.tileset_for_object(bucket_name, object_name)
    if tileset is None:
        city_source = tile_publish.CITY_H3_SOURCE.format(area="<ISO3>_<STATE>_<City>")
        return _skip_tiles(
            object_name, f"not {city_source} or {tile_publish.NATIONAL_ADMIN_SOURCE}"
        )

    client = storage.Client()
    targets = tile_publish.make_targets(os.environ, client)
    if not targets:
        return _skip_tiles(object_name, f"{tile_publish.TILE_TARGETS_ENV} is not set")

    bucket = client.bucket(bucket_name)
    source = bucket.get_blob(object_name)
    if source is None:
        return (
            flask.jsonify({"error": f"gs://{bucket_name}/{object_name} not found"}),
            404,
        )

    if action == "plan":
        published = _published_reason(_read_publish_status(bucket, tileset), source)
        if published and not request_json.get("force"):
            return _skip_tiles(object_name, published)
        return flask.jsonify(
            {"status": "ready", "tileset": tileset.name, "source_md5": source.md5_hash}
        )

    if action == "convert":
        written = util_io.convert_blob(bucket, tileset.source_blob, tileset.staged_blob)
        return flask.jsonify(
            {
                "status": "converted" if written else "up_to_date",
                "tileset": tileset.name,
                "staged": tileset.staged_blob,
            }
        )

    if action == "upload":
        for target in targets:
            target.upload_source(tileset)
        return flask.jsonify(
            {
                "status": "uploaded",
                "tileset": tileset.name,
                "targets": [target.name for target in targets],
            }
        )

    staged = bucket.get_blob(tileset.staged_blob)
    if staged is None:
        return (
            flask.jsonify(
                {
                    "error": f"gs://{bucket_name}/{tileset.staged_blob} not found; "
                    "run convert and upload first"
                }
            ),
            409,
        )
    staged_metadata = staged.metadata or {}
    staged_generation = staged_metadata.get(util_io.SOURCE_GENERATION_KEY)

    results = [dataclasses.asdict(target.publish(tileset)) for target in targets]
    status = {
        "tileset": tileset.name,
        "source": tileset.source_blob,
        "source_generation": int(staged_generation) if staged_generation else None,
        "source_md5": staged_metadata.get(util_io.SOURCE_MD5_KEY),
        "published_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "results": results,
    }
    bucket.blob(tileset.status_blob).upload_from_string(
        json.dumps(status, indent=2), content_type="application/json"
    )
    return flask.jsonify({"status": "published", **status})


def _start_city_job(
    pipeline_bucket: str, predictions_bucket: str, city_folder: str
) -> str:
    """Starts an h3-pipeline-job execution for city_folder and returns its name."""
    credentials, project = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    url = (
        f"https://run.googleapis.com/v2/projects/{project}/locations/"
        f"{H3_PIPELINE_JOB_REGION}/jobs/{H3_PIPELINE_JOB}:run"
    )
    env = {
        "CITY_FOLDER": city_folder,
        "PREDICTIONS_BUCKET": predictions_bucket,
        "GCS_BUCKET": pipeline_bucket,
    }
    body = {
        "overrides": {
            "containerOverrides": [
                {"env": [{"name": k, "value": v} for k, v in env.items()]}
            ]
        }
    }
    response = auth_requests.AuthorizedSession(credentials).post(
        url, json=body, timeout=60
    )
    response.raise_for_status()
    execution = response.json()["metadata"]["name"]
    logging.info("Started %s for %s", execution, city_folder)
    return execution


def _bad_request(message: str) -> tuple[flask.Response, int]:
    return flask.jsonify({"error": message}), 400


def _skip_tiles(object_name: str, reason: str) -> flask.Response:
    logging.warning("Not publishing tiles for %s: %s", object_name, reason)
    return flask.jsonify({"status": "skipped", "reason": reason})


def _published_reason(published: dict, source: storage.Blob) -> str | None:
    """Says why source's content is already published, or returns None."""
    if source.md5_hash and published.get("source_md5") == source.md5_hash:
        return f"content (MD5 {source.md5_hash}) is already published"
    if published.get("source_generation") == source.generation:
        return f"generation {source.generation} is already published"
    return None


def _read_publish_status(
    bucket: storage.Bucket, tileset: "tile_publish.Tileset"
) -> dict:
    blob = bucket.get_blob(tileset.status_blob)
    if blob is None:
        return {}
    return json.loads(blob.download_as_text())

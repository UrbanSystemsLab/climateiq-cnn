"""Builds PMTiles, uploads them and releases them to tileserver-gl."""

import dataclasses
import datetime
import json
import logging
import os
import shutil
import subprocess
import time

from google.api_core import exceptions
from google.cloud import storage

from postprocessing import pmtiles_build
from postprocessing import tile_publish

TILESERVER_BUCKET_ENV = "TILESERVER_BUCKET"


PMTILES_CONTENT_TYPE = "application/vnd.pmtiles"

BOOTSTRAP_RELEASE = "bootstrap"

_RELEASE_ATTEMPTS = 5


@dataclasses.dataclass(frozen=True)
class PmtilesBuild:
    tileset: tile_publish.Tileset
    path: str
    source_dir: str  # Deleted once the build finishes.
    process: subprocess.Popen


def _layer_file(layer):
    return f"admin_level_{layer.admin_level}_all_cities.geojson"


def _start_pmtiles_build(tileset, source_dir, tiles_dir):
    """Starts building tileset's PMTiles from source_dir's per-level GeoJSON."""
    layers = [
        pmtiles_build.LayerSource(
            layer.layer_name,
            layer.minzoom,
            layer.maxzoom,
            os.path.join(source_dir, _layer_file(layer)),
        )
        for layer in tileset.layers
    ]
    path = os.path.join(tiles_dir, f"{tileset.name}.pmtiles")
    process = pmtiles_build.start(path, tileset.name, tileset.description, layers)
    return PmtilesBuild(tileset, path, source_dir, process)


def _start_city_pmtiles_build(city_folder, tiles_bucket, output_dir, tiles_dir):
    """Starts building city_folder's H3 PMTiles, or returns None if no tileset.

    The build reads hard links to output_dir's files, so they can be deleted
    from output_dir while it runs.
    """
    tileset = tile_publish.tileset_for_object(
        tiles_bucket, tile_publish.CITY_H3_SOURCE.format(area=city_folder)
    )
    if tileset is None:
        logging.warning(
            "%s is not an <ISO3>_<STATE>_<City> name; skipping PMTiles", city_folder
        )
        return None
    source_dir = os.path.join(tiles_dir, tileset.name)
    os.makedirs(source_dir, exist_ok=True)
    for layer in tileset.layers:
        path = os.path.join(output_dir, _layer_file(layer))
        if os.path.exists(path):
            os.link(path, os.path.join(source_dir, _layer_file(layer)))
    return _start_pmtiles_build(tileset, source_dir, tiles_dir)


def _start_national_admin_pmtiles_build(
    pipeline_bucket, output_prefix, tiles_bucket, tiles_dir
):
    """Starts building the national admin PMTiles from the all-cities GeoJSON."""
    tileset = tile_publish.tileset_for_object(
        tiles_bucket, tile_publish.NATIONAL_ADMIN_SOURCE
    )
    source_dir = os.path.join(tiles_dir, tileset.name)
    os.makedirs(source_dir, exist_ok=True)
    bucket = storage.Client().bucket(pipeline_bucket)
    for layer in tileset.layers:
        fname = _layer_file(layer)
        blob = bucket.blob(f"{output_prefix}{fname}")
        if blob.exists():
            blob.download_to_filename(os.path.join(source_dir, fname))
    return _start_pmtiles_build(tileset, source_dir, tiles_dir)


def _upload_pmtiles_build(build, tiles_bucket):
    """Waits for a build and uploads its PMTiles to the tileserver's latest/ prefix.

    Returns:
        The uploaded blob, or None if the tileset had no features.

    Raises:
        RuntimeError: The build failed.
    """
    returncode = build.process.wait()
    shutil.rmtree(build.source_dir, ignore_errors=True)
    if returncode != 0:
        raise RuntimeError(
            f"Building {build.tileset.name} PMTiles exited with {returncode}"
        )
    if not os.path.exists(build.path):
        return None
    blob_name = f"{tile_publish.TILESERVER_LATEST_PREFIX}/{build.tileset.name}.pmtiles"
    blob = storage.Client().bucket(tiles_bucket).blob(blob_name)
    blob.upload_from_filename(build.path, content_type=PMTILES_CONTENT_TYPE)
    os.remove(build.path)
    logging.info("Uploaded gs://%s/%s#%s", tiles_bucket, blob_name, blob.generation)
    return blob


def _freeze_pmtiles(tileset_name, blob):
    """Copies an uploaded PMTiles generation to files/<tileset>/<generation>.pmtiles.

    Returns:
        The copy's path relative to files/.
    """
    path = f"{tileset_name}/{blob.generation}.pmtiles"
    frozen = blob.bucket.blob(f"{tile_publish.TILESERVER_FILES_PREFIX}/{path}")
    frozen.content_type = PMTILES_CONTENT_TYPE
    source = blob.bucket.blob(blob.name, generation=blob.generation)
    token = None
    try:
        while True:
            token, _, _ = frozen.rewrite(source, token=token, if_generation_match=0)
            if token is None:
                break
    except exceptions.PreconditionFailed:
        logging.info("Already frozen: %s", frozen.name)
    return path


def _release_config_blob(bucket, release):
    return bucket.blob(
        f"{tile_publish.TILESERVER_RELEASES_PREFIX}/{release}-config.json"
    )


def _release_pmtiles(bucket, files):
    """Makes a release serving files plus the current release's other tilesets.

    Args:
        bucket: The tiles bucket.
        files: Path relative to files/, by tileset name.

    Returns:
        The new release's name.
    """
    for attempt in range(_RELEASE_ATTEMPTS):
        pointer = bucket.blob(tile_publish.TILESERVER_CURRENT_RELEASE)
        try:
            current = pointer.download_as_text().strip()
            pointer_generation = pointer.generation
        except exceptions.NotFound:
            current, pointer_generation = BOOTSTRAP_RELEASE, 0
        current_config = json.loads(
            _release_config_blob(bucket, current).download_as_text()
        )
        data = dict(current_config.get("data", {}))
        data.update({name: {"pmtiles": path} for name, path in files.items()})
        config = {"options": {"paths": {"pmtiles": "../files"}}, "data": data}

        release = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        try:
            _release_config_blob(bucket, release).upload_from_string(
                json.dumps(config, indent=2, sort_keys=True),
                content_type="application/json",
                if_generation_match=0,
            )
            pointer.upload_from_string(
                f"{release}\n",
                content_type="text/plain",
                if_generation_match=pointer_generation,
            )
        except exceptions.PreconditionFailed:
            logging.info("Release %s lost a race; retrying", release)
            time.sleep(1 + attempt)
            continue
        logging.info("Released %s after %s with %s", release, current, sorted(files))
        return release
    raise RuntimeError(f"No release for {sorted(files)} in {_RELEASE_ATTEMPTS} tries")


def _publish_pmtiles_builds(builds, tiles_bucket):
    """Uploads finished builds and releases them to tileserver-gl.

    Returns:
        The uploaded gs:// URIs, and the new release's name or None.

    Raises:
        RuntimeError: A build failed. The others are still released.
    """
    uris = []
    files = {}
    errors = []
    for build in builds:
        try:
            blob = _upload_pmtiles_build(build, tiles_bucket)
        except RuntimeError as e:
            logging.error("%s", e)
            errors.append(e)
            continue
        if blob is None:
            continue
        uris.append(f"gs://{tiles_bucket}/{blob.name}")
        files[build.tileset.name] = _freeze_pmtiles(build.tileset.name, blob)

    release = None
    if files:
        release = _release_pmtiles(storage.Client().bucket(tiles_bucket), files)
    if errors:
        raise errors[0]
    return uris, release

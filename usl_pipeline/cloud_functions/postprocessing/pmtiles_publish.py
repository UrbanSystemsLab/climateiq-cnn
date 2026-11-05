"""Starts PMTiles builds and uploads them for tileserver-gl."""

import logging
import os

from google.cloud import storage

from postprocessing import pmtiles_build
from postprocessing import tile_publish

TILESERVER_BUCKET_ENV = "TILESERVER_BUCKET"


PMTILES_CONTENT_TYPE = "application/vnd.pmtiles"


def _start_pmtiles_build(tileset, source_dir, tiles_dir):
    """Starts building tileset's PMTiles from source_dir's per-level GeoJSON."""
    layers = [
        pmtiles_build.LayerSource(
            layer.layer_name,
            layer.minzoom,
            layer.maxzoom,
            os.path.join(
                source_dir, f"admin_level_{layer.admin_level}_all_cities.geojson"
            ),
        )
        for layer in tileset.layers
    ]
    path = os.path.join(tiles_dir, f"{tileset.name}.pmtiles")
    process = pmtiles_build.start(path, tileset.name, tileset.description, layers)
    return tileset, path, process


def _start_city_pmtiles_build(city_folder, tiles_bucket, output_dir, tiles_dir):
    """Starts building city_folder's H3 PMTiles, or returns None if no tileset."""
    tileset = tile_publish.tileset_for_object(
        tiles_bucket, tile_publish.CITY_H3_SOURCE.format(area=city_folder)
    )
    if tileset is None:
        logging.warning(
            "%s is not an <ISO3>_<STATE>_<City> name; skipping PMTiles", city_folder
        )
        return None
    return _start_pmtiles_build(tileset, output_dir, tiles_dir)


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
        fname = f"admin_level_{layer.admin_level}_all_cities.geojson"
        blob = bucket.blob(f"{output_prefix}{fname}")
        if blob.exists():
            blob.download_to_filename(os.path.join(source_dir, fname))
    return _start_pmtiles_build(tileset, source_dir, tiles_dir)


def _upload_pmtiles_build(build, tiles_bucket):
    """Waits for a build and uploads its PMTiles to the tileserver's latest/ prefix.

    Returns:
        The gs:// URI, or None if the tileset had no features.

    Raises:
        RuntimeError: The build failed.
    """
    tileset, path, process = build
    if process.wait() != 0:
        raise RuntimeError(
            f"Building {tileset.name} PMTiles exited with {process.returncode}"
        )
    if not os.path.exists(path):
        return None
    blob_name = f"{tile_publish.TILESERVER_LATEST_PREFIX}/{tileset.name}.pmtiles"
    blob = storage.Client().bucket(tiles_bucket).blob(blob_name)
    blob.upload_from_filename(path, content_type=PMTILES_CONTENT_TYPE)
    logging.info("Uploaded gs://%s/%s", tiles_bucket, blob_name)
    return f"gs://{tiles_bucket}/{blob_name}"

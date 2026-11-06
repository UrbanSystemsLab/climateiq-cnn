"""Runs the H3 pipeline for one city, end to end."""

import logging
import os

from google.cloud import storage

from postprocessing.breadcrumbs import (
    _add_breadcrumbs_to_outputs,
    _build_city_breadcrumbs,
    _city_output_folder,
)
from postprocessing.city_config import (
    SCENARIO_MAPPING,
    _build_city_config,
    _build_pipeline_config,
    _derive_admin_fields,
    _parse_city_folder,
)
from postprocessing.cli_preprocess_tiff_to_h3csv import preprocess_city
from postprocessing.cli_unified_pipeline import (
    build_h3_breadcrumb_index,
    process_admin_levels,
    process_h3_tilesets,
)
from postprocessing.combined_outputs import (
    _merge_and_upload_outputs,
    _publish_combined_outputs,
)
from postprocessing.gcs_io import (
    _download_from_gcs,
    _upload_to_gcs,
)
from postprocessing.mongo_sync import _sync_to_mongodb
from postprocessing.pmtiles_publish import (
    TILESERVER_BUCKET_ENV,
    _start_city_pmtiles_build,
    _start_national_admin_pmtiles_build,
    _upload_pmtiles_build,
)
from postprocessing.water_polygons import _ensure_water_polygon

OUTPUT_BUCKET = "climateiq-h3-tiles"


def _run_pipeline(
    config_dict, work_dir, run_mode="full", on_level_done=None, h3_breadcrumbs=None
):
    config_dict["output_dir"] = os.path.join(work_dir, "output")
    os.makedirs(config_dict["output_dir"], exist_ok=True)

    logging.info(
        "Starting pipeline with %d cities, mode=%s",
        len(config_dict["cities"]),
        run_mode,
    )

    admin_outputs = {}
    h3_outputs = {}

    if run_mode in ("full", "admin_only"):
        admin_outputs = process_admin_levels(config_dict, on_level_done=on_level_done)
        logging.info("Admin boundary tilesets created: %d", len(admin_outputs))

    h3_breadcrumb_lookup = None
    if run_mode in ("full", "h3_only"):
        level_10_path = os.path.join(
            config_dict["output_dir"], "admin_level_10_all_cities.geojson"
        )
        h3_breadcrumb_lookup = build_h3_breadcrumb_index(level_10_path)
        if h3_breadcrumb_lookup:
            logging.info("Built H3 spatial breadcrumb index from %s", level_10_path)

        h3_outputs = process_h3_tilesets(
            config_dict,
            max_workers=4,
            on_resolution_done=on_level_done,
            h3_breadcrumbs=h3_breadcrumbs,
            h3_breadcrumb_lookup=h3_breadcrumb_lookup,
        )
        logging.info("H3 tilesets created: %d", len(h3_outputs))

    all_outputs = {**admin_outputs, **h3_outputs}
    total_size_mb = sum(
        os.path.getsize(fp) / (1024 * 1024)
        for fp in all_outputs.values()
        if os.path.exists(fp)
    )
    return {
        "status": "success",
        "tilesets_created": len(all_outputs),
        "total_size_mb": round(total_size_mb, 2),
        "output_files": list(all_outputs.values()),
        "admin_levels": list(admin_outputs.keys()),
        "h3_resolutions": list(h3_outputs.keys()),
    }


def _process_city_from_predictions(
    pipeline_bucket, predictions_bucket, city_folder, work_dir
):
    """Fully automatic: derive everything from the city_folder name."""
    city_name, config_prefix, mosaic_filename = _parse_city_folder(city_folder)

    city_config = _build_city_config(city_name, city_folder, predictions_bucket)
    config_dict = _build_pipeline_config(work_dir, pipeline_bucket, [city_config])

    logging.info(
        "Processing %s (folder=%s, predictions=%s)",
        city_name,
        city_folder,
        predictions_bucket,
    )

    csv_cache_prefix = "h3_csv_cache"
    cache_blob = f"{csv_cache_prefix}/{city_name}_max.csv"
    bbox_blob = f"{csv_cache_prefix}/{city_name}_bbox.txt"
    client = storage.Client()
    bucket_obj = client.bucket(pipeline_bucket)

    if bucket_obj.blob(cache_blob).exists():
        logging.info("Cache hit: downloading %s", cache_blob)
        local_h3_csv = os.path.join(work_dir, f"{city_name}_max.csv")
        _download_from_gcs(pipeline_bucket, cache_blob, local_h3_csv)

        tiff_bbox = None
        if bucket_obj.blob(bbox_blob).exists():
            local_bbox = os.path.join(work_dir, f"{city_name}_bbox.txt")
            _download_from_gcs(pipeline_bucket, bbox_blob, local_bbox)
            with open(local_bbox) as f:
                parts = f.read().strip().split(",")
                tiff_bbox = tuple(float(x) for x in parts)
    else:
        logging.info("Preprocessing tiffs for %s...", city_name)
        local_h3_csv, tiff_bbox = preprocess_city(
            city_config=city_config,
            scenario_mapping=SCENARIO_MAPPING,
            work_dir=work_dir,
            n_workers=4,
            intermediates_bucket=pipeline_bucket,
            intermediates_prefix="intermediates",
        )
        _upload_to_gcs(pipeline_bucket, local_h3_csv, cache_blob)
        if tiff_bbox is not None:
            local_bbox = os.path.join(work_dir, f"{city_name}_bbox.txt")
            with open(local_bbox, "w") as f:
                f.write(",".join(str(x) for x in tiff_bbox))
            _upload_to_gcs(pipeline_bucket, local_bbox, bbox_blob)

    city_config["tiff_bbox"] = tiff_bbox
    city_config["h3_csv"] = local_h3_csv

    _derive_admin_fields(city_config, tiff_bbox, config_dict["admin_boundaries"])

    water_pkl = _ensure_water_polygon(city_name, tiff_bbox, pipeline_bucket, work_dir)
    if water_pkl:
        city_config["water_polygon_path"] = water_pkl

    os.makedirs(config_dict["output_dir"], exist_ok=True)

    output_dir = os.path.join(work_dir, "output")
    uploaded_levels = []
    state_fips = city_config.get("state_fips")
    city_output_dir = _city_output_folder(city_name, state_fips)

    def _on_level_done(level, output_path):
        """Track completed levels; breadcrumbs are added per level."""
        try:
            _add_breadcrumbs_to_outputs(
                os.path.dirname(output_path), config_dict=config_dict
            )
            uploaded_levels.append(str(level if level < 100 else level))
        except Exception as e:
            logging.error("Failed to process level %s: %s", level, e)

    h3_breadcrumbs = {city_name: _build_city_breadcrumbs(city_config)}

    results = _run_pipeline(
        config_dict,
        work_dir,
        run_mode="full",
        on_level_done=_on_level_done,
        h3_breadcrumbs=h3_breadcrumbs,
    )

    tiles_bucket = os.environ.get(TILESERVER_BUCKET_ENV)
    tiles_dir = os.path.join(work_dir, "tiles")
    pmtiles_builds = []
    if tiles_bucket:
        os.makedirs(tiles_dir, exist_ok=True)
        city_build = _start_city_pmtiles_build(
            city_output_dir, tiles_bucket, output_dir, tiles_dir
        )
        if city_build:
            pmtiles_builds.append(city_build)

    out_bucket = os.environ.get("OUTPUT_BUCKET", OUTPUT_BUCKET)

    _merge_and_upload_outputs(
        pipeline_bucket,
        "output/",
        output_dir,
        {city_name},
        output_bucket=out_bucket,
        city_output_dir=city_output_dir,
    )

    if tiles_bucket:
        pmtiles_builds.append(
            _start_national_admin_pmtiles_build(
                pipeline_bucket, "output/", tiles_bucket, tiles_dir
            )
        )

    _publish_combined_outputs(
        output_dir, city_output_dir, out_bucket, pipeline_bucket, state_fips=state_fips
    )

    if local_h3_csv and os.path.exists(local_h3_csv):
        client = storage.Client()
        bucket_obj = client.bucket(out_bucket)
        csv_dest = f"{city_output_dir}/h3_sampled.csv"
        bucket_obj.blob(csv_dest).upload_from_filename(local_h3_csv)
        logging.info("Published H3 CSV: gs://%s/%s", out_bucket, csv_dest)

    results["city"] = city_name
    results["gcs_output"] = f"gs://{pipeline_bucket}/output/"
    results["output_bucket"] = out_bucket
    results["city_output_folder"] = city_output_dir

    _sync_to_mongodb(
        output_dir,
        city_name,
        output_bucket=out_bucket,
        city_output_dir=city_output_dir,
    )

    if tiles_bucket:
        uploaded = [_upload_pmtiles_build(b, tiles_bucket) for b in pmtiles_builds]
        results["pmtiles"] = [uri for uri in uploaded if uri]

    return results

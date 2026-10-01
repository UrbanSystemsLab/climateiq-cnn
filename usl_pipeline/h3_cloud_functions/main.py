import datetime
import functools
import json
import logging
import os
import sys
import tempfile
import traceback
from typing import Any, Dict

import flask
import functions_framework
from google.cloud import error_reporting
from google.cloud import storage

sys.path.insert(0, os.path.dirname(__file__))

from cli_unified_pipeline import (
    load_config,
    process_admin_levels,
    process_h3_tilesets,
)
from cli_add_admin_breadcrumbs import (
    add_parent_fields_point_within,
    add_h3_parents_from_admin10,
    ensure_wgs84,
)
from cli_preprocess_tiff_to_h3csv import preprocess_city

_MAX_RETRY_SECONDS = 60 * 60


def _retry_and_report_errors():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(cloud_event):
            logging.basicConfig(level=logging.INFO)

            if cloud_event.data.get("name", "").startswith(
                "gcloud/tmp/parallel_composite_uploads"
            ):
                logging.debug("Skipping tmp upload file %s", cloud_event.data["name"])
                return

            event_time = datetime.datetime.fromisoformat(
                cloud_event.data["timeCreated"]
            )
            event_age = (
                datetime.datetime.now(datetime.timezone.utc) - event_time
            ).total_seconds()
            if event_age > _MAX_RETRY_SECONDS:
                logging.error(
                    "Dropped event id: %s, source: %s, name: %s after %s seconds",
                    cloud_event["id"],
                    cloud_event["source"],
                    cloud_event.data.get("name"),
                    _MAX_RETRY_SECONDS,
                )
                return

            logging.info(
                "Received event id: %s, source: %s, bucket: %s, name: %s",
                cloud_event["id"],
                cloud_event["source"],
                cloud_event.data.get("bucket"),
                cloud_event.data.get("name"),
            )

            try:
                func(cloud_event)
            except Exception:
                error_reporting.Client().report_exception()
                raise

        return wrapper
    return decorator


def _error_to_response(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception:
            return flask.make_response(traceback.format_exc(limit=10), 500)
    return decorated


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------

def _download_from_gcs(bucket_name, source_blob, dest_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob)
    blob.download_to_filename(dest_path)
    logging.info("Downloaded gs://%s/%s", bucket_name, source_blob)


def _upload_to_gcs(bucket_name, source_path, dest_blob):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(dest_blob)
    blob.upload_from_filename(source_path)
    logging.info("Uploaded to gs://%s/%s", bucket_name, dest_blob)


def _download_directory_from_gcs(bucket_name, prefix, dest_dir):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    os.makedirs(dest_dir, exist_ok=True)
    for blob in blobs:
        if not blob.name.endswith("/"):
            relative_path = blob.name[len(prefix):].lstrip("/")
            dest_path = os.path.join(dest_dir, relative_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            blob.download_to_filename(dest_path)
            logging.info("Downloaded %s", blob.name)


# ---------------------------------------------------------------------------
# Breadcrumbs, title-casing, and post-processing
# ---------------------------------------------------------------------------

_SMALL_WORDS = {"of", "the", "and", "in", "on", "at", "to", "for", "de", "del", "la", "las", "los"}
_NAME_CORRECTIONS = {
    "Dekalb": "DeKalb",
    "Mcdonald": "McDonald",
    "Mchenry": "McHenry",
    "Mclean": "McLean",
    "Mckinley": "McKinley",
    "Desoto": "DeSoto",
    "Dupage": "DuPage",
    "Lasalle": "LaSalle",
}


def _smart_title(s):
    if not isinstance(s, str):
        return s
    words = s.title().split()
    for i in range(1, len(words)):
        if words[i].lower() in _SMALL_WORDS:
            words[i] = words[i].lower()
    result = " ".join(words)
    for wrong, right in _NAME_CORRECTIONS.items():
        result = result.replace(wrong, right)
    return result


def _fix_bbox(v):
    if isinstance(v, str) and v.startswith("["):
        try:
            parts = v.strip("[]").replace(",", " ").split()
            return [float(x) for x in parts if x]
        except ValueError:
            return v
    if hasattr(v, "__iter__") and not isinstance(v, str):
        return list(v)
    return v


def _title_case_gdf(gdf):
    if "name" in gdf.columns:
        gdf["name"] = gdf["name"].apply(_smart_title)
    for col in gdf.columns:
        if col.endswith("_localname"):
            gdf[col] = gdf[col].apply(_smart_title)
    if "place_name" in gdf.columns:
        gdf["place_name"] = gdf["place_name"].apply(_smart_title)
    if "county_name" in gdf.columns:
        gdf["county_name"] = gdf["county_name"].apply(_smart_title)
    if "bbox" in gdf.columns:
        gdf["bbox"] = gdf["bbox"].apply(_fix_bbox)
    return gdf


STATE_FIPS_TO_INFO = {
    "04": {"name": "Arizona", "osm_id": "162018"},
    "06": {"name": "California", "osm_id": "165475"},
    "10": {"name": "Delaware", "osm_id": "162110"},
    "12": {"name": "Florida", "osm_id": "162050"},
    "13": {"name": "Georgia", "osm_id": "161957"},
    "17": {"name": "Illinois", "osm_id": "122586"},
    "22": {"name": "Louisiana", "osm_id": "224922"},
    "24": {"name": "Maryland", "osm_id": "162112"},
    "34": {"name": "New Jersey", "osm_id": "224951"},
    "36": {"name": "New York", "osm_id": "61320"},
    "42": {"name": "Pennsylvania", "osm_id": "162109"},
    "48": {"name": "Texas", "osm_id": "114690"},
}

USA_BOUNDARY_ID = "148838"
USA_NAME = "United States of America"


def _add_breadcrumbs_to_outputs(output_dir, config_dict=None):
    import geopandas as gpd
    import glob as _glob
    import numpy as np
    import pandas as pd
    from shapely.ops import unary_union

    logging.info("Adding breadcrumbs to output files...")

    level_6_path = os.path.join(output_dir, "admin_level_6_all_cities.geojson")
    level_8_path = os.path.join(output_dir, "admin_level_8_all_cities.geojson")
    level_10_path = os.path.join(output_dir, "admin_level_10_all_cities.geojson")

    files_exist = {
        6: os.path.exists(level_6_path),
        8: os.path.exists(level_8_path),
        10: os.path.exists(level_10_path),
    }

    if not any(files_exist.values()):
        logging.info("No admin level files found for breadcrumbs")
        return

    gdfs = {}
    if files_exist[6]:
        gdfs[6] = gpd.read_file(level_6_path)
    if files_exist[8]:
        gdfs[8] = gpd.read_file(level_8_path)
    if files_exist[10]:
        gdfs[10] = gpd.read_file(level_10_path)

    # is_city flag
    if 6 in gdfs:
        gdfs[6]["is_city"] = False
    if 8 in gdfs:
        gdfs[8]["is_city"] = True
    if 10 in gdfs:
        gdfs[10]["is_city"] = False

    # Title-case before joins
    for lvl, gdf in gdfs.items():
        _title_case_gdf(gdf)

    # Country-level breadcrumbs
    for lvl, gdf in gdfs.items():
        gdf["admin_level_2_parent_boundary_id"] = USA_BOUNDARY_ID
        gdf["admin_level_2_parent_localname"] = USA_NAME

    # State-level breadcrumbs
    for lvl, gdf in gdfs.items():
        if "STATEFP" in gdf.columns:
            gdf["admin_level_4_parent_boundary_id"] = gdf["STATEFP"].map(
                lambda fp: STATE_FIPS_TO_INFO.get(str(fp), {}).get("osm_id", str(fp))
            )
            gdf["admin_level_4_parent_localname"] = gdf["STATEFP"].map(
                lambda fp: STATE_FIPS_TO_INFO.get(str(fp), {}).get("name", "")
            )
        elif config_dict:
            city_to_state = {}
            for city_cfg in config_dict.get("cities", []):
                sfips = city_cfg.get("state_fips", "")
                info = STATE_FIPS_TO_INFO.get(sfips, {})
                city_to_state[city_cfg["name"]] = {
                    "osm_id": info.get("osm_id", sfips),
                    "name": info.get("name", ""),
                }
            if "city" in gdf.columns:
                gdf["admin_level_4_parent_boundary_id"] = gdf["city"].map(
                    lambda c: city_to_state.get(c, {}).get("osm_id", "")
                )
                gdf["admin_level_4_parent_localname"] = gdf["city"].map(
                    lambda c: city_to_state.get(c, {}).get("name", "")
                )

    # Parent-child relationships
    if 8 in gdfs and 6 in gdfs:
        gdfs[8] = add_parent_fields_point_within(gdfs[8], gdfs[6], parent_level=6)

    if 10 in gdfs and 8 in gdfs:
        gdfs[10] = add_parent_fields_point_within(gdfs[10], gdfs[8], parent_level=8)
        if "admin_level_8_parent_localname" in gdfs[10].columns:
            gdfs[10]["place_name"] = gdfs[10]["admin_level_8_parent_localname"]

    if 10 in gdfs and 6 in gdfs:
        gdfs[10] = add_parent_fields_point_within(gdfs[10], gdfs[6], parent_level=6)
        if "admin_level_6_parent_localname" in gdfs[10].columns:
            gdfs[10]["county_name"] = gdfs[10]["admin_level_6_parent_localname"]

    # Final title-case pass
    for lvl, gdf in gdfs.items():
        _title_case_gdf(gdf)

    # Save admin level files
    if 8 in gdfs:
        gdfs[8].to_file(level_8_path, driver="GeoJSON")
    if 10 in gdfs:
        gdfs[10].to_file(level_10_path, driver="GeoJSON")
    if 6 in gdfs:
        gdfs[6].to_file(level_6_path, driver="GeoJSON")

    # NYC Level 4 aggregation
    if 6 in gdfs:
        nyc_geoids = {"36061", "36047", "36081", "36005", "36085"}
        gdf6 = gdfs[6]
        id_col = "boundary_id" if "boundary_id" in gdf6.columns else "id"
        nyc_mask = gdf6[id_col].astype(str).isin(nyc_geoids)
        nyc_counties = gdf6[nyc_mask]

        if len(nyc_counties) > 0:
            merged_geom = unary_union(nyc_counties.geometry)
            rec = {
                "admin_level": 4,
                "id": "61320",
                "boundary_id": "61320",
                "name": "New York",
                "is_city": True,
                "admin_level_2_parent_boundary_id": USA_BOUNDARY_ID,
                "admin_level_2_parent_localname": USA_NAME,
                "admin_level_4_parent_boundary_id": "61320",
                "admin_level_4_parent_localname": "New York",
                "city": "NYC",
                "STATEFP": "36",
            }
            numeric_cols = [
                c for c in nyc_counties.columns
                if c not in ("geometry", id_col, "name", "city", "STATEFP",
                             "COUNTYFP", "GEOID", "admin_level", "is_city")
                and nyc_counties[c].dtype in ("float64", "int64", "float32")
            ]
            for col in numeric_cols:
                vals = nyc_counties[col].dropna().values
                rec[col] = round(float(np.mean(vals)), 6) if len(vals) > 0 else 0.0

            level_4_gdf = gpd.GeoDataFrame([rec], geometry=[merged_geom], crs="EPSG:4326")
            level_4_path = os.path.join(output_dir, "admin_level_4_all_cities.geojson")
            level_4_gdf.to_file(level_4_path, driver="GeoJSON")
            logging.info("Created admin_level_4 (NYC aggregation)")

    # H3 tileset breadcrumbs
    if 10 in gdfs:
        city_breadcrumbs = {}
        city_best_score = {}
        for _, row in gdfs[10].iterrows():
            city = row.get("city")
            if not city:
                continue
            score = 0
            candidate = {}
            for lvl in (2, 4, 6, 8):
                for suffix in ("_parent_boundary_id", "_parent_localname"):
                    key = f"admin_level_{lvl}{suffix}"
                    val = row.get(key)
                    if pd.notna(val) and str(val) not in ("<NA>", "<Na>", "None", "nan"):
                        candidate[key] = val
                        score += 1
            if score > city_best_score.get(city, -1):
                city_breadcrumbs[city] = candidate
                city_best_score[city] = score

        h3_pattern = os.path.join(output_dir, "admin_level_1??_all_cities.geojson")
        h3_files = sorted(_glob.glob(h3_pattern))
        for h3_path in h3_files:
            try:
                h3_gdf = gpd.read_file(h3_path)
                h3_gdf = add_h3_parents_from_admin10(h3_gdf, gdfs[10])

                filled = 0
                if "city" in h3_gdf.columns:
                    for lvl in (2, 4, 6, 8):
                        pid_key = f"admin_level_{lvl}_parent_boundary_id"
                        pname_key = f"admin_level_{lvl}_parent_localname"
                        if pname_key not in h3_gdf.columns:
                            h3_gdf[pname_key] = None
                        if pid_key not in h3_gdf.columns:
                            h3_gdf[pid_key] = None
                        is_null = h3_gdf[pname_key].isna() | h3_gdf[pname_key].isin(
                            ["", "<NA>", "<Na>", "None"]
                        )
                        if not is_null.any():
                            continue
                        lookup_pname = {
                            c: v.get(pname_key) for c, v in city_breadcrumbs.items()
                            if v.get(pname_key)
                        }
                        lookup_pid = {
                            c: v.get(pid_key) for c, v in city_breadcrumbs.items()
                            if v.get(pid_key)
                        }
                        fallback_pname = h3_gdf["city"].map(lookup_pname)
                        fallback_pid = h3_gdf["city"].map(lookup_pid)
                        fill_mask = is_null & fallback_pname.notna()
                        cnt = fill_mask.sum()
                        if cnt > 0:
                            h3_gdf.loc[fill_mask, pname_key] = fallback_pname[fill_mask]
                            h3_gdf.loc[fill_mask, pid_key] = fallback_pid[fill_mask]
                            filled += cnt
                    if filled:
                        logging.info("City fallback filled %d null breadcrumbs in %s",
                                     filled, os.path.basename(h3_path))

                _title_case_gdf(h3_gdf)
                h3_gdf = ensure_wgs84(h3_gdf)
                h3_gdf.to_file(h3_path, driver="GeoJSON")
                logging.info("Updated %s with admin breadcrumbs", os.path.basename(h3_path))
            except Exception as e:
                logging.error("Failed to add breadcrumbs to %s: %s",
                              os.path.basename(h3_path), e)

    logging.info("Breadcrumbs added successfully")


# ---------------------------------------------------------------------------
# Merge & upload (keeps other cities' data intact on per-city runs)
# ---------------------------------------------------------------------------

def _merge_and_upload_outputs(bucket_name, output_prefix, output_dir, cities_in_run):
    import glob as _glob
    import re as _re

    output_files = _glob.glob(os.path.join(output_dir, "*.geojson"))
    if not output_files:
        logging.info("No output files to upload")
        return

    def _level_key(path):
        m = _re.search(r"admin_level_(\d+)_", os.path.basename(path))
        return int(m.group(1)) if m else 9999

    output_files = sorted(output_files, key=_level_key)

    client = storage.Client()
    bucket_obj = client.bucket(bucket_name)
    STREAM_THRESHOLD_MB = 50

    for local_path in output_files:
        fname = os.path.basename(local_path)
        dest = f"{output_prefix}{fname}"

        with open(local_path) as f:
            new_fc = json.load(f)
        new_features = new_fc.get("features", [])

        blob = bucket_obj.blob(dest)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False) as tmp_out:
            merged_path = tmp_out.name

        if not blob.exists():
            logging.info("%s: new file, %d features", fname, len(new_features))
            with open(merged_path, "w") as out:
                json.dump({"type": "FeatureCollection", "features": new_features}, out)
        else:
            with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp_dl:
                existing_path = tmp_dl.name
            try:
                blob.download_to_filename(existing_path)
                existing_mb = os.path.getsize(existing_path) / (1024 * 1024)

                if existing_mb > STREAM_THRESHOLD_MB:
                    streamed_ok = False
                    try:
                        import ijson
                        import decimal as _decimal

                        class _FloatEncoder(json.JSONEncoder):
                            def default(self, o):
                                if isinstance(o, _decimal.Decimal):
                                    return float(o)
                                return super().default(o)

                        kept = 0
                        with open(merged_path, "w") as out:
                            out.write('{"type":"FeatureCollection","features":[')
                            first = True
                            with open(existing_path, "rb") as src:
                                for feat in ijson.items(src, "features.item"):
                                    if feat.get("properties", {}).get("city") not in cities_in_run:
                                        if not first:
                                            out.write(",")
                                        out.write(json.dumps(feat, cls=_FloatEncoder))
                                        first = False
                                        kept += 1
                            for feat in new_features:
                                if not first:
                                    out.write(",")
                                json.dump(feat, out)
                                first = False
                            out.write("]}")
                        logging.info("%s: streamed %d existing + %d new features (%.0f MB)",
                                     fname, kept, len(new_features), existing_mb)
                        streamed_ok = True
                    except ImportError:
                        pass

                    if not streamed_ok:
                        with open(existing_path) as f:
                            existing_fc = json.load(f)
                        existing_features = [
                            feat for feat in existing_fc.get("features", [])
                            if feat.get("properties", {}).get("city") not in cities_in_run
                        ]
                        with open(merged_path, "w") as out:
                            json.dump({"type": "FeatureCollection",
                                       "features": existing_features + new_features}, out)
                else:
                    with open(existing_path) as f:
                        existing_fc = json.load(f)
                    existing_features = [
                        feat for feat in existing_fc.get("features", [])
                        if feat.get("properties", {}).get("city") not in cities_in_run
                    ]
                    logging.info("%s: keeping %d existing + %d new features",
                                 fname, len(existing_features), len(new_features))
                    with open(merged_path, "w") as out:
                        json.dump({"type": "FeatureCollection",
                                   "features": existing_features + new_features}, out)
            except Exception as e:
                logging.error("Could not merge %s: %s — skipping upload", fname, e)
                if os.path.exists(existing_path):
                    os.unlink(existing_path)
                if os.path.exists(merged_path):
                    os.unlink(merged_path)
                continue
            finally:
                if os.path.exists(existing_path):
                    os.unlink(existing_path)

        _upload_to_gcs(bucket_name, merged_path, dest)
        if os.path.exists(merged_path):
            os.unlink(merged_path)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline(config_dict, work_dir, run_mode="full"):
    config_dict["output_dir"] = os.path.join(work_dir, "output")
    os.makedirs(config_dict["output_dir"], exist_ok=True)

    logging.info("Starting pipeline with %d cities, mode=%s",
                 len(config_dict["cities"]), run_mode)

    admin_outputs = {}
    h3_outputs = {}

    if run_mode in ("full", "admin_only"):
        admin_outputs = process_admin_levels(config_dict)
        logging.info("Admin boundary tilesets created: %d", len(admin_outputs))

    if run_mode in ("full", "h3_only"):
        h3_outputs = process_h3_tilesets(config_dict, max_workers=4)
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


# ---------------------------------------------------------------------------
# HTTP entry point
# ---------------------------------------------------------------------------

@functions_framework.http
@_error_to_response
def process_h3_pipeline(request: flask.Request) -> flask.Response:
    request_json = request.get_json(silent=True) or {}

    bucket_name = request_json.get("bucket") or os.environ.get("GCS_BUCKET")
    if not bucket_name:
        return flask.jsonify(
            {"error": "GCS_BUCKET environment variable or 'bucket' parameter required"}
        ), 400

    config_path = request_json.get("config_path", "config/cities_config.yaml")
    output_prefix = request_json.get("output_prefix", "output/")
    run_mode = request_json.get("run_mode", "full")
    city_filter = request_json.get("city_filter")
    h3_resolutions = request_json.get("h3_resolutions")

    logging.info("Starting H3 pipeline — bucket=%s mode=%s", bucket_name, run_mode)

    with tempfile.TemporaryDirectory() as work_dir:
        local_config_path = os.path.join(work_dir, "cities_config.yaml")
        _download_from_gcs(bucket_name, config_path, local_config_path)
        config_dict = load_config(local_config_path)

        data_dir = os.path.join(work_dir, "data")
        _download_directory_from_gcs(bucket_name, "admin_boundaries/", data_dir)

        config_dict["admin_boundaries"]["level_6"] = os.path.join(
            data_dir, os.path.basename(config_dict["admin_boundaries"]["level_6"])
        )
        config_dict["admin_boundaries"]["level_8"] = os.path.join(
            data_dir, os.path.basename(config_dict["admin_boundaries"]["level_8"])
        )
        config_dict["admin_boundaries"]["level_10"] = os.path.join(
            data_dir, os.path.basename(config_dict["admin_boundaries"]["level_10"])
        )

        if city_filter:
            original_cities = config_dict["cities"]
            config_dict["cities"] = [c for c in original_cities if c["name"] == city_filter]
            if not config_dict["cities"]:
                return flask.jsonify(
                    {"error": f"city_filter={city_filter!r} not found in config"}
                ), 400
            logging.info("city_filter=%s: processing 1 of %d cities",
                         city_filter, len(original_cities))

        global_scenario_mapping = config_dict.get("scenario_mapping", {})
        csv_cache_prefix = "h3_csv_cache"
        for city in config_dict["cities"]:
            scenario_mapping = city.get("scenario_mapping") or global_scenario_mapping
            if "predictions" in city and scenario_mapping:
                city_name = city["name"]
                cache_blob = f"{csv_cache_prefix}/{city_name}_max.csv"
                bbox_blob = f"{csv_cache_prefix}/{city_name}_bbox.txt"
                client = storage.Client()
                bucket_obj = client.bucket(bucket_name)

                if bucket_obj.blob(cache_blob).exists():
                    logging.info("Cache hit: downloading %s", cache_blob)
                    local_h3_csv = os.path.join(work_dir, f"{city_name}_max.csv")
                    _download_from_gcs(bucket_name, cache_blob, local_h3_csv)

                    tiff_bbox = None
                    if bucket_obj.blob(bbox_blob).exists():
                        local_bbox = os.path.join(work_dir, f"{city_name}_bbox.txt")
                        _download_from_gcs(bucket_name, bbox_blob, local_bbox)
                        with open(local_bbox) as f:
                            parts = f.read().strip().split(",")
                            tiff_bbox = tuple(float(x) for x in parts)
                else:
                    logging.info("Preprocessing tiffs for %s...", city_name)
                    local_h3_csv, tiff_bbox = preprocess_city(
                        city_config=city,
                        scenario_mapping=scenario_mapping,
                        work_dir=work_dir,
                        n_workers=4,
                        intermediates_bucket=bucket_name,
                        intermediates_prefix="intermediates",
                    )
                    _upload_to_gcs(bucket_name, local_h3_csv, cache_blob)
                    if tiff_bbox is not None:
                        local_bbox = os.path.join(work_dir, f"{city_name}_bbox.txt")
                        with open(local_bbox, "w") as f:
                            f.write(",".join(str(x) for x in tiff_bbox))
                        _upload_to_gcs(bucket_name, local_bbox, bbox_blob)

                city["tiff_bbox"] = tiff_bbox
            else:
                h3_csv_gcs = city["h3_csv"]
                local_h3_csv = os.path.join(work_dir, os.path.basename(h3_csv_gcs))
                _download_from_gcs(bucket_name, f"h3_input/{h3_csv_gcs}", local_h3_csv)

            city["h3_csv"] = local_h3_csv

        config_dict["output_dir"] = os.path.join(work_dir, "output")
        os.makedirs(config_dict["output_dir"], exist_ok=True)

        if h3_resolutions:
            config_dict["h3_hex_resolutions"] = h3_resolutions

        if run_mode == "preprocess_only":
            return flask.jsonify({
                "status": "success",
                "mode": "preprocess_only",
                "cities_preprocessed": [c["name"] for c in config_dict["cities"]],
                "gcs_output": f"gs://{bucket_name}/h3_csv_cache/",
            })

        results = _run_pipeline(config_dict, work_dir, run_mode=run_mode)

        output_dir = os.path.join(work_dir, "output")
        _add_breadcrumbs_to_outputs(output_dir, config_dict=config_dict)

        cities_in_run = {c["name"] for c in config_dict["cities"]}
        _merge_and_upload_outputs(bucket_name, output_prefix, output_dir, cities_in_run)

        results["gcs_output"] = f"gs://{bucket_name}/{output_prefix}"
        logging.info("Pipeline completed successfully")

        return flask.jsonify(results)


# ---------------------------------------------------------------------------
# Cloud Storage event trigger entry point
# ---------------------------------------------------------------------------

@functions_framework.cloud_event
@_retry_and_report_errors()
def process_h3_pipeline_on_tiff_upload(
    cloud_event: functions_framework.CloudEvent,
) -> None:
    data = cloud_event.data
    src_bucket_name = data["bucket"]
    file_name = data["name"]

    logging.info("GCS finalize: gs://%s/%s", src_bucket_name, file_name)

    if not file_name.startswith("flood_predictions/"):
        logging.info("Ignoring: not under flood_predictions/")
        return

    parts = file_name.split("/")
    if len(parts) < 4:
        logging.info("Ignoring: path too short")
        return

    city_folder = parts[1]
    mosaic_filename = parts[-1]

    if not mosaic_filename.lower().endswith(".tif"):
        logging.info("Ignoring: not a .tif file")
        return
    if "chunk" in mosaic_filename.lower():
        logging.info("Ignoring: chunk TIF, waiting for mosaic")
        return

    logging.info("Mosaic TIF detected: %s / %s", city_folder, mosaic_filename)

    pipeline_bucket = os.environ.get("GCS_BUCKET")
    if not pipeline_bucket:
        logging.error("GCS_BUCKET environment variable is required")
        return
    config_path = os.environ.get("CONFIG_PATH", "config/cities_config.yaml")

    client = storage.Client()
    pipeline_bucket_obj = client.bucket(pipeline_bucket)

    with tempfile.TemporaryDirectory() as work_dir:
        local_config = os.path.join(work_dir, "cities_config.yaml")
        _download_from_gcs(pipeline_bucket, config_path, local_config)
        config_dict = load_config(local_config)

    city_config = None
    for city in config_dict["cities"]:
        pred = city.get("predictions", {})
        if pred.get("city_folder") == city_folder:
            city_config = city
            break

    if city_config is None:
        logging.info("No city config found for city_folder=%s — ignoring", city_folder)
        return

    city_name = city_config["name"]
    pred = city_config["predictions"]
    config_prefix = pred["config_prefix"]
    expected_mosaic = pred["mosaic_filename"]

    if mosaic_filename != expected_mosaic:
        logging.info("%s != expected %s — ignoring", mosaic_filename, expected_mosaic)
        return

    global_scenario_mapping = config_dict.get("scenario_mapping", {})
    scenario_mapping = city_config.get("scenario_mapping") or global_scenario_mapping
    expected_scenarios = list(scenario_mapping.keys())
    logging.info("%s: expecting %d scenarios", city_name, len(expected_scenarios))

    src_bucket_obj = client.bucket(src_bucket_name)
    present = []
    missing = []
    for scenario_file in expected_scenarios:
        blob_path = (
            f"flood_predictions/{city_folder}/"
            f"{config_prefix}%2F{scenario_file}/{expected_mosaic}"
        )
        if src_bucket_obj.blob(blob_path).exists():
            present.append(scenario_file)
        else:
            missing.append(scenario_file)

    logging.info("%s: %d/%d scenarios ready", city_name, len(present), len(expected_scenarios))
    if missing:
        logging.info("Still waiting for: %s", missing)
        return

    cache_blob = f"h3_csv_cache/{city_name}_max.csv"
    if pipeline_bucket_obj.blob(cache_blob).exists():
        logging.info("%s: CSV cache already exists — skipping", city_name)
        return

    logging.info("%s: All %d scenarios ready — starting pipeline!",
                 city_name, len(expected_scenarios))

    class _MockRequest:
        def get_json(self, silent=False):
            return {
                "bucket": pipeline_bucket,
                "config_path": config_path,
                "output_prefix": "output/",
                "run_mode": "full",
                "city_filter": city_name,
            }

    process_h3_pipeline(_MockRequest())

import datetime
import functools
import json
import logging
import os
import re
import sys
import tempfile
import traceback

import flask
import functions_framework
from google.cloud import error_reporting
from google.cloud import storage

sys.path.insert(0, os.path.dirname(__file__))

from cli_unified_pipeline import (
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

# These are constant across all cities — no config file needed.
SCENARIO_MAPPING = {
    "Rainfall_Data_1.txt": "flood_depth_1y",
    "Rainfall_Data_2.txt": "flood_depth_5y",
    "Rainfall_Data_3.txt": "flood_depth_10y",
    "Rainfall_Data_4.txt": "flood_depth_25y",
    "Rainfall_Data_5.txt": "flood_depth_50y",
    "Rainfall_Data_6.txt": "flood_depth_100y",
    "Rainfall_Data_7.txt": "flood_depth_200y",
    "Rainfall_Data_8.txt": "flood_depth_500y",
    "Rainfall_Data_9.txt": "flood_depth_1000y",
}

VALUE_COLS = list(SCENARIO_MAPPING.values())

H3_HEX_RESOLUTIONS = [6, 7, 8, 9, 10, 11, 12]

ADMIN_BOUNDARY_FILES = {
    "level_6": "counties.parquet",
    "level_8": "places.parquet",
    "level_10": "tracts.parquet",
}

FILTER_THRESHOLD = 0.001

EXPECTED_SCENARIO_COUNT = 9

WATER_POLYGONS_PREFIX = "water_polygons"
JRC_TILE_URL = "https://storage.googleapis.com/global-surface-water/downloads2021/occurrence/occurrence_{tile}v1_4_2021.tif"
JRC_OCCURRENCE_THRESHOLD = 80


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
# Derive city info from the predictions bucket path
# ---------------------------------------------------------------------------

def _parse_city_folder(city_folder):
    """Derive city name and config prefix from folder name.

    Pattern: '{CityName}_Predictions' or '{CityName}_Prediction'
    e.g. 'NYC_Predictions' -> city_name='NYC', config_prefix='NYC'
         'Chicago_Predictions' -> city_name='Chicago', config_prefix='Chicago'
    """
    m = re.match(r"^(.+?)_Predictions?(?:_.*)?$", city_folder)
    if m:
        city_name = m.group(1)
    else:
        city_name = city_folder
    config_prefix = city_name
    mosaic_filename = f"{city_folder}_mosaic_peak_wgs84.tif"
    return city_name, config_prefix, mosaic_filename


def _build_city_config(city_name, city_folder, predictions_bucket):
    """Build a city config dict from just the folder name — no YAML needed."""
    config_prefix = city_name
    mosaic_filename = f"{city_folder}_mosaic_peak_wgs84.tif"
    return {
        "name": city_name,
        "predictions": {
            "bucket": predictions_bucket,
            "city_folder": city_folder,
            "config_prefix": config_prefix,
            "mosaic_filename": mosaic_filename,
        },
    }


def _derive_admin_fields(city_config, tiff_bbox, admin_boundaries):
    """Derive state_fips, county_geoids, and place_name from the TIF bbox.

    Uses spatial intersection against the Census parquet files so no
    per-city config is needed.
    """
    import geopandas as gpd
    from shapely.geometry import box

    if tiff_bbox is None:
        logging.warning("No tiff_bbox — skipping admin field derivation")
        return city_config

    bbox_geom = box(*tiff_bbox)

    counties_path = admin_boundaries.get("level_6")
    if counties_path and os.path.exists(counties_path):
        counties = gpd.read_parquet(counties_path)
        counties = counties.to_crs("EPSG:4326")
        hits = counties[counties.intersects(bbox_geom)]
        if not hits.empty:
            state_fips = hits["STATEFP"].mode().iloc[0]
            county_geoids = sorted(hits["GEOID"].unique().tolist())
            city_config["state_fips"] = state_fips
            city_config["county_geoids"] = county_geoids
            logging.info("Derived state_fips=%s, county_geoids=%s", state_fips, county_geoids)

    places_path = admin_boundaries.get("level_8")
    if places_path and os.path.exists(places_path):
        places = gpd.read_parquet(places_path)
        places = places.to_crs("EPSG:4326")
        hits = places[places.intersects(bbox_geom)]
        if not hits.empty:
            areas = hits.geometry.area
            largest = hits.loc[areas.idxmax()]
            city_config["place_name"] = largest["NAME"]
            logging.info("Derived place_name=%s", city_config["place_name"])

    return city_config


def _build_pipeline_config(work_dir, pipeline_bucket, cities):
    """Build the full pipeline config dict without any YAML file."""
    data_dir = os.path.join(work_dir, "data")
    _download_directory_from_gcs(pipeline_bucket, "admin_boundaries/", data_dir)

    return {
        "output_dir": os.path.join(work_dir, "output"),
        "h3_col": "cell_code",
        "value_cols": VALUE_COLS,
        "filter_threshold": FILTER_THRESHOLD,
        "h3_hex_resolutions": H3_HEX_RESOLUTIONS,
        "scenario_mapping": SCENARIO_MAPPING,
        "admin_boundaries": {
            "level_6": os.path.join(data_dir, ADMIN_BOUNDARY_FILES["level_6"]),
            "level_8": os.path.join(data_dir, ADMIN_BOUNDARY_FILES["level_8"]),
            "level_10": os.path.join(data_dir, ADMIN_BOUNDARY_FILES["level_10"]),
        },
        "cities": cities,
    }


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

    if 6 in gdfs:
        gdfs[6]["is_city"] = False
    if 8 in gdfs:
        gdfs[8]["is_city"] = True
    if 10 in gdfs:
        gdfs[10]["is_city"] = False

    for lvl, gdf in gdfs.items():
        _title_case_gdf(gdf)

    for lvl, gdf in gdfs.items():
        gdf["admin_level_2_parent_boundary_id"] = USA_BOUNDARY_ID
        gdf["admin_level_2_parent_localname"] = USA_NAME

    for lvl, gdf in gdfs.items():
        if "STATEFP" in gdf.columns:
            gdf["admin_level_4_parent_boundary_id"] = gdf["STATEFP"].map(
                lambda fp: STATE_FIPS_TO_INFO.get(str(fp), {}).get("osm_id", str(fp))
            )
            gdf["admin_level_4_parent_localname"] = gdf["STATEFP"].map(
                lambda fp: STATE_FIPS_TO_INFO.get(str(fp), {}).get("name", "")
            )

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

    for lvl, gdf in gdfs.items():
        _title_case_gdf(gdf)

    if 8 in gdfs:
        gdfs[8].to_file(level_8_path, driver="GeoJSON")
    if 10 in gdfs:
        gdfs[10].to_file(level_10_path, driver="GeoJSON")
    if 6 in gdfs:
        gdfs[6].to_file(level_6_path, driver="GeoJSON")

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
# Upload per-city + rebuild all_cities from per-city parts
# ---------------------------------------------------------------------------

KNOWN_CITIES = {
    "Atlanta", "Chicago", "LosAngeles", "Miami", "NYC",
    "NewOrleans", "Philadelphia", "Phoenix", "Pittsburgh",
    "SanAntonio", "SanDiego",
}


def _merge_and_upload_outputs(bucket_name, output_prefix, output_dir, cities_in_run):
    """Save per-city file, then rebuild all_cities only when safe.

    1. Upload this city's GeoJSONs to output/by_city/{city}/
    2. Check if all known cities have per-city files. Only rebuild all_cities
       when every city is represented — otherwise skip rebuild to avoid
       overwriting the existing all_cities with incomplete data.
    """
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
    city_name = next(iter(cities_in_run))

    for local_path in output_files:
        fname = os.path.basename(local_path)
        per_city_dest = f"{output_prefix}by_city/{city_name}/{fname}"
        _upload_to_gcs(bucket_name, local_path, per_city_dest)
        logging.info("Uploaded per-city: %s", per_city_dest)

    per_city_prefix = f"{output_prefix}by_city/"
    city_dirs = set()
    for b in bucket_obj.list_blobs(prefix=per_city_prefix):
        parts = b.name[len(per_city_prefix):].split("/")
        if len(parts) >= 2:
            city_dirs.add(parts[0])
    city_dirs = sorted(city_dirs)

    missing = KNOWN_CITIES - set(city_dirs)
    if missing:
        logging.info(
            "Skipping all_cities rebuild: %d/%d known cities missing per-city files: %s. "
            "Per-city file for %s uploaded successfully.",
            len(missing), len(KNOWN_CITIES), sorted(missing), city_name,
        )
        return

    logging.info("All %d known cities have per-city files — rebuilding all_cities from: %s",
                 len(city_dirs), city_dirs)

    for local_path in output_files:
        fname = os.path.basename(local_path)
        all_cities_dest = f"{output_prefix}{fname}"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".geojson", delete=False) as tmp_out:
            merged_path = tmp_out.name

        try:
            total = 0
            with open(merged_path, "w") as out:
                out.write('{"type":"FeatureCollection","features":[')
                first = True
                for city_dir in city_dirs:
                    blob_path = f"{output_prefix}by_city/{city_dir}/{fname}"
                    city_blob = bucket_obj.blob(blob_path)
                    if not city_blob.exists():
                        continue
                    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp_city:
                        city_local = tmp_city.name
                    try:
                        city_blob.download_to_filename(city_local)
                        with open(city_local) as cf:
                            city_fc = json.load(cf)
                        count = 0
                        for feat in city_fc.get("features", []):
                            if not first:
                                out.write(",")
                            json.dump(feat, out)
                            first = False
                            count += 1
                        total += count
                        logging.info("  %s: %d features from %s", fname, count, city_dir)
                    finally:
                        if os.path.exists(city_local):
                            os.unlink(city_local)
                out.write("]}")

            _upload_to_gcs(bucket_name, merged_path, all_cities_dest)
            logging.info("Rebuilt %s: %d total features", fname, total)
        except Exception as e:
            logging.error("Failed to rebuild %s: %s", fname, e)
        finally:
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
# JRC water polygon: auto-generate at pipeline time
# ---------------------------------------------------------------------------

def _get_jrc_tile_name(left_lon, top_lat):
    lon_dir = "E" if left_lon >= 0 else "W"
    lat_dir = "N" if top_lat >= 0 else "S"
    return f"{int(abs(left_lon))}{lon_dir}_{int(abs(top_lat))}{lat_dir}"


def _get_jrc_tiles_for_bbox(bbox):
    import math
    min_lon, min_lat, max_lon, max_lat = bbox
    start_lon = int(math.floor(min_lon / 10) * 10)
    end_lon = int(math.floor(max_lon / 10) * 10)
    start_top = int(math.ceil(min_lat / 10) * 10)
    end_top = int(math.ceil(max_lat / 10) * 10)
    tiles = []
    for left in range(start_lon, end_lon + 10, 10):
        for top in range(start_top, end_top + 10, 10):
            tiles.append(_get_jrc_tile_name(left, top))
    return tiles


def _generate_water_polygon(tiff_bbox, work_dir):
    import pickle
    import requests
    import rasterio
    from rasterio.features import shapes as rasterio_shapes
    from rasterio.mask import mask as rasterio_mask
    from rasterio.merge import merge as rasterio_merge
    import numpy as np
    from shapely.geometry import shape, box, MultiPolygon, Polygon
    from shapely.ops import unary_union
    from shapely.validation import make_valid

    buffer = 0.05
    bbox = (tiff_bbox[0] - buffer, tiff_bbox[1] - buffer,
            tiff_bbox[2] + buffer, tiff_bbox[3] + buffer)

    tiles = _get_jrc_tiles_for_bbox(bbox)
    logging.info("JRC tiles needed: %s", tiles)

    tile_paths = []
    for tile_name in tiles:
        url = JRC_TILE_URL.format(tile=tile_name)
        local_path = os.path.join(work_dir, f"jrc_{tile_name}.tif")
        logging.info("Downloading JRC tile %s...", tile_name)
        resp = requests.get(url, stream=True, timeout=300)
        if resp.status_code == 404:
            logging.info("JRC tile %s not found (no data)", tile_name)
            continue
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        tile_paths.append(local_path)
        logging.info("Downloaded JRC tile %s (%.1f MB)",
                     tile_name, os.path.getsize(local_path) / 1e6)

    if not tile_paths:
        logging.warning("No JRC tiles found — skipping water polygon")
        return None

    datasets = [rasterio.open(p) for p in tile_paths]
    if len(datasets) == 1:
        merged_data = datasets[0].read(1)
        merged_transform = datasets[0].transform
        merged_crs = datasets[0].crs
        merged_shape = merged_data.shape
    else:
        merged_data, merged_transform = rasterio_merge(datasets)
        merged_data = merged_data[0]
        merged_crs = datasets[0].crs
        merged_shape = merged_data.shape
    for ds in datasets:
        ds.close()

    bbox_geom = box(*bbox)
    merged_path = os.path.join(work_dir, "jrc_merged.tif")
    with rasterio.open(
        merged_path, "w", driver="GTiff",
        height=merged_shape[0], width=merged_shape[1],
        count=1, dtype=merged_data.dtype,
        crs=merged_crs, transform=merged_transform,
    ) as dst:
        dst.write(merged_data, 1)
    del merged_data

    with rasterio.open(merged_path) as src:
        cropped, crop_transform = rasterio_mask(src, [bbox_geom], crop=True, nodata=0)
        cropped = cropped[0]
    os.unlink(merged_path)

    water_mask = (cropped >= JRC_OCCURRENCE_THRESHOLD).astype(np.uint8)
    water_count = int(water_mask.sum())
    logging.info("JRC water pixels (>=%d%% occurrence): %d", JRC_OCCURRENCE_THRESHOLD, water_count)
    del cropped

    if water_count == 0:
        logging.info("No permanent water found — skipping water polygon")
        return None

    polygons = []
    for geom_dict, value in rasterio_shapes(water_mask, transform=crop_transform):
        if value == 1:
            poly = shape(geom_dict)
            if poly.is_valid and not poly.is_empty:
                polygons.append(poly)
    del water_mask

    if not polygons:
        return None

    logging.info("Merging %d water polygons...", len(polygons))
    water_union = unary_union(polygons)
    water_union = make_valid(water_union)
    del polygons

    if water_union.geom_type == "Polygon":
        water_union = MultiPolygon([water_union])
    elif water_union.geom_type == "GeometryCollection":
        poly_parts = [g for g in water_union.geoms
                      if isinstance(g, (Polygon, MultiPolygon))]
        water_union = unary_union(poly_parts) if poly_parts else None
        if water_union and water_union.geom_type == "Polygon":
            water_union = MultiPolygon([water_union])

    if water_union is None or water_union.is_empty:
        return None

    for p in tile_paths:
        if os.path.exists(p):
            os.unlink(p)

    logging.info("Water polygon: %s, bounds=%s", water_union.geom_type, water_union.bounds)
    return water_union


def _ensure_water_polygon(city_name, tiff_bbox, pipeline_bucket, work_dir):
    import pickle

    pkl_name = f"{city_name}_coastal_water.pkl"
    gcs_path = f"{WATER_POLYGONS_PREFIX}/{pkl_name}"
    local_pkl = os.path.join(work_dir, pkl_name)

    client = storage.Client()
    bucket_obj = client.bucket(pipeline_bucket)
    blob = bucket_obj.blob(gcs_path)

    if blob.exists():
        logging.info("Water polygon cache hit: %s", gcs_path)
        blob.download_to_filename(local_pkl)
        return local_pkl

    if tiff_bbox is None:
        logging.warning("No tiff_bbox — cannot generate water polygon")
        return None

    logging.info("Generating JRC water polygon for %s...", city_name)
    water_poly = _generate_water_polygon(tiff_bbox, work_dir)
    if water_poly is None:
        return None

    with open(local_pkl, "wb") as f:
        pickle.dump(water_poly, f)

    blob.upload_from_filename(local_pkl)
    logging.info("Uploaded water polygon to gs://%s/%s", pipeline_bucket, gcs_path)
    return local_pkl


# ---------------------------------------------------------------------------
# Core processing: takes a city_folder name and runs end to end
# ---------------------------------------------------------------------------

def _process_city_from_predictions(pipeline_bucket, predictions_bucket, city_folder, work_dir):
    """Fully automatic: derive everything from the city_folder name."""
    city_name, config_prefix, mosaic_filename = _parse_city_folder(city_folder)

    city_config = _build_city_config(city_name, city_folder, predictions_bucket)
    config_dict = _build_pipeline_config(work_dir, pipeline_bucket, [city_config])

    logging.info("Processing %s (folder=%s, predictions=%s)",
                 city_name, city_folder, predictions_bucket)

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

    results = _run_pipeline(config_dict, work_dir, run_mode="full")

    output_dir = os.path.join(work_dir, "output")
    _add_breadcrumbs_to_outputs(output_dir, config_dict=config_dict)

    _merge_and_upload_outputs(pipeline_bucket, "output/", output_dir, {city_name})

    results["city"] = city_name
    results["gcs_output"] = f"gs://{pipeline_bucket}/output/"
    return results


# ---------------------------------------------------------------------------
# HTTP entry point
# ---------------------------------------------------------------------------

@functions_framework.http
@_error_to_response
def process_h3_pipeline(request: flask.Request) -> flask.Response:
    """Process a city. Requires 'city_folder' and 'predictions_bucket' in JSON body.

    Example: {"city_folder": "NYC_Predictions", "predictions_bucket": "test-climateiq-predictions"}
    """
    request_json = request.get_json(silent=True) or {}

    pipeline_bucket = request_json.get("bucket") or os.environ.get("GCS_BUCKET")
    if not pipeline_bucket:
        return flask.jsonify(
            {"error": "GCS_BUCKET environment variable or 'bucket' parameter required"}
        ), 400

    city_folder = request_json.get("city_folder")
    predictions_bucket = request_json.get("predictions_bucket")

    if not city_folder or not predictions_bucket:
        return flask.jsonify(
            {"error": "'city_folder' and 'predictions_bucket' are required"}
        ), 400

    logging.info("Starting H3 pipeline — city_folder=%s predictions=%s",
                 city_folder, predictions_bucket)

    with tempfile.TemporaryDirectory() as work_dir:
        results = _process_city_from_predictions(
            pipeline_bucket, predictions_bucket, city_folder, work_dir
        )
        logging.info("Pipeline completed successfully for %s", results.get("city"))
        return flask.jsonify(results)


# ---------------------------------------------------------------------------
# Cloud Storage event trigger entry point
# ---------------------------------------------------------------------------

@functions_framework.cloud_event
@_retry_and_report_errors()
def process_h3_pipeline_on_tiff_upload(
    cloud_event: functions_framework.CloudEvent,
) -> None:
    """Auto-triggers when a mosaic TIF is uploaded to the predictions bucket.

    Detects city from the path, checks if all 9 scenarios are present,
    and kicks off the full pipeline. No config file needed.
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

    logging.info("%s: %d/%d scenarios ready", city_name, len(present), EXPECTED_SCENARIO_COUNT)
    if missing:
        logging.info("Still waiting for: %s", missing)
        return

    pipeline_bucket_obj = client.bucket(pipeline_bucket)
    cache_blob = f"h3_csv_cache/{city_name}_max.csv"
    if pipeline_bucket_obj.blob(cache_blob).exists():
        logging.info("%s: CSV cache already exists — skipping", city_name)
        return

    logging.info("%s: All %d scenarios ready — starting pipeline!", city_name, EXPECTED_SCENARIO_COUNT)

    with tempfile.TemporaryDirectory() as work_dir:
        results = _process_city_from_predictions(
            pipeline_bucket, src_bucket_name, city_folder, work_dir
        )
        logging.info("Pipeline result for %s: %s", city_name, results.get("status"))

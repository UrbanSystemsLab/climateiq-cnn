"""City settings derived from the predictions bucket path."""

import logging
import os
import re

from postprocessing.gcs_io import _download_directory_from_gcs

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
            logging.info(
                "Derived state_fips=%s, county_geoids=%s", state_fips, county_geoids
            )

    places_path = admin_boundaries.get("level_8")
    if places_path and os.path.exists(places_path):
        places = gpd.read_parquet(places_path)
        places = places.to_crs("EPSG:4326")
        hits = places[places.intersects(bbox_geom)]
        if not hits.empty:
            areas = hits.geometry.area
            largest = hits.loc[areas.idxmax()]
            city_config["place_name"] = largest["NAME"]
            if "STATEFP" in largest.index:
                city_config["state_fips"] = largest["STATEFP"]
                logging.info(
                    "Derived state_fips=%s from place %s",
                    largest["STATEFP"],
                    largest["NAME"],
                )
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

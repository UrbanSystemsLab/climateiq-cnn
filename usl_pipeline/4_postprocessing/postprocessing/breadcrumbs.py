"""Breadcrumbs, title-casing and output folder names."""

import logging
import os

from postprocessing.cli_add_admin_breadcrumbs import add_parent_fields_point_within

_SMALL_WORDS = {
    "of",
    "the",
    "and",
    "in",
    "on",
    "at",
    "to",
    "for",
    "de",
    "del",
    "la",
    "las",
    "los",
}


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
    "04": {"name": "Arizona", "osm_id": "162018", "abbr": "AZ"},
    "06": {"name": "California", "osm_id": "165475", "abbr": "CA"},
    "10": {"name": "Delaware", "osm_id": "162110", "abbr": "DE"},
    "12": {"name": "Florida", "osm_id": "162050", "abbr": "FL"},
    "13": {"name": "Georgia", "osm_id": "161957", "abbr": "GA"},
    "17": {"name": "Illinois", "osm_id": "122586", "abbr": "IL"},
    "22": {"name": "Louisiana", "osm_id": "224922", "abbr": "LA"},
    "24": {"name": "Maryland", "osm_id": "162112", "abbr": "MD"},
    "34": {"name": "New Jersey", "osm_id": "224951", "abbr": "NJ"},
    "36": {"name": "New York", "osm_id": "61320", "abbr": "NY"},
    "42": {"name": "Pennsylvania", "osm_id": "162109", "abbr": "PA"},
    "48": {"name": "Texas", "osm_id": "114690", "abbr": "TX"},
}


def _city_output_folder(city_name, state_fips=None):
    """Build output folder name: USA_{STATE}_{CityPascalCase}.

    Falls back to USA_{CityName} if state_fips is unknown.
    """
    pascal = city_name.replace(" ", "")
    if state_fips:
        abbr = STATE_FIPS_TO_INFO.get(str(state_fips), {}).get("abbr")
        if abbr:
            return f"USA_{abbr}_{pascal}"
    return f"USA_{pascal}"


USA_BOUNDARY_ID = "148838"


USA_NAME = "United States of America"


def _build_city_breadcrumbs(city_config):
    """Build breadcrumb dict for a city from its config.

    Returns dict with admin_level_N_parent_boundary_id/localname keys.
    """
    crumbs = {
        "admin_level_2_parent_boundary_id": USA_BOUNDARY_ID,
        "admin_level_2_parent_localname": USA_NAME,
    }
    state_fips = city_config.get("state_fips")
    if state_fips:
        info = STATE_FIPS_TO_INFO.get(str(state_fips), {})
        crumbs["admin_level_4_parent_boundary_id"] = info.get("osm_id", str(state_fips))
        crumbs["admin_level_4_parent_localname"] = info.get("name", "")
    place_name = city_config.get("place_name")
    if place_name:
        crumbs["admin_level_8_parent_localname"] = place_name
    return crumbs


def _add_breadcrumbs_to_outputs(output_dir, config_dict=None):
    import geopandas as gpd
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
        place_name = None
        if config_dict and "cities" in config_dict:
            for c in config_dict["cities"]:
                pn = c.get("place_name")
                if pn:
                    place_name = pn.lower()
                    break
        if place_name and "name" in gdfs[8].columns:
            names = gdfs[8]["name"].str.lower().str.strip()
            gdfs[8]["is_city"] = names == place_name.lower().strip()
        else:
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
            col8 = gdfs[10]["admin_level_8_parent_localname"]
            gdfs[10]["admin_level_8_parent_localname"] = col8.where(
                ~(col8.isna() | (col8.astype(str) == "<NA>")), ""
            )
            gdfs[10]["place_name"] = gdfs[10]["admin_level_8_parent_localname"]
        if "admin_level_8_parent_boundary_id" in gdfs[10].columns:
            col8id = gdfs[10]["admin_level_8_parent_boundary_id"]
            gdfs[10]["admin_level_8_parent_boundary_id"] = col8id.where(
                ~(col8id.isna() | (col8id.astype(str) == "<NA>")), ""
            )

    if 10 in gdfs and 6 in gdfs:
        gdfs[10] = add_parent_fields_point_within(gdfs[10], gdfs[6], parent_level=6)
        # Fallback: fill <NA> county breadcrumbs from FIPS codes
        gdf10 = gdfs[10]
        county_col = "admin_level_6_parent_localname"
        county_id_col = "admin_level_6_parent_boundary_id"
        has_fips = (
            county_col in gdf10.columns
            and "STATEFP" in gdf10.columns
            and "COUNTYFP" in gdf10.columns
        )
        if has_fips:
            missing = gdf10[county_col].isna() | (
                gdf10[county_col].astype(str) == "<NA>"
            )
            if missing.any() and 6 in gdfs:
                county_lookup = (
                    dict(zip(gdfs[6]["GEOID"].astype(str), gdfs[6]["name"].astype(str)))
                    if "GEOID" in gdfs[6].columns and "name" in gdfs[6].columns
                    else {}
                )
                for idx in gdf10.index[missing]:
                    sfp = str(gdf10.at[idx, "STATEFP"])
                    cfp = str(gdf10.at[idx, "COUNTYFP"])
                    geoid = sfp + cfp
                    if geoid in county_lookup:
                        gdf10.at[idx, county_id_col] = geoid
                        gdf10.at[idx, county_col] = county_lookup[geoid]
        if county_col in gdf10.columns:
            gdf10["county_name"] = gdf10[county_col]

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
                c
                for c in nyc_counties.columns
                if c
                not in (
                    "geometry",
                    id_col,
                    "name",
                    "city",
                    "STATEFP",
                    "COUNTYFP",
                    "GEOID",
                    "admin_level",
                    "is_city",
                )
                and nyc_counties[c].dtype in ("float64", "int64", "float32")
            ]
            for col in numeric_cols:
                vals = nyc_counties[col].dropna().values
                rec[col] = round(float(np.mean(vals)), 6) if len(vals) > 0 else 0.0

            level_4_gdf = gpd.GeoDataFrame(
                [rec], geometry=[merged_geom], crs="EPSG:4326"
            )
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
                    if pd.notna(val) and str(val) not in (
                        "<NA>",
                        "<Na>",
                        "None",
                        "nan",
                    ):
                        candidate[key] = val
                        score += 1
            if score > city_best_score.get(city, -1):
                city_breadcrumbs[city] = candidate
                city_best_score[city] = score

        # H3 hex files get breadcrumbs injected at write time via
        # h3_breadcrumbs passed to process_h3_tilesets — no need to
        # re-read multi-GB files here.

    logging.info("Breadcrumbs added successfully")

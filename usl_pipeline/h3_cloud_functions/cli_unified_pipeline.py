#!/usr/bin/env python3
"""
Unified multi-city H3 pipeline.

Processes multiple cities in a single run and combines outputs into unified tilesets.
This solves the scalability issue of running separate commands for each city.

Usage:
    python cli_unified_pipeline.py --config cities_config.yaml
    python cli_unified_pipeline.py --config cities_config.yaml --skip-h3-tiles
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import geopandas as gpd
import pandas as pd
import yaml
import h3

from usl_h3_tools.admin_enrich import enrich_admin


# Configuration for admin levels
ADMIN_LEVELS = [
    {"level": 6, "key": "level_6", "geoid_col": "GEOID"},
    {"level": 8, "key": "level_8", "geoid_col": "GEOID"},
    {"level": 10, "key": "level_10", "geoid_col": "GEOID"},
]


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate YAML configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required = ["cities", "admin_boundaries", "output_dir"]
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")

    if not config["cities"]:
        raise ValueError("No cities defined in config")

    return config


def build_filter_expr(city: Dict[str, Any], level: int) -> str:
    """Build pandas query filter for a specific city and admin level.
    Supports both single county_geoid and multi-county county_geoids list.
    """
    state = city["state_fips"]

    # Normalise: accept either county_geoid (str) or county_geoids (list)
    county_geoids: List[str] = city.get("county_geoids") or (
        [city["county_geoid"]] if "county_geoid" in city else []
    )

    if level == 6:
        if len(county_geoids) == 1:
            return f"GEOID == '{county_geoids[0]}'"
        geoids_str = ", ".join(f"'{g}'" for g in county_geoids)
        return f"GEOID in [{geoids_str}]"
    elif level == 8:
        place_name = city["place_name"]
        return f"STATEFP == '{state}' and NAME == '{place_name}'"
    elif level == 10:
        # US FIPS: 2-digit numeric state code ("04", "36"). International: alpha ("ZA").
        is_us = state.isdigit() and len(state) == 2
        if is_us:
            # Census tract level: filter by state and county code(s) (last 3 digits of GEOID)
            county_codes = [g[-3:] for g in county_geoids]
            if len(county_codes) == 1:
                return f"STATEFP == '{state}' and COUNTYFP == '{county_codes[0]}'"
            codes_str = ", ".join(f"'{c}'" for c in county_codes)
            return f"STATEFP == '{state}' and COUNTYFP in [{codes_str}]"
        else:
            # International city: filter by STATEFP only (OSM wards all share the same state code)
            return f"STATEFP == '{state}'"
    else:
        raise ValueError(f"Unsupported admin level: {level}")


def process_single_city_level(
    city: Dict[str, Any],
    admin_path: str,
    level: int,
    h3_col: str,
    value_cols: Optional[List[str]],
    geoid_col: str,
    temp_dir: str,
) -> str:
    """Process one city at one admin level, return path to temp output file."""
    city_name = city["name"]
    h3_csv = city["h3_csv"]

    if not os.path.exists(h3_csv):
        print(f"[unified] WARNING: H3 CSV not found for {city_name}: {h3_csv}")
        return None

    # Skip admin level processing when required config is missing
    county_geoids = city.get("county_geoids") or ([city["county_geoid"]] if "county_geoid" in city else [])
    place_name = city.get("place_name", "")
    if not county_geoids and not place_name:
        print(f"[unified] Skipping admin level {level} for {city_name} — no boundary config (international city)")
        return None
    if level == 8 and not place_name:
        print(f"[unified] Skipping admin level 8 for {city_name} — no place_name configured")
        return None

    filter_expr = build_filter_expr(city, level)
    output_name = f"temp_{city_name}_level_{level}.geojson"
    output_path = os.path.join(temp_dir, output_name)

    # Use tiff bbox (set by preprocessing) to clip admin polygons to prediction extent
    clip_bbox = city.get("tiff_bbox")

    # JRC water polygon for coastline clipping (replaces H3 footprint clip)
    water_polygon_path = city.get("water_polygon_path")

    print(f"[unified] Processing {city_name} - Admin Level {level}")
    print(f"[unified]   Filter: {filter_expr}")

    try:
        enrich_admin(
            admin_path=admin_path,
            h3_csv=h3_csv,
            value_cols=value_cols,
            out_path=output_path,
            name=f"temp_{city_name}_admin_level_{level}",
            h3_col=h3_col,
            level=level,
            filter_expr=filter_expr,
            geoid_col=geoid_col,
            clip_bbox=clip_bbox,
            h3_resolution=9,  # Use res 9 only — avoids 4x over-counting from multi-res CSV
            water_polygon_path=water_polygon_path,
        )
        # Inject city field so merge_and_upload_outputs can filter by city
        import json as _json
        with open(output_path) as _f:
            _fc = _json.load(_f)
        for _feat in _fc.get("features", []):
            _feat["properties"]["city"] = city_name
        with open(output_path, "w") as _f:
            _json.dump(_fc, _f)
        print(f"[unified]   ✓ Wrote {output_path}")
        return output_path
    except Exception as e:
        print(f"[unified]   ✗ Error processing {city_name} level {level}: {e}")
        return None


def combine_geojson_files(file_paths: List[str], output_path: str, admin_level: int) -> None:
    """Combine multiple GeoJSON files into a single unified tileset."""
    import shutil

    file_paths = [f for f in file_paths if f is not None and os.path.exists(f)]

    if not file_paths:
        print(f"[unified] No valid files to combine for admin level {admin_level}")
        return

    print(f"[unified] Combining {len(file_paths)} files into {output_path}")

    # Fast path: single file — just move it, no need to load through geopandas
    if len(file_paths) == 1:
        shutil.copy2(file_paths[0], output_path)
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[unified]   ✓ Combined tileset: {file_size_mb:.2f} MB")
        return

    # Multi-city path: load and concatenate via geopandas
    gdfs = []
    for fp in file_paths:
        try:
            gdf = gpd.read_file(fp)
            if not gdf.empty:
                gdfs.append(gdf)
        except Exception as e:
            print(f"[unified]   Warning: Could not read {fp}: {e}")

    if not gdfs:
        print(f"[unified]   No valid data to combine")
        return

    combined = pd.concat(gdfs, ignore_index=True)

    if combined.crs is None:
        combined = combined.set_crs("EPSG:4326")
    else:
        combined = combined.to_crs("EPSG:4326")

    combined.to_file(output_path, driver="GeoJSON")

    feature_count = len(combined)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[unified]   ✓ Combined tileset: {feature_count} features, {file_size_mb:.2f} MB")


def process_admin_levels(config: Dict[str, Any]) -> Dict[int, str]:
    """Process all admin levels for all cities and combine into unified tilesets."""
    output_dir = config["output_dir"]
    temp_dir = os.path.join(output_dir, "_temp")
    os.makedirs(temp_dir, exist_ok=True)

    h3_col = config.get("h3_col", "h3")
    value_cols = config.get("value_cols", None)
    cities = config["cities"]
    admin_boundaries = config["admin_boundaries"]

    combined_outputs = {}

    # Process each admin level
    for level_config in ADMIN_LEVELS:
        level = level_config["level"]
        admin_path = admin_boundaries[level_config["key"]]
        geoid_col = level_config["geoid_col"]

        print(f"\n[unified] === Processing Admin Level {level} ===")

        # Process all cities for this level
        temp_files = []
        for city in cities:
            temp_file = process_single_city_level(
                city=city,
                admin_path=admin_path,
                level=level,
                h3_col=h3_col,
                value_cols=value_cols,
                geoid_col=geoid_col,
                temp_dir=temp_dir,
            )
            if temp_file:
                temp_files.append(temp_file)

        # Combine all cities into single tileset
        combined_output = os.path.join(output_dir, f"admin_level_{level}_all_cities.geojson")
        combine_geojson_files(temp_files, combined_output, level)
        combined_outputs[level] = combined_output

    return combined_outputs


def process_h3_tile_for_city(args_tuple):
    """Process H3 tileset for one city at one resolution (for parallel execution)."""
    city, resolution, h3_col, value_cols, temp_dir = args_tuple

    city_name = city["name"]
    h3_csv = city["h3_csv"]

    if not os.path.exists(h3_csv):
        return None

    try:
        # Load CSV and filter to target resolution
        df = pd.read_csv(h3_csv)

        # Determine H3 resolution
        if "h3_res" in df.columns:
            df["h3_res"] = df["h3_res"].astype(int)
        else:
            df["h3_res"] = df[h3_col].apply(h3.get_resolution)

        # Filter to target resolution
        sub = df[df["h3_res"] == resolution].copy()

        if sub.empty:
            return None

        admin_level = 100 + resolution
        print(f"[unified] Processing {city_name} - H3 Resolution {resolution} ({len(sub)} cells)")

        # Build GeoJSON features
        features = []
        for idx, row in sub.iterrows():
            cell = row[h3_col]

            try:
                boundary = h3.cell_to_boundary(cell)
            except Exception:
                continue

            # Convert to GeoJSON coordinates
            ring = [[lon, lat] for lat, lon in boundary]
            if ring[0] != ring[-1]:
                ring.append(ring[0])

            props = {
                "admin_level": int(admin_level),
                "cell_code": str(cell),
                "id": str(cell),
                "city": city_name,  # Add city identifier
            }

            # Add value columns
            for col in (value_cols or []):
                if col in row:
                    v = row[col]
                    props[col] = None if pd.isna(v) else float(v)

            features.append({
                "type": "Feature",
                "properties": props,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [ring],
                },
            })

        # Write temporary file for this city/resolution
        temp_file = os.path.join(temp_dir, f"temp_{city_name}_h3_res_{resolution}.geojson")
        fc = {
            "type": "FeatureCollection",
            "name": f"temp_{city_name}_admin_level_{admin_level}",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
            },
            "features": features,
        }

        with open(temp_file, "w") as f:
            json.dump(fc, f)

        print(f"[unified]   ✓ {city_name} res {resolution}: {len(features)} features")
        return temp_file

    except Exception as e:
        print(f"[unified]   ✗ Error processing {city_name} H3 res {resolution}: {e}")
        return None


def process_h3_tilesets(config: Dict[str, Any], max_workers: int = 4, on_resolution_done=None) -> Dict[int, str]:
    """Generate H3 hexagon tilesets for all cities.

    Loads each city's CSV once and processes all resolutions from it to avoid
    loading a multi-GB CSV repeatedly.

    Args:
        on_resolution_done: optional callback(resolution, output_path) called immediately
                            after each resolution's combined output is written to disk.
    """
    output_dir = config["output_dir"]
    temp_dir = os.path.join(output_dir, "_temp")
    os.makedirs(temp_dir, exist_ok=True)

    h3_col = config.get("h3_col", "h3")
    value_cols = config.get("value_cols", None)
    cities = config["cities"]
    h3_resolutions = sorted(set(config.get("h3_hex_resolutions", [9, 10, 11, 12])))

    if not h3_resolutions:
        print("[unified] No H3 tilesets requested")
        return {}

    print(f"\n[unified] === Generating H3 Tilesets (Resolutions: {h3_resolutions}) ===")

    temp_files_by_resolution = {res: [] for res in h3_resolutions}

    for city in cities:
        city_name = city["name"]
        h3_csv = city["h3_csv"]

        if not os.path.exists(h3_csv):
            print(f"[unified] No CSV found for {city_name}: {h3_csv}")
            continue

        print(f"[unified] Loading CSV for {city_name}...")
        df = pd.read_csv(h3_csv)

        # Compute h3_res once for all resolutions
        if "h3_res" in df.columns:
            df["h3_res"] = df["h3_res"].astype(int)
        else:
            print(f"[unified] Computing h3_res for {len(df)} rows...")
            df["h3_res"] = df[h3_col].apply(h3.get_resolution)

        for resolution in h3_resolutions:
            sub = df[df["h3_res"] == resolution]
            if sub.empty:
                continue

            admin_level = 100 + resolution
            print(f"[unified] Processing {city_name} res {resolution}: {len(sub)} cells")

            temp_file = os.path.join(temp_dir, f"temp_{city_name}_h3_res_{resolution}.geojson")

            # Pre-build column index list for fast access via itertuples
            all_cols = list(sub.columns)
            h3_col_idx = all_cols.index(h3_col)
            val_col_indices = {col: all_cols.index(col) for col in (value_cols or []) if col in all_cols}

            # Stream features directly to file using itertuples (much less memory than iterrows)
            GC_INTERVAL = 50_000  # call gc.collect() every N features to release Python allocator memory
            with open(temp_file, "w") as f:
                f.write('{"type":"FeatureCollection","name":"temp_%s_admin_level_%d",'
                        '"crs":{"type":"name","properties":{"name":"urn:ogc:def:crs:OGC:1.3:CRS84"}},'
                        '"features":[' % (city_name, admin_level))
                first = True
                count = 0
                for row_tuple in sub.itertuples(index=False, name=None):
                    cell = row_tuple[h3_col_idx]
                    try:
                        boundary = h3.cell_to_boundary(cell)
                    except Exception:
                        continue

                    ring = [[lon, lat] for lat, lon in boundary]
                    if ring[0] != ring[-1]:
                        ring.append(ring[0])

                    lons = [c[0] for c in ring]
                    lats = [c[1] for c in ring]
                    feat_bbox = [min(lons), min(lats), max(lons), max(lats)]

                    props = {
                        "admin_level": int(admin_level),
                        "cell_code": str(cell),
                        "id": str(cell),
                        "city": city_name,
                        "bbox": feat_bbox,
                    }
                    for col, idx in val_col_indices.items():
                        v = row_tuple[idx]
                        props[col] = None if (v != v) else float(v)  # NaN check without pd.isna

                    feature = {
                        "type": "Feature",
                        "properties": props,
                        "geometry": {"type": "Polygon", "coordinates": [ring]},
                    }
                    if not first:
                        f.write(",")
                    json.dump(feature, f)
                    first = False
                    count += 1
                    if count % GC_INTERVAL == 0:
                        gc.collect()
                        try:
                            ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
                        except Exception:
                            pass
                f.write("]}")

            print(f"[unified]   ✓ {city_name} res {resolution}: {count} features")
            if count > 0:
                temp_files_by_resolution[resolution].append(temp_file)
            del sub
            gc.collect()
            try:
                ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
            except Exception:
                pass

        del df  # Free memory before loading next city's CSV
        gc.collect()
        try:
            ctypes.cdll.LoadLibrary("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    # Combine results for each resolution
    combined_outputs = {}
    for resolution in h3_resolutions:
        admin_level = 100 + resolution
        temp_files = temp_files_by_resolution[resolution]

        if not temp_files:
            print(f"[unified] No H3 data for resolution {resolution}")
            continue

        combined_output = os.path.join(output_dir, f"admin_level_{admin_level}_all_cities.geojson")
        print(f"\n[unified] Combining H3 resolution {resolution} from {len(temp_files)} cities")
        combine_geojson_files(temp_files, combined_output, admin_level)
        combined_outputs[resolution] = combined_output

        # Upload immediately after each resolution finishes — don't wait for all resolutions
        if on_resolution_done:
            on_resolution_done(resolution, combined_output)

    return combined_outputs


def main():
    parser = argparse.ArgumentParser(
        description="Unified multi-city H3 pipeline - processes all cities in one run"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (e.g., cities_config.yaml)",
    )
    parser.add_argument(
        "--skip-admin",
        action="store_true",
        help="Skip admin boundary processing (only generate H3 tilesets)",
    )
    parser.add_argument(
        "--skip-h3-tiles",
        action="store_true",
        help="Skip H3 tileset generation (only process admin boundaries)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for H3 tileset generation (default: 4)",
    )

    args = parser.parse_args()

    # Load configuration
    print(f"[unified] Loading config from {args.config}")
    config = load_config(args.config)

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"[unified] Processing {len(config['cities'])} cities")
    print(f"[unified] Output directory: {output_dir}")

    # Process admin boundaries
    if not args.skip_admin:
        admin_outputs = process_admin_levels(config)
        print(f"\n[unified] Admin boundary tilesets created: {len(admin_outputs)}")
    else:
        print("[unified] Skipping admin boundary processing")
        admin_outputs = {}

    # Generate H3 tilesets
    if not args.skip_h3_tiles:
        h3_outputs = process_h3_tilesets(config, max_workers=args.workers)
        print(f"\n[unified] H3 tilesets created: {len(h3_outputs)}")
    else:
        print("[unified] Skipping H3 tileset generation")
        h3_outputs = {}

    # Summary
    all_outputs = {**admin_outputs, **h3_outputs}
    total_size_mb = sum(
        os.path.getsize(fp) / (1024 * 1024)
        for fp in all_outputs.values()
        if os.path.exists(fp)
    )

    print(f"\n[unified] ===== PIPELINE COMPLETE =====")
    print(f"[unified] Total tilesets created: {len(all_outputs)}")
    print(f"[unified] Total size: {total_size_mb:.2f} MB")
    print(f"[unified] Output directory: {output_dir}")

    if total_size_mb > 20 * 1024:  # 20 GB
        print(f"[unified] ⚠️  WARNING: Total size exceeds Mapbox 20GB limit!")
        print(f"[unified]    Consider reducing H3 resolutions or filtering cities")


if __name__ == "__main__":
    main()

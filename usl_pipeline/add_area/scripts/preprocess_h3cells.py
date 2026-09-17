# -*- coding: utf-8 -*-
"""Extracts city-clipped geo datasets from the H3 source tiles."""

import argparse
import inspect
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import traceback

import geopandas as gpd
import pandas as pd
import rasterio

from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.crs import CRS
from rasterio import features
from shapely.geometry import mapping, shape


DEFAULT_TARGET_CRS = "EPSG:3395"
DEFAULT_WORKING_DIR = "gs://raw-data-h3index/Working_files"
DEFAULT_H3_ROOT = "gs://raw-data-h3index/CONUS_Data_H3Index"
# Generated outputs go under data/, which is gitignored.
DEFAULT_OUTPUT_DIR = Path("data/output")

# Urban areas shapefile.
DEFAULT_URBAN_AREAS_FILE = "01_urban_areas_simplified_with_state.shp"
DEFAULT_COUNTRY_FIELD = "CNTRY_NAME"
DEFAULT_CITY_FIELD = "JRC_NAME_M"
DEFAULT_STATE_FIELD = "State"
DEFAULT_STATE_ABBR_FIELD = "State_Abbr"
DEFAULT_ISO_FIELD = "ISO"
DEFAULT_SMOD_FIELD = "SMOD_LEVEL"
DEFAULT_AREA_FIELD = "AREA_SQKM"

GCS_PREFIX = "gs://"

# Layout of a staged city folder (<destination>/<City>/...), following the convention
# already used under gs://citycat-preprocessing/BETA_10_CITIES (folder named by city,
# boundary file named <City>_boundary.shp).
# Shapefile entries copy all sidecar files (.shx/.dbf/.prj/.cpg) along with the .shp.
STAGED_LAYOUT = {
    "dem": ("DEM", "DEM_2m.tif"),
    "buildings": ("Buildings", "Building_footprints.shp"),
    "soil": ("Soil", "soil.shp"),
    "green_spaces_polygons": ("Green_Spaces", "Green_spaces.shp"),
    "green_spaces": ("Green_Spaces", "Green_spaces_30m.tif"),
    "landcover": ("Landcover", "Landcover_30m.tif"),
    "city_boundary": ("City_Boundary", "{city}_boundary.shp"),
    "h3_cells": ("H3", "H3_cells.shp"),
    "h3_index_list": ("H3", "h3_index_list.txt"),
}
# A city counts as already staged when all of these are present at the destination.
REQUIRED_STAGED_KEYS = [
    "dem",
    "buildings",
    "soil",
    "green_spaces_polygons",
    "city_boundary",
]
# File extensions that make up an ESRI Shapefile.
SHAPEFILE_EXTENSIONS = {".shp", ".shx", ".dbf", ".prj", ".cpg", ".sbn", ".sbx", ".qix"}
ADC_FILE = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"

_storage_client = None


# Path helpers: every input location may be a local path or a gs:// URI
def is_gcs(path):
    """True if the path is a gs:// URI."""
    return str(path).startswith(GCS_PREFIX)


def split_gcs(uri):
    """Split gs://bucket/key into (bucket, key)."""
    bucket, _, key = str(uri)[len(GCS_PREFIX) :].partition("/")
    return bucket, key


def join_path(base, *parts):
    """Join components onto a local path or a gs:// URI."""
    if is_gcs(base):
        return "/".join([str(base).rstrip("/"), *parts])
    return Path(base).joinpath(*parts)


def basename(path):
    """Final component of a local path or gs:// URI."""
    return PurePosixPath(str(path)).name


def gdal_path(path):
    """Path in a form GDAL can open: a local path, or /vsigs/ for GCS."""
    if is_gcs(path):
        bucket, key = split_gcs(path)
        return f"/vsigs/{bucket}/{key}"
    return str(path)


def get_storage_client():
    """Lazily created google-cloud-storage client (only needed for gs:// inputs)."""
    global _storage_client
    if _storage_client is None:
        from google.cloud import storage

        _storage_client = storage.Client()
    return _storage_client


def path_exists(path, is_dir=False):
    """Existence check for a local path or gs:// URI.

    A GCS "directory" exists if any object lives under the prefix.
    """
    if not is_gcs(path):
        return Path(path).exists()

    bucket, key = split_gcs(path)
    client = get_storage_client()
    if is_dir:
        blobs = client.list_blobs(bucket, prefix=key.rstrip("/") + "/", max_results=1)
        return next(iter(blobs), None) is not None
    return client.bucket(bucket).blob(key).exists()


def configure_gdal_gcs_access():
    """Point GDAL at gcloud's application default credentials for /vsigs/ reads."""
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
        if not ADC_FILE.exists():
            raise RuntimeError(
                "Reading gs:// inputs needs Google credentials for GDAL. Run "
                "`gcloud auth application-default login` or set "
                "GOOGLE_APPLICATION_CREDENTIALS."
            )
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(ADC_FILE)

    # Skip the directory listing GDAL would otherwise do on every open. Shapefile
    # sidecars (.shx/.dbf/.prj) are still found because they are probed by name.
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")


def union_all(geoseries):
    """Union of all geometries.

    geopandas >= 1.0 exposes union_all; older versions only unary_union.
    """
    if hasattr(geoseries, "union_all"):
        return geoseries.union_all()
    return geoseries.unary_union


def copy_file(local_file, destination_path):
    """Copy a local file to a local path or gs:// URI."""
    if is_gcs(destination_path):
        bucket, key = split_gcs(destination_path)
        get_storage_client().bucket(bucket).blob(key).upload_from_filename(
            str(local_file)
        )
    else:
        Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_file, destination_path)


def staged_path(destination, city_name, key):
    """Destination path of one staged output for a city."""
    subfolder, name = STAGED_LAYOUT[key]
    return join_path(
        destination, city_name, subfolder, name.format(city=safe_name(city_name))
    )


def staged_files_present(destination, city_name):
    """True if every required staged output already exists at the destination."""
    return all(
        path_exists(staged_path(destination, city_name, key))
        for key in REQUIRED_STAGED_KEYS
    )


def stage_outputs(outputs, destination, city_name):
    """Copy a city's final outputs into the staged layout at the destination."""
    print("\n" + "=" * 80)
    print("STEP 10 — STAGE OUTPUTS")
    print("=" * 80)
    print(f"Destination: {join_path(destination, city_name)}")

    for key in STAGED_LAYOUT:
        local_file = Path(outputs[key])
        target = staged_path(destination, city_name, key)

        if local_file.suffix == ".shp":
            # Shapefiles are several files sharing a stem; only copy shapefile members
            # (a raster with the same stem must not be swept up).
            for sidecar in sorted(local_file.parent.glob(local_file.stem + ".*")):
                if sidecar.suffix.lower() not in SHAPEFILE_EXTENSIONS:
                    continue
                sidecar_target = str(target)[: -len(local_file.suffix)] + sidecar.suffix
                copy_file(sidecar, sidecar_target)
                print(f"  {sidecar.name} -> {sidecar_target}")
        else:
            copy_file(local_file, target)
            print(f"  {local_file.name} -> {target}")


# Supporting functions
def safe_name(text):
    """Create a safe folder/file name."""
    text = re.sub(r'[<>:"/\\|?*]+', "_", str(text).strip())
    text = re.sub(r"\s+", "_", text)
    return text.strip("_")


def validate_target_crs(target_crs):
    """Allow EPSG:4326 or EPSG:3395."""
    target_crs = target_crs.upper().strip()
    if target_crs not in {"EPSG:4326", "EPSG:3395"}:
        raise ValueError(
            f"TARGET_CRS must be EPSG:4326 or EPSG:3395. Received: {target_crs}"
        )
    return target_crs


def _casefold_match(series, value):
    """Case-insensitive, whitespace-insensitive equality mask for a column."""
    return (
        series.fillna("").astype(str).str.strip().str.casefold()
        == str(value).strip().casefold()
    )


def select_city(
    urban_shp,
    country_name,
    city_name,
    state=None,
    country_field=DEFAULT_COUNTRY_FIELD,
    city_field=DEFAULT_CITY_FIELD,
    state_field=DEFAULT_STATE_FIELD,
    state_abbr_field=DEFAULT_STATE_ABBR_FIELD,
    allow_multiple_features=False,
    smod_level=None,
    smod_field=DEFAULT_SMOD_FIELD,
    iso_field=DEFAULT_ISO_FIELD,
    area_field=DEFAULT_AREA_FIELD,
):
    """Select requested city polygon.

    Returns:
      A GeoDataFrame of the matching features.

    Raises:
      ValueError: If nothing matches, or if several features match and
        allow_multiple_features is not set.
    """
    print("\n" + "=" * 80)
    print("STEP 1 — SELECT CITY")
    print("=" * 80)

    if is_gcs(urban_shp):
        configure_gdal_gcs_access()

    urban = gpd.read_file(gdal_path(urban_shp))

    for field in [country_field, city_field]:
        if field not in urban.columns:
            raise KeyError(f"Required field '{field}' not found in {urban_shp}")

    if urban.crs is None:
        raise ValueError(f"Urban-area file has no defined CRS: {urban_shp}")

    country_mask = _casefold_match(urban[country_field], country_name)
    city_mask = _casefold_match(urban[city_field], city_name)
    selected_mask = country_mask & city_mask
    available_state_fields = [
        f for f in (state_field, state_abbr_field) if f in urban.columns
    ]

    if state is not None:
        state_mask = False
        for field in available_state_fields:
            state_mask = state_mask | _casefold_match(urban[field], state)
        selected_mask = selected_mask & state_mask

    if smod_level is not None:
        if smod_field not in urban.columns:
            raise KeyError(f"Required field '{smod_field}' not found in {urban_shp}")
        selected_mask = selected_mask & _casefold_match(urban[smod_field], smod_level)

    selected_city = urban[selected_mask].copy()

    if selected_city.empty:
        location = f"'{city_name}'" + (
            f" in state '{state}'" if state is not None else ""
        )
        print(f"\nCity {location} was not found in '{country_name}'.")

        available = (
            urban.loc[country_mask, city_field]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
        )
        if len(available) > 0:
            print("\nSome available city names:")
            for name in available[:50]:
                print(f"  {name}")

        raise ValueError("Country/city selection returned no polygons.")

    if len(selected_city) > 1 and not allow_multiple_features:
        print(
            f"\n{len(selected_city)} features match '{city_name}' in '{country_name}':"
        )
        for _, row in selected_city.iterrows():
            parts = []
            for field in (state_abbr_field, smod_field, area_field):
                if field in selected_city.columns:
                    value = row[field]
                    if field == area_field:
                        parts.append(f"{field}={float(value):,.1f}")
                    else:
                        parts.append(f"{field}={value}")
            print("  " + ", ".join(parts))

        raise ValueError(
            f"Country/city selection matched {len(selected_city)} features for "
            f"'{city_name}' in '{country_name}' (listed above). These would be merged "
            "into a single study area."
        )

    print(f"Country: {country_name}")
    if state is not None:
        print(f"State: {state}")
    print(f"City: {city_name}")
    print(f"Matching polygon(s): {len(selected_city)}")
    print(f"Input CRS: {selected_city.crs}")

    return selected_city


def find_intersecting_h3(h3_shp, selected_city, h3_field="h3_index"):
    """Find H3 cells intersecting selected city."""
    print("\n" + "=" * 80)
    print("STEP 2 — FIND INTERSECTING H3 CELLS")
    print("=" * 80)

    h3 = gpd.read_file(gdal_path(h3_shp))

    if h3_field not in h3.columns:
        raise KeyError(f"Required field '{h3_field}' not found in {h3_shp}")

    if h3.crs is None:
        raise ValueError(f"H3 shapefile has no defined CRS: {h3_shp}")

    h3_test = h3.to_crs(selected_city.crs) if h3.crs != selected_city.crs else h3.copy()
    city_geometry = union_all(selected_city.geometry)
    selected_h3 = h3.loc[h3_test.geometry.intersects(city_geometry)].copy()

    if selected_h3.empty:
        raise RuntimeError("No H3 cells intersect the selected city.")

    h3_indexes = (
        selected_h3[h3_field]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    print(f"Number of intersecting H3 cells: {len(h3_indexes)}")
    for h3_index in h3_indexes:
        print(f"  {h3_index}")

    return selected_h3, h3_indexes


def create_h3_city_masks(selected_city, selected_h3, h3_field="h3_index"):
    """Create city ∩ H3 geometries in memory."""
    print("\n" + "=" * 80)
    print("STEP 3 — CREATE CITY ∩ H3 MASKS")
    print("=" * 80)

    h3_local = selected_h3.to_crs(selected_city.crs)
    city_geometry = union_all(selected_city.geometry)
    masks = {}

    for _, row in h3_local.iterrows():
        h3_index = str(row[h3_field]).strip()
        intersection = row.geometry.intersection(city_geometry)

        if intersection.is_empty:
            continue

        masks[h3_index] = gpd.GeoDataFrame(
            {h3_field: [h3_index]}, geometry=[intersection], crs=selected_city.crs
        )
        print(f"Created mask geometry: {h3_index}")

    if not masks:
        raise RuntimeError("No valid city/H3 intersection masks were created.")

    return masks


def collect_h3_inputs(h3_root, h3_indexes):
    """Locate all required H3 datasets."""
    print("\n" + "=" * 80)
    print("STEP 4 — LOCATE H3 INPUT DATA")
    print("=" * 80)

    datasets = {}
    missing = []

    for h3_index in h3_indexes:
        print(f"\nH3: {h3_index}")
        h3_dir = join_path(h3_root, h3_index)

        if not path_exists(h3_dir, is_dir=True):
            print(f"  MISSING H3 folder: {h3_dir}")
            missing.append((h3_index, "H3 folder", h3_dir))
            continue

        expected = {
            "dem": join_path(h3_dir, "Elevation", f"DEM_with_buildings_{h3_index}.tif"),
            "buildings": join_path(h3_dir, "Buildings", f"Buildings_{h3_index}.shp"),
            "green_spaces": join_path(
                h3_dir, "Green_spaces", f"green_spaces_{h3_index}.tif"
            ),
            "soil": join_path(h3_dir, "Soil", f"Soil_texture_{h3_index}.shp"),
            "landcover": join_path(h3_dir, "Land_use", f"Landcover_{h3_index}.tif"),
        }

        datasets[h3_index] = expected

        for dataset_name, path in expected.items():
            if path_exists(path):
                print(f"  FOUND   {dataset_name}: {basename(path)}")
            else:
                print(f"  MISSING {dataset_name}: {path}")
                missing.append((h3_index, dataset_name, path))

    if missing:
        print("\nRequired input files are missing:")
        for h3_index, dataset_name, path in missing:
            print(f"  {h3_index} | {dataset_name} | {path}")
        raise FileNotFoundError(
            "Processing stopped because required H3 datasets are missing."
        )

    return datasets


def extract_raster_by_mask(
    input_raster,
    mask_gdf,
    output_raster,
    target_crs,
    resolution=None,
    resampling=Resampling.nearest,
):
    """Extract city portion from one H3 raster and optionally reproject/resample.

    Only the blocks covering the mask are read, so this works on remote (gs://)
    rasters without downloading them in full.
    """
    output_raster.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(gdal_path(input_raster)) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {input_raster}")

        target_crs_obj = CRS.from_string(target_crs)
        mask_target = mask_gdf.to_crs(target_crs_obj)
        geometries = [
            mapping(geom)
            for geom in mask_target.geometry
            if geom is not None and not geom.is_empty
        ]

        if not geometries:
            raise RuntimeError(f"No valid mask geometry for {input_raster}")

        vrt_options = {"crs": target_crs_obj, "resampling": resampling}
        if resolution is not None:
            vrt_options["resolution"] = (resolution, resolution)

        with WarpedVRT(src, **vrt_options) as vrt:
            out_image, out_transform = mask(
                vrt, geometries, crop=True, filled=True, nodata=vrt.nodata
            )
            profile = vrt.profile.copy()
            profile.update(
                driver="GTiff",
                height=out_image.shape[1],
                width=out_image.shape[2],
                transform=out_transform,
                crs=target_crs_obj,
                compress="LZW",
                tiled=True,
                BIGTIFF="IF_SAFER",
            )

            with rasterio.open(output_raster, "w", **profile) as dst:
                dst.write(out_image)

    print(f"  Created raster piece: {output_raster.name}")


def clip_vector_by_mask(input_vector, mask_gdf, output_vector, target_crs):
    """Clip one H3 vector dataset to city/H3 intersection.

    Only features within the mask's bounding box are loaded; a full per-cell
    buildings file can be several GB.
    """
    source_path = gdal_path(input_vector)
    source_crs = gpd.read_file(source_path, rows=1).crs

    if source_crs is None:
        raise ValueError(f"Vector file has no CRS: {input_vector}")

    mask_local = mask_gdf.to_crs(source_crs)
    source = gpd.read_file(source_path, bbox=tuple(mask_local.total_bounds))

    if source.empty:
        print(f"  No features within mask bounds: {basename(input_vector)}")
        return False

    clipped = gpd.clip(source, mask_local)

    if clipped.empty:
        print(f"  No features remain after clipping: {basename(input_vector)}")
        return False

    clipped = clipped.to_crs(target_crs)
    output_vector.parent.mkdir(parents=True, exist_ok=True)
    clipped.to_file(output_vector, layer="data", driver="GPKG")

    print(f"  Created vector piece: {output_vector.name}")
    return True


def mosaic_raster_parts(
    raster_parts, output_raster, target_resolution=None, resampling=Resampling.nearest
):
    """Mosaic already cropped raster pieces directly to storage."""
    if not raster_parts:
        raise RuntimeError(f"No raster pieces supplied for {output_raster.name}")

    print(f"\nMosaicking {len(raster_parts)} cropped raster piece(s)...")
    sources = [rasterio.open(path) for path in raster_parts]

    try:
        merge_kwargs = {
            "dst_path": output_raster,
            "resampling": resampling,
            "method": "first",
            "target_aligned_pixels": True,
            "dst_kwds": {
                "driver": "GTiff",
                "compress": "LZW",
                "tiled": True,
                "BIGTIFF": "IF_SAFER",
            },
        }

        # mem_limit only exists in rasterio >= 1.4; the first parameter is also named
        # differently across versions (datasets vs sources), so pass it positionally.
        if "mem_limit" in inspect.signature(merge).parameters:
            merge_kwargs["mem_limit"] = 256

        if target_resolution is not None:
            merge_kwargs["res"] = (target_resolution, target_resolution)

        merge(sources, **merge_kwargs)

    finally:
        for src in sources:
            src.close()

    if not output_raster.exists():
        raise RuntimeError(f"Final mosaic was not created: {output_raster}")

    print(f"Created final raster: {output_raster}")


def merge_vector_parts(vector_parts, output_shp, target_crs):
    """Merge already-clipped vector pieces."""
    if not vector_parts:
        raise RuntimeError(f"No clipped vector pieces supplied for {output_shp.name}")

    layers = []

    for path in vector_parts:
        gdf = gpd.read_file(path, layer="data")

        if not gdf.empty:
            if str(gdf.crs) != target_crs:
                gdf = gdf.to_crs(target_crs)
            layers.append(gdf)

    if not layers:
        raise RuntimeError(f"No vector features remain for {output_shp.name}")

    merged = gpd.GeoDataFrame(pd.concat(layers, ignore_index=True), crs=layers[0].crs)
    merged.to_file(output_shp, driver="ESRI Shapefile")

    print(f"Features written: {len(merged):,}")
    print(f"Created final vector: {output_shp}")


def polygonize_green_spaces(green_raster, output_shp):
    """Convert the green-space raster into polygons.

    The study area uploader and the already-staged cities use a green areas
    shapefile, so the raster alone is not usable downstream.
    """
    with rasterio.open(green_raster) as src:
        band = src.read(1)
        keep = band > 0
        if src.nodata is not None:
            keep &= band != src.nodata
        records = [
            (shape(geom), int(value))
            for geom, value in features.shapes(band, mask=keep, transform=src.transform)
        ]
        crs = src.crs

    if not records:
        raise RuntimeError(f"No green-space pixels found in {green_raster}")

    gdf = gpd.GeoDataFrame(
        {"value": [value for _, value in records]},
        geometry=[geom for geom, _ in records],
        crs=crs,
    )
    gdf.to_file(output_shp, driver="ESRI Shapefile")

    print(f"Green-space polygons written: {len(gdf):,}")
    print(f"Created final vector: {output_shp}")


def save_reference_outputs(
    selected_city, selected_h3, h3_indexes, output_dir, location_name, target_crs
):
    """Save city boundary, selected H3 polygons, and H3 index list."""
    print("\n" + "=" * 80)
    print("STEP 5 — SAVE REFERENCE OUTPUTS")
    print("=" * 80)

    h3_list_file = output_dir / "h3_index_list.txt"
    city_file = output_dir / f"City_boundary_{location_name}.shp"
    h3_file = output_dir / f"H3_cells_{location_name}.shp"

    with open(h3_list_file, "w", encoding="utf-8") as f:
        for h3_index in h3_indexes:
            f.write(f"{h3_index}\n")

    selected_city.to_crs(target_crs).to_file(city_file, driver="ESRI Shapefile")
    selected_h3.to_crs(target_crs).to_file(h3_file, driver="ESRI Shapefile")

    print(f"H3 index list: {h3_list_file}")
    print(f"City boundary: {city_file}")
    print(f"H3 polygons: {h3_file}")

    return h3_list_file, city_file, h3_file


def resolve_area_identity(
    urban_shp,
    country_name,
    city_name,
    state=None,
    smod_level=None,
    iso_field=DEFAULT_ISO_FIELD,
    state_abbr_field=DEFAULT_STATE_ABBR_FIELD,
    city_field=DEFAULT_CITY_FIELD,
    allow_multiple_features=False,
):
    """Resolves a city to its ISO code, state abbreviation and selected geometry.

    Returns:
      A tuple of (iso, state_abbr, city_name, selected_city). state_abbr is
      "XX" outside the US, where the shapefile carries no state.
    """
    selected_city = select_city(
        urban_shp,
        country_name,
        city_name,
        state=state,
        smod_level=smod_level,
        allow_multiple_features=allow_multiple_features,
    )

    first = selected_city.iloc[0]
    iso = str(first[iso_field]).strip() if iso_field in selected_city.columns else None
    state_abbr = (
        str(first[state_abbr_field]).strip()
        if state_abbr_field in selected_city.columns
        else None
    )
    resolved_city = str(first[city_field]).strip()

    return iso, state_abbr, resolved_city, selected_city


# Main executable function
def process_city_data(
    country_name,
    city_name,
    target_crs=DEFAULT_TARGET_CRS,
    working_dir=DEFAULT_WORKING_DIR,
    h3_root=DEFAULT_H3_ROOT,
    output_dir=DEFAULT_OUTPUT_DIR,
    state=None,
    area_name=None,
    urban_areas_file=DEFAULT_URBAN_AREAS_FILE,
    state_field=DEFAULT_STATE_FIELD,
    state_abbr_field=DEFAULT_STATE_ABBR_FIELD,
    allow_multiple_features=False,
    smod_level=None,
    preselected_city=None,
):
    """Extract and assemble city-only datasets from H3 source folders.

    working_dir and h3_root may be local directories or gs:// URIs. Results are
    written to <output_dir>/<area_name>/ on the local filesystem, defaulting to
    <Country>_<City> when no area_name is given.
    """
    print("\n" + "=" * 80)
    print("CITY DATA EXTRACTION AND ASSEMBLY")
    print("=" * 80)

    target_crs = validate_target_crs(target_crs)
    urban_shp = join_path(working_dir, urban_areas_file)
    h3_shp = join_path(working_dir, "h3_cells_level_2.shp")

    if is_gcs(working_dir) or is_gcs(h3_root):
        configure_gdal_gcs_access()

    if not path_exists(urban_shp):
        raise FileNotFoundError(str(urban_shp))
    if not path_exists(h3_shp):
        raise FileNotFoundError(str(h3_shp))
    if not path_exists(h3_root, is_dir=True):
        raise FileNotFoundError(str(h3_root))

    location_name = area_name or f"{safe_name(country_name)}_{safe_name(city_name)}"
    output_dir = Path(output_dir) / location_name
    temp_dir = output_dir / "_temp"
    raster_temp = temp_dir / "rasters"
    vector_temp = temp_dir / "vectors"

    output_dir.mkdir(parents=True, exist_ok=True)
    raster_temp.mkdir(parents=True, exist_ok=True)
    vector_temp.mkdir(parents=True, exist_ok=True)

    print(f"\nCountry: {country_name}")
    if state is not None:
        print(f"State: {state}")
    print(f"City: {city_name}")
    print(f"Study area name: {location_name}")
    print(f"Target CRS: {target_crs}")
    print(f"Working files: {working_dir}")
    print(f"H3 root: {h3_root}")
    print(f"Output directory: {output_dir}")

    # Callers that already resolved the city (to derive its canonical name) can
    # pass the selection in rather than paying for a second read of the shapefile.
    if preselected_city is not None:
        selected_city = preselected_city
    else:
        selected_city = select_city(
            urban_shp,
            country_name,
            city_name,
            state=state,
            state_field=state_field,
            state_abbr_field=state_abbr_field,
            allow_multiple_features=allow_multiple_features,
            smod_level=smod_level,
        )
    selected_h3, h3_indexes = find_intersecting_h3(h3_shp, selected_city)
    masks = create_h3_city_masks(selected_city, selected_h3)
    datasets = collect_h3_inputs(h3_root, h3_indexes)
    h3_list_file, city_file, h3_file = save_reference_outputs(
        selected_city, selected_h3, h3_indexes, output_dir, location_name, target_crs
    )

    if target_crs == "EPSG:3395":
        dem_resolution, green_resolution, landcover_resolution = 2, 30, 30
    else:
        dem_resolution, green_resolution, landcover_resolution = None, None, None

    dem_parts, building_parts, green_parts, soil_parts, landcover_parts = (
        [],
        [],
        [],
        [],
        [],
    )

    print("\n" + "=" * 80)
    print("STEP 6 — EXTRACT CITY DATA FROM EACH H3")
    print("=" * 80)

    for number, h3_index in enumerate(h3_indexes, start=1):
        print("\n" + "-" * 80)
        print(f"H3 {number}/{len(h3_indexes)}: {h3_index}")
        print("-" * 80)

        mask_gdf = masks[h3_index]
        source = datasets[h3_index]

        print("\nDEM")
        dem_part = raster_temp / f"DEM_{h3_index}.tif"
        extract_raster_by_mask(
            source["dem"],
            mask_gdf,
            dem_part,
            target_crs,
            dem_resolution,
            Resampling.bilinear,
        )
        dem_parts.append(dem_part)

        print("\nBuildings")
        building_part = vector_temp / f"Buildings_{h3_index}.gpkg"
        if clip_vector_by_mask(
            source["buildings"], mask_gdf, building_part, target_crs
        ):
            building_parts.append(building_part)

        print("\nGreen spaces")
        green_part = raster_temp / f"green_spaces_{h3_index}.tif"
        extract_raster_by_mask(
            source["green_spaces"],
            mask_gdf,
            green_part,
            target_crs,
            green_resolution,
            Resampling.nearest,
        )
        green_parts.append(green_part)

        print("\nSoil")
        soil_part = vector_temp / f"Soil_{h3_index}.gpkg"
        if clip_vector_by_mask(source["soil"], mask_gdf, soil_part, target_crs):
            soil_parts.append(soil_part)

        print("\nLandcover")
        landcover_part = raster_temp / f"Landcover_{h3_index}.tif"
        extract_raster_by_mask(
            source["landcover"],
            mask_gdf,
            landcover_part,
            target_crs,
            landcover_resolution,
            Resampling.nearest,
        )
        landcover_parts.append(landcover_part)

    dem_output = output_dir / f"DEM_{location_name}.tif"
    buildings_output = output_dir / f"Buildings_{location_name}.shp"
    green_output = output_dir / f"green_spaces_{location_name}.tif"
    soil_output = output_dir / f"Soil_{location_name}.shp"
    landcover_output = output_dir / f"Landcover_{location_name}.tif"
    green_polygons_output = output_dir / f"green_spaces_{location_name}.shp"

    print("\n" + "=" * 80)
    print("STEP 7 — MOSAIC EXTRACTED CITY RASTERS")
    print("=" * 80)

    print("\nDEM")
    mosaic_raster_parts(dem_parts, dem_output, dem_resolution, Resampling.bilinear)

    print("\nGreen spaces")
    mosaic_raster_parts(green_parts, green_output, green_resolution, Resampling.nearest)

    print("\nLandcover")
    mosaic_raster_parts(
        landcover_parts, landcover_output, landcover_resolution, Resampling.nearest
    )

    print("\n" + "=" * 80)
    print("STEP 8 — MERGE CLIPPED CITY VECTORS")
    print("=" * 80)

    print("\nBuildings")
    merge_vector_parts(building_parts, buildings_output, target_crs)

    print("\nSoil")
    merge_vector_parts(soil_parts, soil_output, target_crs)

    print("\nGreen spaces (polygons from the mosaicked raster)")
    polygonize_green_spaces(green_output, green_polygons_output)

    print("\n" + "=" * 80)
    print("STEP 9 — CLEAN TEMPORARY FILES")
    print("=" * 80)

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        print(f"Removed: {temp_dir}")

    print("\n" + "=" * 80)
    print("PROCESS COMPLETE")
    print("=" * 80)

    print(f"\nCountry: {country_name}")
    print(f"City: {city_name}")
    print(f"Target CRS: {target_crs}")
    print(f"H3 cells used: {len(h3_indexes)}")

    if target_crs == "EPSG:3395":
        print("\nFinal raster resolutions:")
        print("  DEM:          2 m")
        print("  Green spaces: 30 m")
        print("  Landcover:    30 m")
    else:
        print(
            "\nEPSG:4326 selected. Raster resolution remains in geographic "
            "coordinates and is not forced to meter-based resolution."
        )

    print("\nFinal outputs:")
    print(f"  {dem_output}")
    print(f"  {buildings_output}")
    print(f"  {green_output}")
    print(f"  {soil_output}")
    print(f"  {landcover_output}")
    print(f"  {green_polygons_output}")

    return {
        "output_dir": output_dir,
        "h3_indexes": h3_indexes,
        "h3_index_list": h3_list_file,
        "city_boundary": city_file,
        "h3_cells": h3_file,
        "dem": dem_output,
        "buildings": buildings_output,
        "green_spaces": green_output,
        "green_spaces_polygons": green_polygons_output,
        "soil": soil_output,
        "landcover": landcover_output,
    }


def run_cities(
    country_name,
    city_names,
    target_crs,
    working_dir,
    h3_root,
    output_dir,
    destination=None,
    overwrite=False,
    dry_run=False,
    state=None,
    area_name=None,
    urban_areas_file=DEFAULT_URBAN_AREAS_FILE,
    state_field=DEFAULT_STATE_FIELD,
    state_abbr_field=DEFAULT_STATE_ABBR_FIELD,
    allow_multiple_features=False,
    smod_level=None,
):
    """Process several cities, skipping ones already staged.

    One failure does not stop the rest.

    Returns {city_name: status string}; a status starting with "FAILED" marks an error.
    """
    if area_name is not None and len(city_names) > 1:
        raise ValueError("--area-name can only be used with a single city.")

    statuses = {}

    for city_name in city_names:
        # With --area-name (how the onboarding orchestrator calls this) both the
        # local and staged folders use the canonical study area name. Without it,
        # staged folders keep the bare city name matching BETA_10_CITIES and local
        # working outputs use the safe <Country>_<City> name.
        staged_name = area_name or city_name.strip()
        print("\n" + "#" * 80)
        print(f"# {country_name} / {city_name}")
        print("#" * 80)

        try:
            if (
                destination
                and not overwrite
                and staged_files_present(destination, staged_name)
            ):
                statuses[city_name] = (
                    f"skipped, already staged at {join_path(destination, staged_name)}"
                )
                print(statuses[city_name])
                continue

            if dry_run:
                statuses[city_name] = "would be processed"
                print(statuses[city_name])
                continue

            outputs = process_city_data(
                country_name,
                city_name,
                target_crs,
                working_dir,
                h3_root,
                output_dir,
                state=state,
                area_name=area_name,
                urban_areas_file=urban_areas_file,
                state_field=state_field,
                state_abbr_field=state_abbr_field,
                allow_multiple_features=allow_multiple_features,
                smod_level=smod_level,
            )

            if destination:
                stage_outputs(outputs, destination, staged_name)
                statuses[city_name] = f"staged at {join_path(destination, staged_name)}"
            else:
                statuses[city_name] = f"written to {outputs['output_dir']}"

        except Exception as exc:
            traceback.print_exc()
            statuses[city_name] = f"FAILED: {type(exc).__name__}: {exc}"

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for city_name, status in statuses.items():
        print(f"  {city_name}: {status}")

    return statuses


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Extracts city-clipped DEM, buildings, green spaces, soil and "
            "landcover from the H3 source tiles."
        )
    )
    parser.add_argument("country", help="Country name, as in the urban areas file.")
    parser.add_argument(
        "city", nargs="+", help="City names, as in the urban areas file."
    )
    parser.add_argument(
        "--working-dir",
        default=DEFAULT_WORKING_DIR,
        help="Directory holding the urban areas and H3 cell shapefiles. "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--h3-root",
        default=DEFAULT_H3_ROOT,
        help="Root holding one folder per H3 index. (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Local directory for results. (default: %(default)s)",
    )
    parser.add_argument(
        "--destination",
        default=None,
        help="Staging root receiving a folder per city in the BETA_10_CITIES layout.",
    )
    parser.add_argument("--state", default=None, help="State or subdivision, e.g. MO.")
    parser.add_argument(
        "--area-name",
        default=None,
        help="Name for the output and staging folders. Single city only.",
    )
    parser.add_argument(
        "--urban-areas-file",
        default=DEFAULT_URBAN_AREAS_FILE,
        help="Urban areas shapefile within --working-dir. (default: %(default)s)",
    )
    parser.add_argument(
        "--state-field",
        default=DEFAULT_STATE_FIELD,
        help="Column holding the full state name. (default: %(default)s)",
    )
    parser.add_argument(
        "--state-abbr-field",
        default=DEFAULT_STATE_ABBR_FIELD,
        help="Column holding the state abbreviation. (default: %(default)s)",
    )
    parser.add_argument(
        "--smod-level",
        default=None,
        help=f"Filter candidates by the '{DEFAULT_SMOD_FIELD}' column.",
    )
    parser.add_argument(
        "--allow-multiple-features",
        action="store_true",
        help="Permit a selection matching several features, which are merged.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess cities already staged at --destination.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which cities would be processed, then stop.",
    )
    args = parser.parse_args(argv)

    # WGS 84 / World Mercator, metres: DEM = 2 m, Green spaces = 30 m, Landcover = 30 m
    target_crs = "EPSG:3395"

    statuses = run_cities(
        args.country,
        args.city,
        target_crs,
        working_dir=args.working_dir,
        h3_root=args.h3_root,
        output_dir=args.output_dir,
        destination=args.destination,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        state=args.state,
        area_name=args.area_name,
        urban_areas_file=args.urban_areas_file,
        state_field=args.state_field,
        state_abbr_field=args.state_abbr_field,
        allow_multiple_features=args.allow_multiple_features,
        smod_level=args.smod_level,
    )

    if any(status.startswith("FAILED") for status in statuses.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

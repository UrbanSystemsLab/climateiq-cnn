"""Generates water polygons from JRC surface water tiles."""

import logging
import os

from google.cloud import storage

WATER_POLYGONS_PREFIX = "water_polygons"


JRC_TILE_URL = (
    "https://storage.googleapis.com/global-surface-water"
    "/downloads2021/occurrence/occurrence_{tile}v1_4_2021.tif"
)


JRC_OCCURRENCE_THRESHOLD = 80


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
    bbox = (
        tiff_bbox[0] - buffer,
        tiff_bbox[1] - buffer,
        tiff_bbox[2] + buffer,
        tiff_bbox[3] + buffer,
    )

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
        logging.info(
            "Downloaded JRC tile %s (%.1f MB)",
            tile_name,
            os.path.getsize(local_path) / 1e6,
        )

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
        merged_path,
        "w",
        driver="GTiff",
        height=merged_shape[0],
        width=merged_shape[1],
        count=1,
        dtype=merged_data.dtype,
        crs=merged_crs,
        transform=merged_transform,
    ) as dst:
        dst.write(merged_data, 1)
    del merged_data

    with rasterio.open(merged_path) as src:
        cropped, crop_transform = rasterio_mask(src, [bbox_geom], crop=True, nodata=0)
        cropped = cropped[0]
    os.unlink(merged_path)

    water_mask = (cropped >= JRC_OCCURRENCE_THRESHOLD).astype(np.uint8)
    water_count = int(water_mask.sum())
    logging.info(
        "JRC water pixels (>=%d%% occurrence): %d",
        JRC_OCCURRENCE_THRESHOLD,
        water_count,
    )
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
        poly_parts = [
            g for g in water_union.geoms if isinstance(g, (Polygon, MultiPolygon))
        ]
        water_union = unary_union(poly_parts) if poly_parts else None
        if water_union and water_union.geom_type == "Polygon":
            water_union = MultiPolygon([water_union])

    if water_union is None or water_union.is_empty:
        return None

    for p in tile_paths:
        if os.path.exists(p):
            os.unlink(p)

    logging.info(
        "Water polygon: %s, bounds=%s", water_union.geom_type, water_union.bounds
    )
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

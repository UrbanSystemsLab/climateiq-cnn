"""cli_preprocess_tiff_to_h3csv.py.

Preprocessing pipeline: per-city, per-scenario GeoTIFF -> merged multi-scenario H3 CSV.

Steps:
  1. For each scenario (e.g. Rainfall_Data_1.txt ... Rainfall_Data_9.txt):
       a. Download 2m mosaic GeoTIFF from climateiq-predictions bucket
       b. Resample 2m -> 10m (average)
       c. Convert to H3 cells at res 12 (jittered sampling) window-by-window
       d. Aggregate bottom-up: res 12 -> 11 -> 10 -> ... -> 6
  2. Merge all scenario DataFrames on (cell_code, h3_res) -> single CSV
     Columns: cell_code, h3_res, flood_depth_1y, flood_depth_5y, ... flood_depth_1000y

GCS path expected:
  flood_predictions/{city_folder}/{config_prefix}/{scenario_file}/
  {mosaic_filename}
  e.g. flood_predictions/Atlanta_Prediction/Atlanta_config/
  Rainfall_Data_1.txt/
  Atlanta_Prediction_mosaic_peak_wgs84.tif
"""

import gc
import os
import threading
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
)
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

import h3

# ── Constants ────────────────────────────────────────────────────────────────

WINDOW_SIZE = 1024  # pixels per processing tile
JITTER_SAMPLES = 3  # jitter grid size for res-12 (3×3 = 9 samples)
H3_BASE_RES = 12  # finest resolution — computed directly from pixels
H3_PARENT_RESOLUTIONS = list(range(11, 5, -1))  # 11→6, aggregated bottom-up
RESAMPLE_FACTOR = 5.0  # 2m -> 10m
MAX_CONCURRENT_SCENARIOS = 3  # parallel scenario processing (each uses n_workers procs)


# ── Step 1: Resample 2m -> 10m ────────────────────────────────────────────────


def resample_to_10m(src_path: str, dst_path: str) -> None:
    """Downsample a 2m GeoTIFF to ~10m using average resampling."""
    with rasterio.open(src_path) as src:
        new_w = max(1, int(src.width / RESAMPLE_FACTOR))
        new_h = max(1, int(src.height / RESAMPLE_FACTOR))

        dst_transform, width, height = calculate_default_transform(
            src.crs,
            src.crs,
            src.width,
            src.height,
            *src.bounds,
            dst_width=new_w,
            dst_height=new_h,
        )

        meta = src.meta.copy()
        meta.update(
            {
                "transform": dst_transform,
                "width": width,
                "height": height,
                "compress": "deflate",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            }
        )

        with rasterio.open(dst_path, "w", **meta) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=src.crs,
                resampling=Resampling.average,
            )


# ── Step 2: Tiff -> H3 (worker) ───────────────────────────────────────────────


def _process_window(args: Tuple) -> Optional[Tuple[dict, dict]]:
    """Worker: aggregate pixel values into H3 res-12 cells via jittered sampling.

    Returns (h3_sum, h3_count) dicts keyed by cell_code, or None if empty.
    Coarser resolutions (6-11) are derived bottom-up after all windows merge.
    """
    tiff_path, window = args
    h3_sum: dict = defaultdict(float)
    h3_count: dict = defaultdict(int)
    offsets = np.linspace(-0.4, 0.4, JITTER_SAMPLES)

    with rasterio.open(tiff_path) as src:
        data = src.read(1, window=window)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs

        mask = ~np.isnan(data)
        if nodata is not None:
            mask &= data != nodata

        if not mask.any():
            return None

        rows, cols = np.where(mask)
        vals = data[rows, cols].astype(float)
        abs_rows = rows + window.row_off
        abs_cols = cols + window.col_off

        base_xs, base_ys = xy(transform, abs_rows, abs_cols, offset="center")

        for dy in offsets:
            for dx in offsets:
                xs = [x + dx * transform.a for x in base_xs]
                ys = [y + dy * abs(transform.e) for y in base_ys]
                lons, lats = warp_transform(crs, "EPSG:4326", xs, ys)
                for lat, lon, val in zip(lats, lons, vals):
                    cell = h3.latlng_to_cell(lat, lon, H3_BASE_RES)
                    h3_sum[cell] += val
                    h3_count[cell] += 1

    return h3_sum, h3_count


def _aggregate_bottom_up(res12_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate res-12 H3 cells bottom-up through res 11 down to res 6.

    Each parent cell's value is the mean of its children's values.
    This preserves spatial variance — large cells reflect the distribution
    of their children rather than independently sampling raw pixels.
    """
    all_frames = [res12_df]
    child_df = res12_df

    for parent_res in H3_PARENT_RESOLUTIONS:
        parents = child_df["cell_code"].apply(
            lambda c, r=parent_res: h3.cell_to_parent(c, r)
        )
        agg = (
            child_df.assign(parent=parents)
            .groupby("parent", as_index=False)["value"]
            .mean()
        )
        agg = agg.rename(columns={"parent": "cell_code"})
        agg["h3_res"] = parent_res
        all_frames.append(agg)
        child_df = agg

    result = pd.concat(all_frames, ignore_index=True)
    print(
        "    [aggregate] bottom-up: "
        + ", ".join(
            f"res {r}: {int((result['h3_res'] == r).sum()):,}"
            for r in [H3_BASE_RES] + H3_PARENT_RESOLUTIONS
        )
    )
    return result


def tiff_to_h3_dataframe(tiff_10m_path: str, n_workers: int = 4) -> pd.DataFrame:
    """Convert a 10m GeoTIFF to an H3 DataFrame (res 6-12, bottom-up).

    Columns: cell_code, h3_res, value.
    """
    with rasterio.open(tiff_10m_path) as src:
        windows = [
            Window(
                col,
                row,
                min(WINDOW_SIZE, src.width - col),
                min(WINDOW_SIZE, src.height - row),
            )
            for row in range(0, src.height, WINDOW_SIZE)
            for col in range(0, src.width, WINDOW_SIZE)
        ]

    print(f"    [tiff->h3] {len(windows)} windows, {n_workers} workers")

    global_sum: dict = defaultdict(float)
    global_count: dict = defaultdict(int)

    args = [(tiff_10m_path, w) for w in windows]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for result in executor.map(_process_window, args, chunksize=8):
            if result is None:
                continue
            h3_sum, h3_count = result
            for cell, v in h3_sum.items():
                global_sum[cell] += v
                global_count[cell] += h3_count[cell]

    res12_rows = [
        {
            "cell_code": cell,
            "h3_res": H3_BASE_RES,
            "value": global_sum[cell] / global_count[cell],
        }
        for cell in global_sum
    ]
    res12_df = pd.DataFrame(res12_rows)
    print(f"    [tiff->h3] res 12: {len(res12_df):,} cells")

    return _aggregate_bottom_up(res12_df)


# ── Step 3: Process all scenarios for one city ────────────────────────────────


def get_tiff_bbox_wgs84(tiff_path: str) -> Tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) bounding box in WGS84 for a GeoTIFF."""
    with rasterio.open(tiff_path) as src:
        bounds = src.bounds
        # If already WGS84 return directly, else reproject bounds corners
        if src.crs and src.crs.to_epsg() == 4326:
            return (bounds.left, bounds.bottom, bounds.right, bounds.top)
        # Reproject the four corners to WGS84
        xs = [bounds.left, bounds.right, bounds.left, bounds.right]
        ys = [bounds.bottom, bounds.bottom, bounds.top, bounds.top]
        lons, lats = warp_transform(src.crs, "EPSG:4326", xs, ys)
        return (min(lons), min(lats), max(lons), max(lats))


def preprocess_city(
    city_config: dict,
    scenario_mapping: dict,
    work_dir: str,
    n_workers: int = 4,
    intermediates_bucket: Optional[str] = None,
    intermediates_prefix: str = "intermediates",
) -> Tuple[str, Optional[Tuple[float, float, float, float]]]:
    """Download all scenario tiffs for a city, convert to H3, merge into one CSV.

    Args:
        city_config: City entry from cities_config.yaml
            (must have 'predictions' key).
        scenario_mapping: Ordered dict mapping scenario
            filename -> column name, e.g.
            {"Rainfall_Data_1.txt": "flood_depth_1y", ...}
        work_dir: Local temp directory.
        n_workers: Parallel workers for window processing
            within each scenario.
        intermediates_bucket: If set, upload 10m TIFFs and
            per-scenario CSVs to this GCS bucket.
        intermediates_prefix: GCS prefix for intermediate
            files (default: "intermediates").

    Returns:
        (csv_path, bbox) where bbox is
        (minx, miny, maxx, maxy) in WGS84, captured from
        the first scenario tiff to use for spatial clipping.
    """
    from google.cloud import storage

    pred = city_config["predictions"]
    src_bucket_name = pred["bucket"]
    city_folder = pred["city_folder"]  # e.g. "Atlanta_Prediction"
    config_prefix = pred["config_prefix"]  # e.g. "Atlanta_config"
    mosaic_filename = pred[
        "mosaic_filename"
    ]  # e.g. "Atlanta_Prediction_mosaic_peak_wgs84.tif"
    city_name = city_config["name"]

    gcs_client = storage.Client()
    src_bucket = gcs_client.bucket(src_bucket_name)

    # Set up intermediates bucket if provided
    int_bucket = None
    if intermediates_bucket:
        int_bucket = gcs_client.bucket(intermediates_bucket)

    scenario_dfs: Dict[str, pd.DataFrame] = {}
    tiff_bbox: Optional[Tuple[float, float, float, float]] = None

    _bbox_lock = threading.Lock()

    def _process_one_scenario(scenario_file, col_name):
        nonlocal tiff_bbox

        blob_path = (
            f"flood_predictions/{city_folder}/"
            f"{config_prefix}%2F{scenario_file}/{mosaic_filename}"
        )
        print(
            f"  [preprocess] {city_name} | {col_name}"
            f" <- gs://{src_bucket_name}/{blob_path}"
        )

        local_2m = os.path.join(work_dir, f"_raw_{col_name}.tif")
        local_10m = os.path.join(work_dir, f"_10m_{col_name}.tif")

        try:
            src_bucket.blob(blob_path).download_to_filename(local_2m)
        except Exception as e:
            print(f"    WARNING: skipping {col_name} — download failed: {e}")
            return None, col_name

        with _bbox_lock:
            if tiff_bbox is None:
                tiff_bbox = get_tiff_bbox_wgs84(local_2m)
                print(f"    Tiff bbox (WGS84): {tiff_bbox}")

        print(f"    Resampling {col_name} to 10m...")
        resample_to_10m(local_2m, local_10m)
        os.remove(local_2m)

        if int_bucket is not None:
            tiff_gcs_path = (
                f"{intermediates_prefix}/{city_name}/tiffs/{col_name}_10m.tif"
            )
            print(
                f"    Uploading 10m TIFF -> gs://{intermediates_bucket}/{tiff_gcs_path}"
            )
            int_bucket.blob(tiff_gcs_path).upload_from_filename(local_10m)

        df = tiff_to_h3_dataframe(local_10m, n_workers=n_workers)
        df = df.rename(columns={"value": col_name})
        print(f"    {col_name}: {len(df):,} H3 cells")
        os.remove(local_10m)

        if int_bucket is not None:
            local_scenario_csv = os.path.join(work_dir, f"_scenario_{col_name}.csv")
            df.to_csv(local_scenario_csv, index=False)
            csv_gcs_path = f"{intermediates_prefix}/{city_name}/csvs/{col_name}.csv"
            print(
                f"    Uploading scenario CSV -> "
                f"gs://{intermediates_bucket}/{csv_gcs_path}"
            )
            int_bucket.blob(csv_gcs_path).upload_from_filename(local_scenario_csv)
            os.remove(local_scenario_csv)

        gc.collect()
        return df, col_name

    print(
        f"  [preprocess] Processing {len(scenario_mapping)} scenarios "
        f"({MAX_CONCURRENT_SCENARIOS} concurrent)..."
    )
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_SCENARIOS) as executor:
        futures = {
            executor.submit(_process_one_scenario, sf, cn): cn
            for sf, cn in scenario_mapping.items()
        }
        for future in futures:
            df, col_name = future.result()
            if df is not None:
                scenario_dfs[col_name] = df

    if not scenario_dfs:
        raise ValueError(
            f"[preprocess] No scenarios processed successfully for {city_name}"
        )

    # Merge all scenarios on (cell_code, h3_res)
    print(f"  [preprocess] Merging {len(scenario_dfs)} scenarios...")
    merged: Optional[pd.DataFrame] = None
    for col_name, df in scenario_dfs.items():
        if merged is None:
            merged = df
        else:
            merged = merged.merge(
                df[["cell_code", "h3_res", col_name]],
                on=["cell_code", "h3_res"],
                how="outer",
            )

    output_path = os.path.join(work_dir, f"{city_name}_max.csv")
    merged.to_csv(output_path, index=False)  # type: ignore[union-attr]
    row_count = len(merged)  # type: ignore[arg-type]
    print(f"  [preprocess] Saved {row_count:,} rows -> {output_path}")
    return output_path, tiff_bbox

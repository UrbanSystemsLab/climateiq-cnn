"""Predict flood depths for ALL chunks of a prediction-only study area.

Downloads feature matrices from GCS (no labels needed), runs autoregressive
call_n on each chunk at full 1000x1000 resolution, and saves georeferenced
GeoTIFFs with proper CRS to a GCS output bucket.

Each output GeoTIFF has T bands (one per timestep) with flood depth in meters.

Usage:
    CUDA_VISIBLE_DEVICES=1 python run_predict_city.py 2>&1 | tee predict_city.log
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import pathlib
import time
import copy
import numpy as np
import math
import tensorflow as tf
import keras
import rasterio
from rasterio.transform import Affine
from rasterio.merge import merge as rasterio_merge
from rasterio.warp import calculate_default_transform, reproject, Resampling

SEED = 42
keras.utils.set_random_seed(SEED)
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

from google.cloud import firestore  # type:ignore[attr-defined]
from google.cloud import storage  # type:ignore[attr-defined]

from usl_models.flood_ml.dataset import compute_dem_sink_channel, compute_flow_features
# from usl_models.flood_ml.model import (
#     FloodModel,
#     SpatialAttention,
#     FloodConvLSTM,
#     GreenAmptGate,
# )
from usl_models.flood_ml import customloss, metastore
from usl_models.shared import downloader

# =====================================================================
# CONFIGURATION — change these for your city / model
# =====================================================================
STUDY_AREA = "Chicago_Predictions"
RAINFALL_CONFIGS = [
    f"Chicago%2FRainfall_Data_{i}.txt" for i in range(1, 10)
]  # 9 scenarios

# Model checkpoint
MODEL_PATH = pathlib.Path("/scratch/hw4402/climateiq_output/semi_new_test_v03_overall_flow_2/savedmodel_256")
# Output
OUTPUT_BUCKET = "climateiq-predictions"  # GCS bucket for results
OUTPUT_PREFIX = "flood_predictions"  # prefix inside bucket
LOCAL_OUTPUT_DIR = pathlib.Path("/scratch/hw4402/predict_city_output")

# Model params
CHUNK_SIZE = 1000
BATCH_SIZE = 50
PATCH_SIZE = 256
PREDICTION_STEPS = 12
N_FLOOD_MAPS = 5
M_RAINFALL = 6
MAX_DEPTH_CLAMP = 4.0
APPLY_GREEN_AMPT = True
TEMPORAL_FEATURE_VERSION = 2
OUTPUT_CRS = "EPSG:4326"  # WGS84 for final mosaic

print(f"Study area:      {STUDY_AREA}")
print(f"Rainfall configs: {len(RAINFALL_CONFIGS)} scenarios")
print(f"Model:           {MODEL_PATH}")
print(f"Output bucket:   {OUTPUT_BUCKET}")


# =====================================================================
# LOAD MODEL — rebuild at full chunk size
# =====================================================================
def load_model(model_path):
    return tf.saved_model.load(model_path)

def predict_patch(data, model, batch_size, n_step):
    result = np.zeros((batch_size, n_step, CHUNK_SIZE, CHUNK_SIZE, 1), dtype=np.float32)
    for i in range(int(math.ceil(CHUNK_SIZE/PATCH_SIZE))):
        for j in range(int(math.ceil(CHUNK_SIZE/PATCH_SIZE))):

            if i == int(math.ceil(CHUNK_SIZE/PATCH_SIZE))-1:
                x_start = CHUNK_SIZE - PATCH_SIZE
                x_end = CHUNK_SIZE
            else: 
                x_start = i * PATCH_SIZE
                x_end = (i+1) * PATCH_SIZE
            
            if j == int(math.ceil(CHUNK_SIZE/PATCH_SIZE))-1:
                y_start = CHUNK_SIZE - PATCH_SIZE
                y_end = CHUNK_SIZE

            else:
                y_start = j * PATCH_SIZE
                y_end = (j+1) * PATCH_SIZE
            # print(x_start, x_end, y_start, y_end)

            geospatial_data = data["geospatial"][:, y_start:y_end, x_start:x_end, :]
            spatiotemporal_data = data["spatiotemporal"][:, :, y_start:y_end, x_start:x_end, :]

            y_all = model.rollout(geospatial=geospatial_data, temporal_windows=data["temporal"], spatiotemporal=spatiotemporal_data)["depth"]
            result[:, :, y_start:y_end, x_start:x_end, :] = y_all
    return tf.convert_to_tensor(result, dtype=tf.float32)


# =====================================================================
# TEMPORAL FEATURES
# =====================================================================
MAX_RAINFALL_DURATION = 864  # must match constants.MAX_RAINFALL_DURATION


def build_temporal_v2(temporal_vec):
    """Build (T, 6) temporal feature matrix matching training v2."""
    v = temporal_vec.astype(np.float32)
    T = len(v)
    cumsum = np.cumsum(v)
    total = cumsum[-1] if cumsum[-1] > 0 else 1.0
    running_max = np.maximum.accumulate(v)
    delta_rate = np.concatenate([[0.0], np.diff(v)])  # storm phase transitions
    return np.stack(
        [
            v,
            cumsum / total,
            delta_rate,           # replaces rate^2 — encodes storm onset/recession
            np.log1p(cumsum),
            running_max,
            np.arange(T, dtype=np.float32) / max(T - 1, 1),
        ],
        axis=1,
    ).astype(np.float32)


def pad_temporal_to_max(temporal_2d):
    """Pad (T, M) temporal matrix to (MAX_RAINFALL_DURATION, M) for call_n."""
    T, M = temporal_2d.shape
    if T >= MAX_RAINFALL_DURATION:
        return temporal_2d[:MAX_RAINFALL_DURATION]
    pad = np.zeros((MAX_RAINFALL_DURATION - T, M), dtype=np.float32)
    return np.concatenate([temporal_2d, pad], axis=0)


# =====================================================================
# GeoTIFF WRITING — single-band peak flood depth for GIS
# =====================================================================
def _make_profile(H, W, crs_str, x_origin, y_origin, cell_size, chunk_height):
    """Build a single-band rasterio profile with CRS and affine transform."""
    return {
        "driver": "GTiff",
        "height": H,
        "width": W,
        "count": 1,
        "dtype": "float32",
        "crs": rasterio.CRS.from_string(crs_str) if crs_str else None,
        "transform": Affine(
            cell_size, 0, x_origin,
            0, -cell_size, y_origin + chunk_height * cell_size,
        ),
        "nodata": NODATA,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }


def save_single_band_tif(data_2d, output_path, profile):
    """Save a single (H, W) array as a 1-band GeoTIFF in native CRS."""
    with rasterio.open(str(output_path), "w", **profile) as dst:
        dst.write(data_2d.astype(np.float32), 1)


def upload_single_band_to_gcs(gcs_client, bucket_name, blob_path, data_2d, profile):
    """Write a single-band GeoTIFF in native CRS to GCS."""
    buf = io.BytesIO()
    with rasterio.open(buf, "w", **profile) as dst:
        dst.write(data_2d.astype(np.float32), 1)
    buf.seek(0)
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_file(buf, content_type="image/tiff")
    print(f"    Uploaded gs://{bucket_name}/{blob_path}")


SEAM_SMOOTH_SIGMA = 8  # Gaussian sigma (pixels) for seam smoothing
NODATA = np.float32("nan")  # NaN nodata — GIS renders as transparent automatically


def mosaic_and_reproject(chunk_tifs, output_path, dst_crs=OUTPUT_CRS):
    """Merge all chunk TIFFs into one mosaic with seam smoothing, then reproject.

    1. Place all chunks on a global UTM grid using their native transforms
    2. Detect seam lines between adjacent chunks
    3. Apply Gaussian smoothing in a band around seams to hide boundary artifacts
    4. Reproject the smoothed mosaic to WGS84
    """
    from scipy.ndimage import gaussian_filter

    print(f"\n  Mosaicking {len(chunk_tifs)} chunks (seam sigma={SEAM_SMOOTH_SIGMA}px)...")

    # --- First pass: read metadata only to compute global grid ---
    tile_meta = []
    for f in chunk_tifs:
        with rasterio.open(str(f)) as src:
            tile_meta.append({
                "path": f,
                "transform": src.transform,
                "crs": src.crs,
                "height": src.height,
                "width": src.width,
            })

    src_crs = tile_meta[0]["crs"]
    pixel_w = tile_meta[0]["transform"].a
    pixel_h = tile_meta[0]["transform"].e  # negative

    # Global bounds
    lefts = [t["transform"].c for t in tile_meta]
    tops = [t["transform"].f for t in tile_meta]
    rights = [t["transform"].c + t["width"] * pixel_w for t in tile_meta]
    bottoms = [t["transform"].f + t["height"] * pixel_h for t in tile_meta]

    g_left, g_right = min(lefts), max(rights)
    g_top, g_bottom = max(tops), min(bottoms)

    out_w = int(round((g_right - g_left) / pixel_w))
    out_h = int(round((g_top - g_bottom) / abs(pixel_h)))
    mosaic_transform = Affine(pixel_w, 0, g_left, 0, pixel_h, g_top)

    print(f"  Mosaic grid: {out_w} x {out_h} pixels")

    # --- Second pass: place tiles one at a time (memory efficient) ---
    mosaic = np.full((out_h, out_w), np.nan, dtype=np.float32)

    for tm in tile_meta:
        with rasterio.open(str(tm["path"])) as src:
            data = src.read(1)
        th, tw = data.shape
        col_off = int(round((tm["transform"].c - g_left) / pixel_w))
        row_off = int(round((g_top - tm["transform"].f) / abs(pixel_h)))

        r1c, r2c = max(row_off, 0), min(row_off + th, out_h)
        c1c, c2c = max(col_off, 0), min(col_off + tw, out_w)
        tr1, tr2 = r1c - row_off, r2c - row_off
        tc1, tc2 = c1c - col_off, c2c - col_off

        chunk_data = data[tr1:tr2, tc1:tc2]
        valid = np.isfinite(chunk_data)
        mosaic[r1c:r2c, c1c:c2c][valid] = chunk_data[valid]
        del data  # free immediately

    has_data = np.isfinite(mosaic)
    n_nodata = (~has_data).sum()
    print(f"  Valid pixels: {has_data.sum():,}, nodata: {n_nodata:,} ({n_nodata/mosaic.size*100:.1f}%)")

    # --- Gaussian seam smoothing (DISABLED for beta — destroys peak values) ---
    # To re-enable: uncomment the block below
    # print(f"  Applying Gaussian smoothing (sigma={SEAM_SMOOTH_SIGMA}px)...")
    # mosaic_filled = np.where(has_data, mosaic, 0.0)
    # smoothed_vals = gaussian_filter(mosaic_filled, sigma=SEAM_SMOOTH_SIGMA)
    # del mosaic_filled
    # weight_map = has_data.astype(np.float32)
    # smoothed_weight = gaussian_filter(weight_map, sigma=SEAM_SMOOTH_SIGMA)
    # del weight_map
    # smoothed_weight = np.maximum(smoothed_weight, 1e-8)
    # mosaic[has_data] = smoothed_vals[has_data] / smoothed_weight[has_data]
    # del smoothed_vals, smoothed_weight
    # print(f"  Smoothing done.")
    del has_data

    # --- Reproject to WGS84 ---
    dst_crs_obj = rasterio.CRS.from_string(dst_crs)
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, dst_crs_obj, out_w, out_h,
        left=g_left, bottom=g_bottom, right=g_right, top=g_top,
    )

    dst_data = np.full((1, dst_h, dst_w), NODATA, dtype=np.float32)
    reproject(
        source=mosaic[np.newaxis, :, :],
        destination=dst_data,
        src_transform=mosaic_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs_obj,
        resampling=Resampling.bilinear,
        src_nodata=NODATA,
        dst_nodata=NODATA,
    )

    dst_profile = {
        "driver": "GTiff",
        "height": dst_h,
        "width": dst_w,
        "count": 1,
        "dtype": "float32",
        "crs": dst_crs_obj,
        "transform": dst_transform,
        "nodata": NODATA,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    with rasterio.open(str(output_path), "w", **dst_profile) as dst:
        dst.write(dst_data)

    print(f"  Mosaic saved: {output_path}")
    print(f"  Size: {dst_w} x {dst_h}, CRS: {dst_crs}")
    return output_path

def extract_temporal(t: int, n: int, temporal: tf.Tensor) -> tf.Tensor:
    """Generate inputs for a sliding time window of length `n`."""
    (_, D) = temporal.shape
    zeros = tf.zeros(shape=(max(n - t, 0), D))
    data = temporal[max(t - n, 0) : t]
    return tf.concat([zeros, data], axis=0)
# =====================================================================
# MAIN
# =====================================================================
def main():
    db = firestore.Client(project='climateiq-test')
    gcs_client = storage.Client(project='climateiq-test')

    # ── 1. Get study area metadata (CRS, cell_size, chunk layout) ────
    print(f"\n{'='*60}")
    print(f"  Querying Firestore for study area: {STUDY_AREA}")
    print(f"{'='*60}")

    study_area_ref = db.collection("study_areas").document(STUDY_AREA)
    study_area_doc = study_area_ref.get()
    if not study_area_doc.exists:
        raise ValueError(f"Study area '{STUDY_AREA}' not found in Firestore.")

    sa_data = study_area_doc.to_dict()
    crs_str = sa_data.get("crs", None)
    cell_size = sa_data.get("cell_size", 2.0)
    x_ll_corner = sa_data.get("x_ll_corner", 0.0)
    y_ll_corner = sa_data.get("y_ll_corner", 0.0)
    chunk_size = sa_data.get("chunk_size", CHUNK_SIZE)
    chunk_x_count = sa_data.get("chunk_x_count", 0)
    chunk_y_count = sa_data.get("chunk_y_count", 0)
    elevation_max = sa_data.get("elevation_max")
    elevation_min = sa_data.get("elevation_min")
    

    print(f"  CRS:          {crs_str}")
    print(f"  Cell size:    {cell_size}m")
    print(f"  LL corner:    ({x_ll_corner}, {y_ll_corner})")
    print(f"  Chunk size:   {chunk_size}")
    print(f"  Chunk grid:   {chunk_x_count} x {chunk_y_count}")
    print(f"  elevation max:   {elevation_max}")
    print(f"  elevation min:   {elevation_min}")

    # ── 2. Get all feature chunks (download ONCE, reuse for all scenarios) ──
    chunk_metadata = metastore.get_spatial_feature_chunk_metadata_for_prediction(
        db, STUDY_AREA
    )
    total_chunks = len(chunk_metadata)
    print(f"  Found {total_chunks} chunks")

    # Pre-download and cache all geospatial feature matrices
    print(f"\n  Downloading {total_chunks} feature matrices...")
    geo_cache = {}  # chunk_name -> (geo_tf, H, profile_args)
    for ci, chunk_meta in enumerate(chunk_metadata):
        feature_url = chunk_meta["feature_matrix_path"]
        x_idx = chunk_meta.get("x_index", 0)
        y_idx = chunk_meta.get("y_index", 0)
        chunk_name = feature_url.split("/")[-1].replace(".npy", "")

        geo_raw = downloader.download_as_array(gcs_client, feature_url).astype(
            np.float32
        )

        # scale back dem
        geo_raw[..., 0] = geo_raw[..., 0] * (elevation_max - elevation_min) + elevation_min

        # calculate other geofeatures
        dem_sink = compute_dem_sink_channel(geo_raw)
        flow_feats = compute_flow_features(geo_raw, cell_size=cell_size)
        geospatial = np.concatenate([geo_raw, dem_sink, flow_feats], axis=-1)  # (H, W, 12)
        H, W = geospatial.shape[:2]

        # Extract valid_mask: pixels outside study area have elevation == -1.0
        # (ch1/buildings is 0 for parks too, so can't use it as boundary)
        valid_mask = geo_raw[:, :, 1] == 1  # (H, W) boolean — True inside study area
        buildings_mask = geo_raw[:, :, 2] > 0.5  # (H, W) boolean — True where buildings

        geo_tf = tf.expand_dims(
            tf.convert_to_tensor(geospatial, dtype=tf.float32), 0
        )

        chunk_x_origin = chunk_meta.get(
            "x_ll_corner", x_ll_corner + x_idx * chunk_size * cell_size
        )
        chunk_y_origin = chunk_meta.get(
            "y_ll_corner", y_ll_corner + y_idx * chunk_size * cell_size
        )
        profile = _make_profile(
            H, H, crs_str or "", chunk_x_origin, chunk_y_origin, cell_size, H,
        )

        geo_cache[chunk_name] = (geo_tf, profile)
        if (ci + 1) % 20 == 0 or ci == total_chunks - 1:
            print(f"    Downloaded {ci+1}/{total_chunks}")
        

    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 3+4. Loop over all rainfall scenarios (reload model each time to free GPU memory) ──
    for scenario_idx, rainfall_config in enumerate(RAINFALL_CONFIGS):
        print(f"\n{'='*60}")
        print(f"  SCENARIO {scenario_idx+1}/{len(RAINFALL_CONFIGS)}: {rainfall_config}")
        print(f"{'='*60}")

        # Clear GPU memory from previous scenario before loading model
        tf.keras.backend.clear_session()
        model = load_model(MODEL_PATH)

        # Get rainfall metadata
        try:
            config_meta = metastore.get_temporal_feature_metadata_for_prediction(
                db, rainfall_config
            )
        except Exception as e:
            print(f"  SKIP — rainfall config not found: {e}")
            continue

        rainfall_gcs_uri = config_meta["as_vector_gcs_uri"]
        rainfall_duration = config_meta["rainfall_duration"]
        print(f"  Rainfall duration: {rainfall_duration} timesteps ({rainfall_duration * 5} min)")
        
        ###################################################################################################
        #########Change temporal data generation for new prediction model and batch prediction#############
        ###################################################################################################
        

        # Build temporal tensor for this scenario
        temporal_vec = downloader.download_as_array(gcs_client, rainfall_gcs_uri)
        if TEMPORAL_FEATURE_VERSION == 2:
            temporal_2d = build_temporal_v2(temporal_vec) # (864, 6)
        else:
            temporal_2d = np.tile(temporal_vec[:, None], (1, M_RAINFALL)).astype(np.float32)
        
        rainfall_windows = [extract_temporal(0 + k, N_FLOOD_MAPS, temporal_2d) for k in range(rainfall_duration)]

        # Scenario output subfolder — clean old files to avoid stale TIFs
        scenario_dir = LOCAL_OUTPUT_DIR / rainfall_config.replace("%2F", "_")
        if scenario_dir.exists():
            for old_tif in scenario_dir.glob("*_peak.tif"):
                old_tif.unlink()
        scenario_dir.mkdir(parents=True, exist_ok=True)

        ###################################################################################################
        #####################Change chunk prediction to batch of chunks prediction#########################
        ###################################################################################################
        scenario_start = time.time()
        geo_cache_copy = copy.deepcopy(geo_cache)
        # generate batch
        while True:
            if not geo_cache_copy:
                break
            else:
                chunk_name_list = []
                geo_tf_list = []
                profile_list = []
                for i in range(BATCH_SIZE):
                    if not geo_cache_copy:
                        break
                    chunk_name, (geo_tf, profile) = geo_cache_copy.popitem()

                    chunk_name_list.append(chunk_name)
                    geo_tf_list.append(geo_tf)
                    profile_list.append(profile)

                geospatial_tf = tf.concat(geo_tf_list, axis=0) # [B, H, W, 12]
                temporal_tf = tf.convert_to_tensor([rainfall_windows]*len(geo_tf_list), dtype=tf.float32) # [B, T_STEP, 5, 6]
                spatial_temporal_tf = tf.zeros((len(geo_tf_list), 5, 1000, 1000, 1), dtype=tf.float32)

                data = {
                    "geospatial": geospatial_tf,
                    "temporal": temporal_tf,
                    "spatiotemporal": spatial_temporal_tf
                }

                prediction_result = predict_patch(data, model, temporal_tf.shape[0], temporal_tf.shape[1]) # [B, T_STEP, H, W, 1]
                prediction_max = tf.reduce_max(prediction_result, axis=1) # [B, H, W, 1]

                # change buildings flood depth to 0
                prediction_max = prediction_max * (1 - geospatial_tf[..., 2:3])

                for chunk_idx in range(prediction_max.shape[0]):
                    chunk_name = chunk_name_list[chunk_idx]
                    profile = profile_list[chunk_idx]
                    chunk_peak_depth = prediction_max[chunk_idx, ..., 0].numpy() #[H, W]
                    valid_mask = geospatial_tf[chunk_idx, ..., 1].numpy() == 1

                    chunk_peak_depth[~valid_mask] = NODATA
                    valid_pct = valid_mask.sum() / valid_mask.size * 100

                    # save locally
                    local_path = scenario_dir / f"{STUDY_AREA}_{chunk_name}_peak.tif"
                    save_single_band_tif(chunk_peak_depth, local_path, profile)

                    print(f"Done prediction for chunk {chunk_name}, with {valid_pct}% valid pixels")

                    ########################################################

                    # Upload to GCS
                    blob_path = (
                        f"{OUTPUT_PREFIX}/{STUDY_AREA}/{rainfall_config}/{chunk_name}_peak.tif"
                    )
                    try:
                        upload_single_band_to_gcs(
                            gcs_client, OUTPUT_BUCKET, blob_path, peak_depth, profile,
                        )
                    except Exception as e:
                        print(f"    WARNING: GCS upload failed: {e}")
                    ########################################################

        print("Batch Prediction Done!")
        ################################################################################
        ################################################################################
        ################################################################################

        # ── Mosaic for this scenario ──────────────────────────────────
        chunk_tifs = sorted(scenario_dir.glob(f"{STUDY_AREA}_*_peak.tif"))
        mosaic_name = f"{STUDY_AREA}_mosaic_peak_wgs84.tif"
        mosaic_path = scenario_dir / mosaic_name
        mosaic_and_reproject(chunk_tifs, mosaic_path)

        # Upload mosaic
        mosaic_blob = f"{OUTPUT_PREFIX}/{STUDY_AREA}/{rainfall_config}/{mosaic_name}"
        try:
            bucket = gcs_client.bucket(OUTPUT_BUCKET)
            blob = bucket.blob(mosaic_blob)
            blob.upload_from_filename(str(mosaic_path), content_type="image/tiff")
            print(f"  Uploaded mosaic: gs://{OUTPUT_BUCKET}/{mosaic_blob}")
        except Exception as e:
            print(f"  WARNING: mosaic upload failed: {e}")

        # Clean up local files to free disk space
        for f in chunk_tifs:
            f.unlink(missing_ok=True)
        mosaic_path.unlink(missing_ok=True)
        print(f"  Cleaned up {len(chunk_tifs)+1} local files")
        scenario_elapsed = time.time() - scenario_start
        print(f"\n  Scenario {scenario_idx+1} done in {scenario_elapsed:.0f}s")

    # ── Final summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ALL PREDICTIONS COMPLETE")
    print(f"  Study area:  {STUDY_AREA}")
    print(f"  Scenarios:   {len(RAINFALL_CONFIGS)}")
    print(f"  Chunks/scenario: {total_chunks}")
    print(f"  Native CRS:  {crs_str}")
    print(f"  Mosaic CRS:  {OUTPUT_CRS}")
    print(f"  Local:       {LOCAL_OUTPUT_DIR}/")
    print(f"  GCS:         gs://{OUTPUT_BUCKET}/{OUTPUT_PREFIX}/{STUDY_AREA}/")
    print(f"{'='*60}")

    # # Remove local output directory now that everything is on GCS
    # import shutil
    # if LOCAL_OUTPUT_DIR.exists():
    #     shutil.rmtree(LOCAL_OUTPUT_DIR)
    #     print(f"  Removed local output dir: {LOCAL_OUTPUT_DIR}")

if __name__ == "__main__":
    main()




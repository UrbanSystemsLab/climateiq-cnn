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
import numpy as np
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

from usl_models.flood_ml.dataset import compute_dem_sink_channel
from usl_models.flood_ml.model import (
    FloodModel,
    SpatialAttention,
    FloodConvLSTM,
    GreenAmptGate,
)
from usl_models.flood_ml import customloss, metastore
from usl_models.shared import downloader

# =====================================================================
# CONFIGURATION — change these for your city / model
# =====================================================================
STUDY_AREA = "NYC_Predictions"  # Firestore study_areas document name
RAINFALL_CONFIGS = [
    f"NYC%2FRainfall_Data_{i}.txt" for i in range(1, 10)
]  # all 9 scenarios (re-run with updated rainfall data)

# Model checkpoint — new training run with gradient clipping + delta rate + feedback fix
MODEL_DIR = pathlib.Path("train_output/run_20260324-044011")
WEIGHTS_PATH = MODEL_DIR / "weights_ordered.npz"  # won't exist yet, falls back to keras
MODEL_PATH = MODEL_DIR / "best_model.keras"

# Output
OUTPUT_BUCKET = "climateiq-predictions"  # GCS bucket for results
OUTPUT_PREFIX = "flood_predictions"  # prefix inside bucket
LOCAL_OUTPUT_DIR = pathlib.Path("predict_city_output")

# Model params
CHUNK_SIZE = 1000
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
def load_model():
    """Load model weights and rebuild at CHUNK_SIZE spatial dims."""
    print(f"\nLoading model from: {MODEL_PATH}")

    if WEIGHTS_PATH.exists():
        # Use pre-extracted weights (avoids Keras load_model issues)
        model = FloodModel(
            params=FloodModel.Params(lstm_kernel_size=5),
            spatial_dims=(CHUNK_SIZE, CHUNK_SIZE),
        )
        # Build with N_FLOOD_MAPS temporal window (call uses windowed input)
        dummy = {
            "geospatial": tf.zeros((1, CHUNK_SIZE, CHUNK_SIZE, 10)),
            "spatiotemporal": tf.zeros((1, N_FLOOD_MAPS, CHUNK_SIZE, CHUNK_SIZE, 1)),
            "temporal": tf.zeros((1, N_FLOOD_MAPS, M_RAINFALL)),
        }
        _ = model._model(dummy)
        data = np.load(str(WEIGHTS_PATH))
        w_list = [data[f"w{i:02d}"] for i in range(len(data.files))]
        model._model.set_weights(w_list)
        print(f"  Loaded {len(w_list)} weights from {WEIGHTS_PATH}")
        return model
    else:
        # Fallback: load via Keras
        saved = keras.models.load_model(
            str(MODEL_PATH),
            custom_objects={
                "SpatialAttention": SpatialAttention,
                "FloodConvLSTM": FloodConvLSTM,
                "GreenAmptGate": GreenAmptGate,
                "flood_weighted_mse": customloss.flood_weighted_mse,
                "make_hybrid_loss": customloss.make_hybrid_loss,
            },
        )
        saved_weights = saved.get_weights()
        saved_cfg = saved.get_config()
        params = FloodModel.Params.from_dict(saved_cfg["params"])

        loaded = FloodConvLSTM(params=params, spatial_dims=(CHUNK_SIZE, CHUNK_SIZE))
        dummy = {
            "geospatial": tf.zeros((1, CHUNK_SIZE, CHUNK_SIZE, 10)),
            "temporal": tf.zeros((1, N_FLOOD_MAPS, M_RAINFALL)),
            "spatiotemporal": tf.zeros((1, N_FLOOD_MAPS, CHUNK_SIZE, CHUNK_SIZE, 1)),
        }
        _ = loaded(dummy, training=False)
        loaded.set_weights(saved_weights)
        del saved
        print(f"  Loaded {len(saved_weights)} weights via Keras")
        return loaded


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
# PREDICTION — use call_n then take max across timesteps
# =====================================================================
FLOOD_PERCENTILE = 75  # Use 75th percentile to filter AR accumulation spikes


def predict_chunk_peak(model, geo_tf, temporal_full, rainfall_duration, buildings_mask=None):
    """Run call_n and return flood depth (percentile over timesteps, buildings masked).

    Args:
        model: FloodModel instance.
        geo_tf: (1, H, W, 10) geospatial tensor.
        temporal_full: (1, 864, 6) padded temporal tensor.
        rainfall_duration: Number of timesteps to predict.
        buildings_mask: (H, W) boolean — True where buildings exist (set to 0).

    Returns:
        flood_depth: (H, W) numpy array — flood depth in meters.
    """
    H, W = CHUNK_SIZE, CHUNK_SIZE
    spatiotemporal = tf.zeros((1, N_FLOOD_MAPS, H, W, 1), dtype=tf.float32)

    inputs = {
        "geospatial": geo_tf,
        "temporal": temporal_full,
        "spatiotemporal": spatiotemporal,
    }

    # call_n returns (B, n, H, W)
    all_preds = model.call_n(inputs, n=rainfall_duration)  # (1, T, H, W)
    all_preds_np = np.clip(all_preds.numpy()[0], 0, MAX_DEPTH_CLAMP)  # (T, H, W)

    # 85th percentile filters AR spike artifacts while keeping real flood signal
    flood_depth = np.percentile(all_preds_np, FLOOD_PERCENTILE, axis=0)  # (H, W)

    # Zero out building footprints — buildings don't flood on the surface
    if buildings_mask is not None:
        flood_depth[buildings_mask] = 0.0

    print(
        f"    call_n done: {rainfall_duration} steps  "
        f"p{FLOOD_PERCENTILE}={flood_depth.max():.4f}m  "
        f"mean_flooded={flood_depth[flood_depth > 0.01].mean() if (flood_depth > 0.01).any() else 0:.4f}m"
    )
    return flood_depth


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


# =====================================================================
# MAIN
# =====================================================================
def main():
    db = firestore.Client()
    gcs_client = storage.Client()

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

    print(f"  CRS:          {crs_str}")
    print(f"  Cell size:    {cell_size}m")
    print(f"  LL corner:    ({x_ll_corner}, {y_ll_corner})")
    print(f"  Chunk size:   {chunk_size}")
    print(f"  Chunk grid:   {chunk_x_count} x {chunk_y_count}")

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
        dem_sink = compute_dem_sink_channel(geo_raw)
        geospatial = np.concatenate([geo_raw, dem_sink], axis=-1)  # (H, W, 10)
        H, W = geospatial.shape[:2]

        # Extract valid_mask: pixels outside study area have elevation == -1.0
        # (ch1/buildings is 0 for parks too, so can't use it as boundary)
        valid_mask = geo_raw[:, :, 0] != -1.0  # (H, W) boolean — True inside study area
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

        geo_cache[chunk_name] = (geo_tf, H, profile, x_idx, y_idx, valid_mask, buildings_mask)
        if (ci + 1) % 20 == 0 or ci == total_chunks - 1:
            print(f"    Downloaded {ci+1}/{total_chunks}")

    # ── 3. Load model ONCE ───────────────────────────────────────────
    model = load_model()
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 4. Loop over all rainfall scenarios ──────────────────────────
    for scenario_idx, rainfall_config in enumerate(RAINFALL_CONFIGS):
        print(f"\n{'='*60}")
        print(f"  SCENARIO {scenario_idx+1}/{len(RAINFALL_CONFIGS)}: {rainfall_config}")
        print(f"{'='*60}")

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

        # Build temporal tensor for this scenario
        temporal_vec = downloader.download_as_array(gcs_client, rainfall_gcs_uri)
        if TEMPORAL_FEATURE_VERSION == 2:
            temporal_2d = build_temporal_v2(temporal_vec)
        else:
            temporal_2d = np.tile(temporal_vec[:, None], (1, M_RAINFALL)).astype(np.float32)
        temporal_padded = pad_temporal_to_max(temporal_2d)
        temporal_tf = tf.expand_dims(
            tf.convert_to_tensor(temporal_padded, dtype=tf.float32), 0
        )

        # Scenario output subfolder — clean old files to avoid stale TIFs
        scenario_dir = LOCAL_OUTPUT_DIR / rainfall_config.replace("%2F", "_")
        if scenario_dir.exists():
            for old_tif in scenario_dir.glob("*_peak.tif"):
                old_tif.unlink()
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # ── Predict each chunk ────────────────────────────────────────
        scenario_start = time.time()
        for ci, chunk_name in enumerate(geo_cache):
            geo_tf, H, profile, x_idx, y_idx, valid_mask, buildings_mask = geo_cache[chunk_name]
            chunk_start = time.time()

            # Skip chunks entirely outside study area (all invalid)
            if not valid_mask.any():
                print(f"\n  [{ci+1}/{total_chunks}] {chunk_name} — SKIP (outside study area)")
                continue

            print(f"\n  [{ci+1}/{total_chunks}] {chunk_name} (x={x_idx}, y={y_idx})")

            peak_depth = predict_chunk_peak(
                model, geo_tf, temporal_tf, rainfall_duration,
                buildings_mask=buildings_mask,
            )

            # Apply valid_mask: set pixels outside study area to nodata
            peak_depth[~valid_mask] = NODATA

            elapsed = time.time() - chunk_start
            valid_pct = valid_mask.sum() / valid_mask.size * 100
            print(f"    Done in {elapsed:.1f}s  peak={peak_depth[valid_mask].max():.3f}m  valid={valid_pct:.0f}%")

            # Save locally
            local_path = scenario_dir / f"{STUDY_AREA}_{chunk_name}_peak.tif"
            save_single_band_tif(peak_depth, local_path, profile)

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


if __name__ == "__main__":
    main()

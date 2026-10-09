"""tf.data.Datasets for training FloodML model on CityCAT data."""

import logging
import random
import pathlib
import numpy as np
from scipy.ndimage import uniform_filter
from typing import Any, Iterator, Tuple

from google.cloud import firestore  # type:ignore[attr-defined]
from google.cloud import storage  # type:ignore[attr-defined]
import tensorflow as tf

from usl_models.flood_ml import constants
from usl_models.flood_ml import metastore
from usl_models.flood_ml import model
from usl_models.shared import downloader
import random

np.random.seed(constants.RANDOM_SEED)
random.seed(constants.RANDOM_SEED)

TEMPORAL_FILENAME = "temporal.npy"
FEATURE_DIRNAME = "geospatial"
LABEL_DIRNAME = "labels"
CHUNK_PER_CYCLE = 30
SAMPLE_PER_CHUNK_CYCLE = 30

def _extract_temporal(t: int, n: int, temporal: tf.Tensor) -> tf.Tensor:
    """Generate inputs for a sliding time window of length `n`."""
    (_, D) = temporal.shape
    zeros = tf.zeros(shape=(max(n - t, 0), D))
    data = temporal[max(t - n, 0) : t]
    return tf.concat([zeros, data], axis=0)

def _load_geo_with_sink(path, cell_size: float = 2.0) -> np.ndarray:
    """Load geospatial .npy and append physics channels → (H, W, 12)."""
    geo = np.load(path).astype(np.float32)
    return _compute_all_geo_features(geo, cell_size)

def _compute_all_geo_features(geo_np: np.ndarray, cell_size: float = 2.0) -> np.ndarray:
    """Append DEM sink (ch9) and flow features (ch10-11) to raw geo (H,W,9).

    Returns (H, W, 12).
    """
    sink = compute_dem_sink_channel(geo_np)  # (H, W, 1)
    flow = compute_flow_features(geo_np, cell_size)  # (H, W, 2)
    return np.concatenate([geo_np, sink, flow], axis=-1)

def compute_dem_sink_channel(geo_np: np.ndarray) -> np.ndarray:
    """Compute DEM sink depth as a 10th geospatial channel.

    Identifies pixels locally lower than their neighbourhood (DEM depressions)
    weighted by surrounding imperviousness — where water physically accumulates.
    This gives the model an explicit routing signal that raw elevation (ch0) only
    encodes implicitly, particularly important for flat cities (Phoenix PV).

    Channel layout (feature_raster_transformers.py):
        ch0 = elevation (normalised 0-1)   ch1 = valid mask   ch4 = K_s (0=impervious)

    Returns: (H, W, 1) float32, normalised [0, 1].  Append to geo_np to get (H, W, 10).
    """
    elev = geo_np[:, :, 0].astype(np.float64)
    valid = geo_np[:, :, 1].astype(np.float64)
    imperv = ((geo_np[:, :, 4] == 0) & (valid > 0)).astype(np.float64)

    neigh = 80  # ~160 m neighbourhood at 2 m/px resolution
    local_mean = uniform_filter(elev * valid, size=neigh) / np.maximum(
        uniform_filter(valid, size=neigh), 1e-9
    )
    sink = np.maximum(0.0, local_mean - elev) * valid
    local_imperv = uniform_filter(imperv, size=neigh) / np.maximum(
        uniform_filter(valid, size=neigh), 1e-9
    )
    potential = sink * (0.2 + 0.8 * local_imperv)

    max_pot = float(potential.max())
    ch = (
        (potential / max_pot).astype(np.float32)
        if max_pot > 1e-9
        else np.zeros(geo_np.shape[:2], dtype=np.float32)
    )
    return ch[:, :, np.newaxis]  # (H, W, 1)

def compute_flow_features(geo_np: np.ndarray, cell_size: float = 2.0) -> np.ndarray:
    """D8 flow direction + flow accumulation via whitebox on the chunk DEM.

    Per-chunk D8 hydrology (not stitched across chunks — water is routed within
    the chunk only). Pass ``cell_size`` in metres/pixel so flow accumulation
    scales correctly at 2 m / 5 m / 10 m resolution.

    Channel layout (appended after compute_dem_sink_channel):
        ch10: D8 flow direction — whitebox pointer codes (1,2,4,...,128)
              normalised to [0, 1] by dividing by 255.
        ch11: D8 flow accumulation — log1p-transformed upstream cell count,
              normalised to [0, 1].

    Returns: (H, W, 2) float32. Append to (H, W, 10) to get (H, W, 12).

    rasterio + whitebox are heavyweight optional deps not pinned in
    setup.py. If they aren't installed (e.g. CI test env), return a
    zero-filled placeholder so the rest of the pipeline still runs;
    real training / inference machines must have both installed.
    """
    H, W = geo_np.shape[:2]
    try:
        import tempfile
        import rasterio
        import whitebox
        from rasterio.transform import Affine
    except ImportError:
        return np.zeros((H, W, 2), dtype=np.float32)

    valid = geo_np[:, :, 1].astype(np.float32)
    elev = geo_np[:, :, 0].astype(np.float32) * valid  # zero-fill nodata

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        dem_tif = tmp_path / "dem.tif"
        tr = Affine(cell_size, 0, 0, 0, -cell_size, H * cell_size)
        with rasterio.open(
            dem_tif,
            "w",
            driver="GTiff",
            height=H,
            width=W,
            count=1,
            dtype="float32",
            transform=tr,
            crs="EPSG:3857",
        ) as f:
            f.write(elev[np.newaxis])

        wbt = whitebox.WhiteboxTools()
        wbt.set_working_dir(str(tmp_path))
        wbt.set_verbose_mode(False)

        fdir_tif = tmp_path / "fdir.tif"
        facc_tif = tmp_path / "facc.tif"
        wbt.d8_pointer(dem=str(dem_tif), output=str(fdir_tif), esri_pntr=False)
        wbt.d8_flow_accumulation(
            i=str(fdir_tif),
            output=str(facc_tif),
            out_type="cells",
            log=False,
            clip=False,
            pntr=True,
            esri_pntr=False,
        )

        with rasterio.open(fdir_tif) as f:
            fdir = f.read(1).astype(np.float32) / 255.0
        with rasterio.open(facc_tif) as f:
            facc_raw = f.read(1).astype(np.float32)
            facc_log = np.log1p(np.maximum(facc_raw, 0.0))
            facc = facc_log / max(float(facc_log.max()), 1e-9)

    return np.stack([fdir * valid, facc * valid], axis=-1).astype(np.float32)

def _build_temporal_tensor_v2(temporal_vec: np.ndarray) -> tf.Tensor:
    """Build 6-channel temporal tensor from raw rainfall vector.

    Channels (all normalised to [0,1] where applicable):
      0: instantaneous rainfall rate
      1: cumulative rainfall (normalised by total) — key: tells model how much
         water has fallen so far, even when rate=0 at peak
      2: delta rate (rate change between timesteps — storm onset/recession signal)
      3: log1p(cumulative)      (log-scale memory)
      4: running maximum rate   (peak intensity seen so far)
      5: fractional time        (t / T_max)

    BACKTRACK: set temporal_feature_version=1 in load_dataset_windowed_patches
    to revert to the original identical-channel behaviour.
    """
    v = temporal_vec.astype(np.float32)
    T = len(v)
    cumsum = np.cumsum(v)
    total = cumsum[-1] if cumsum[-1] > 0 else 1.0
    running_max = np.maximum.accumulate(v)
    delta_rate = np.concatenate([[0.0], np.diff(v)])  # storm phase transitions
    feat = np.stack(
        [
            v,  # ch0: rate
            cumsum / total,  # ch1: normalised cumulative
            delta_rate,  # ch2: delta rate (replaces rate^2)
            np.log1p(cumsum),  # ch3: log cumulative
            running_max,  # ch4: running max
            np.arange(T, dtype=np.float32) / max(T - 1, 1),  # ch5: frac time
        ],
        axis=1,
    )  # (T, 6)
    return tf.constant(feat, dtype=tf.float32)

def _extract_spatiotemporal_patch(
    t: int, n: int, labels: tf.Tensor, patch_size: int
) -> tf.Tensor:
    """Extract spatiotemporal tensor from patch labels.

    Same logic as _extract_spatiotemporal but for arbitrary patch sizes.
    """
    zeros = tf.zeros(shape=(max(n - t, 0), patch_size, patch_size), dtype=tf.float32)

    if len(labels.shape) == 3:  # (T, H, W)
        data = labels[max(t - n, 0) : t]
    else:  # (H, W) - single frame
        data = tf.zeros((0, patch_size, patch_size), dtype=tf.float32)

    return tf.expand_dims(tf.concat([zeros, data], axis=0), axis=-1)

def load_dataset_windowed_patches(
    filecache_dir: pathlib.Path,
    sim_names: list[str],
    dataset_split: str,
    patch_size: int = 256,
    stride: int | None = None,
    batch_size: int = 4,
    n_flood_maps: int = constants.N_FLOOD_MAPS,
    m_rainfall: int = constants.M_RAINFALL,
    max_chunks: int | None = None,
    max_patches_per_chunk: int | None = None,
    min_flood_fraction: float = 0.01,
    min_max_depth: float = 0.1,
    min_label_max_depth: float = 0.01,
    n_future_steps: int = 1,
    include_labels: bool = True,
    shuffle: bool = True,
    dry_timestep_fraction: float = 0.0,
    temporal_feature_version: int = 1,
) -> tf.data.Dataset:
    """Patch-based dataset loader with sliding window and flood filtering.

    Instead of using full 1000x1000 maps, this extracts smaller patches using
    a sliding window. Only patches containing meaningful flood data are yielded,
    which focuses training on relevant areas and reduces memory usage.

    Args:
        filecache_dir: Path to local filecache.
        sim_names: List of simulation names.
        dataset_split: "train", "val", or "test".
        patch_size: Size of square patches to extract (e.g., 256 -> 256x256).
        stride: Sliding window stride. Defaults to patch_size (no overlap).
                Use stride < patch_size for overlapping patches.
        batch_size: Batch size for the dataset.
        n_flood_maps: Number of historical flood maps as input.
        m_rainfall: Number of temporal features.
        max_chunks: Max spatial chunks to use (None = all).
        max_patches_per_chunk: Max patches to extract per chunk (None = all valid).
                              Useful to balance dataset across chunks.
        min_flood_fraction: Minimum fraction of pixels that must be flooded
                           (across all timesteps) to include the patch.
        min_max_depth: Minimum max depth (m) in the patch to include it.
        min_label_max_depth: Minimum max depth (m) in the target label at
                            a specific timestep. Timesteps where no pixel
                            exceeds this are skipped (sparsity filtering).
        n_future_steps: Number of consecutive future timesteps to include as
                       labels. When > 1, labels have shape (K, H, W) enabling
                       autoregressive unrolling during training. Default 1
                       yields single-step labels of shape (H, W).
        include_labels: Whether to load labels (True for training).
        shuffle: Whether to shuffle patches.
        dry_timestep_fraction: Fraction of dry timesteps (label_max <
                               min_label_max_depth) to randomly include per
                               valid patch. 0.0 = skip all dry timesteps.
                               0.12 = include ~12% of dry timesteps, teaching
                               the model to predict zero where no flooding
                               has occurred yet.
        temporal_feature_version: Controls temporal channel encoding.
            1 (default, backward-compatible): all M channels = same rainfall
              scalar (original behaviour).
            2: 6 distinct channels per timestep giving the model richer
              temporal signal — [rate, cumulative_norm, rate_sq,
              log_cumulative, running_max, fractional_time].
              Directly fixes the "rain=0 at peak" blind-spot.

    Returns:
        tf.data.Dataset yielding (inputs, labels) tuples with patch-sized tensors.
        When n_future_steps=1, labels are (H, W) and temporal is (N, M).
        When n_future_steps>1, labels are (K, H, W) and temporal is (K, N, M),
        i.e. one temporal window per future prediction step.
    """
    if stride is None:
        stride = patch_size

    H, W = constants.MAP_HEIGHT, constants.MAP_WIDTH

    def _is_valid_patch(labels_patch: np.ndarray) -> bool:
        """Check if patch contains meaningful flood data."""
        # labels_patch shape: (T, patch_h, patch_w)
        max_depth = labels_patch.max()
        if max_depth < min_max_depth:
            return False

        # Fraction of pixels that flood at any timestep
        ever_flooded = (labels_patch > 0).any(axis=0)  # (patch_h, patch_w)
        flood_fraction = ever_flooded.mean()
        if flood_fraction < min_flood_fraction:
            return False

        return True

    def _extract_patch(arr: np.ndarray, y0: int, x0: int, is_3d: bool = False):
        """Extract a patch from array. Handles 2D (H,W,C) and 3D (T,H,W)."""
        if is_3d:
            return arr[:, y0 : y0 + patch_size, x0 : x0 + patch_size]
        else:
            return arr[y0 : y0 + patch_size, x0 : x0 + patch_size, :]

    def sample_generator():
        # np.random.seed(constants.RANDOM_SEED)
        # random.seed(constants.RANDOM_SEED)
        # Step 1. Gather all chunk keys
        all_keys = []
        for sim_name in sim_names:
            sim_dir = filecache_dir / sim_name
            feature_dir = sim_dir / dataset_split / FEATURE_DIRNAME
            label_dir = (
                sim_dir / dataset_split / LABEL_DIRNAME if include_labels else None
            )

            if not feature_dir.exists():
                continue

            feature_files = {f.stem: f for f in feature_dir.glob("*.npy")}
            if include_labels:
                if not label_dir or not label_dir.exists():
                    continue
                label_files = {f.stem: f for f in label_dir.glob("*.npy")}
                stems = sorted(set(feature_files) & set(label_files))
            else:
                stems = sorted(feature_files)

            if max_chunks is not None:
                stems = stems[:max_chunks]

            for stem in stems:
                all_keys.append((sim_dir, stem))

        # Step 2. Shuffle chunk keys
        if shuffle:
            random.shuffle(all_keys)
        
        # Step 3. Get all sample index
        sample_row_count = int((constants.MAP_HEIGHT - patch_size)/stride + 1)
        sample_column_count = int((constants.MAP_WIDTH - patch_size)/stride + 1)

        full_chunk_sample_lists = []
        for i in range(len(all_keys)):
            sample_lists = []
            k = all_keys[i]
            label_path = k[0]/ dataset_split / LABEL_DIRNAME / f"{k[1]}.npy"
            geofeature_path = k[0] / dataset_split / FEATURE_DIRNAME / f"{k[1]}.npy"
            temporal_path = k[0] / TEMPORAL_FILENAME

            labels = np.load(label_path)
            time_steps = labels.shape[-1]
            samples_available_steps = time_steps - n_future_steps + 1

            # if available time steps are no more than 4, the select all of them
            if samples_available_steps <= 4:
                for j in range(samples_available_steps):
                    for r in range(sample_row_count):
                        for c in range(sample_column_count):
                            sample_lists.append([j, r, c])
            else:
                # if avaiable time steps are more than 4, then select the first two and last two time step
                for r in range(sample_row_count):
                    for c in range(sample_column_count):
                        sample_lists.append([0, r, c])
                        sample_lists.append([1, r, c])
                        sample_lists.append([samples_available_steps-2, r, c])
                        sample_lists.append([samples_available_steps-1, r, c])
                if dataset_split == "train":
                    # randomly select the middle time steps
                    samples_steps = np.array(range(2, samples_available_steps-2))
                else:
                    samples_steps = np.array(range(2, samples_available_steps-2, max(3, int(samples_available_steps/5))))
                if samples_steps.shape[0] == 0:
                    continue
                else:
                    if dataset_split == "train":
                        samples_selected = samples_steps[np.random.rand(samples_steps.shape[0])<=0.3]
                    else:
                        samples_selected = samples_steps
                    for j in samples_selected:
                        for r in range(sample_row_count):
                            for c in range(sample_column_count):
                                sample_lists.append([j, r, c])
            if max_patches_per_chunk:
                selected_patches = np.array(range(0, len(sample_lists)))
                if shuffle:
                    np.random.shuffle(selected_patches)
                chunk_sample_lists = [sample_lists[sample_idx] for sample_idx in selected_patches[0: max_patches_per_chunk]]
            else:
                chunk_sample_lists = [s for s in sample_lists]
            
            if shuffle:
                random.shuffle(chunk_sample_lists)
            
            chunk_dic = {
                "geofeature_path": geofeature_path,
                "temporal_path": temporal_path,
                "label_path": label_path,
                "sample_list": chunk_sample_lists
            }

            full_chunk_sample_lists.append(chunk_dic)

        # for c in full_chunk_sample_lists:
        #     print(c)
        while True:
            full_chunk_sample_lists = [c for c in full_chunk_sample_lists if len(c["sample_list"]) > 0]
            if len(full_chunk_sample_lists) == 0:
                break
            if shuffle:
                random.shuffle(full_chunk_sample_lists)

            # pick up 10 chunks for sample generation
            chunk_samples_list = []
            geofeature_data_list = []
            temporal_data_list = []
            label_data_list = []

            for i in range(min(CHUNK_PER_CYCLE, len(full_chunk_sample_lists))):

                chunk = full_chunk_sample_lists[i]

                # get chunk sample list
                chunk_samples = chunk["sample_list"]

                # get geofeatures
                geospatial_full = _load_geo_with_sink(chunk["geofeature_path"])  # (H, W, 10)

                # get temporal data
                temporal_vec = np.load(chunk["temporal_path"])

                if temporal_feature_version == 2:
                    temporal_tensor = _build_temporal_tensor_v2(temporal_vec)
                else:
                    # v1 (original): all M channels identical — backward compatible.
                    temporal_tensor = tf.transpose(
                        tf.tile(
                            tf.reshape(
                                tf.convert_to_tensor(temporal_vec, dtype=tf.float32),
                                (1, -1),
                            ),
                            [m_rainfall, 1],
                        )
                    )
                
                # get label
                label_arr = np.load(chunk["label_path"])  # (H, W, T)
                labels_full = np.transpose(label_arr, (2, 0, 1))  # (T, H, W)

                chunk_samples_list.append(chunk_samples)
                geofeature_data_list.append(geospatial_full)
                temporal_data_list.append(temporal_tensor)
                label_data_list.append(labels_full)
            
            # get samples, each selected chunk generate 10 samples
            for sample_idx in range(SAMPLE_PER_CHUNK_CYCLE):
                for s in range(len(geofeature_data_list)):
                    chunk_samples = full_chunk_sample_lists[s]["sample_list"]
                    if len(chunk_samples) == 0:
                        continue

                    # get temporal, row, and column idx for the sample
                    chunk_sample = full_chunk_sample_lists[s]["sample_list"].pop(0)
                    t = chunk_sample[0]
                    y0 = chunk_sample[1] * stride
                    x0 = chunk_sample[2] * stride

                    # get data for the sample
                    geospatial_full = geofeature_data_list[s]
                    temporal_tensor = temporal_data_list[s]
                    labels_full = label_data_list[s]

                    # get patch data

                    # get patch label
                    labels_patch = labels_full[:, y0:y0+patch_size, x0:x0+patch_size]
                    if not _is_valid_patch(labels_patch[t:t+n_future_steps]):
                        if dataset_split == "train":
                            random_possible = np.random.rand()
                            if random_possible > 0.15:
                                continue
                        else:
                            continue
                    labels_tensor = tf.convert_to_tensor(labels_patch, dtype=tf.float32)

                    # get patch temporal
                    if n_future_steps == 1:
                        window_temporal = _extract_temporal(
                            t, n_flood_maps, temporal_tensor
                        )
                        expected_temporal_shape = (n_flood_maps, m_rainfall)
                    else:
                        temporal_windows = [
                            _extract_temporal(t + k, n_flood_maps, temporal_tensor)
                            for k in range(n_future_steps)
                        ]
                        window_temporal = tf.stack(temporal_windows, axis=0)
                        expected_temporal_shape = (
                            n_future_steps,
                            n_flood_maps,
                            m_rainfall,
                        )
                    
                    # get patch geofeature
                    geo_patch = _extract_patch(geospatial_full, y0, x0, is_3d=False)
                    geo_tensor = tf.convert_to_tensor(geo_patch, dtype=tf.float32)

                    expected_geospatial_shape = (
                        patch_size,
                        patch_size,
                        constants.GEO_FEATURES,
                    )

                    # get the spatiotemporal historical flooding map
                    window_spatiotemporal = _extract_spatiotemporal_patch(
                                    t, n_flood_maps, labels_tensor, patch_size
                                )
                    expected_spatiotemporal_shape = (
                        n_flood_maps,
                        patch_size,
                        patch_size,
                        1,
                    )

                    # check if the sample shape is valid
                    if (
                        tuple(geo_tensor.shape) != expected_geospatial_shape
                        or tuple(window_temporal.shape) != expected_temporal_shape
                        or tuple(window_spatiotemporal.shape)
                        != expected_spatiotemporal_shape
                    ):
                        print(
                            f"WARNING: skipping malformed sample "
                            f"sim={sim_dir.name} stem={stem} t={t}: "
                            f"geospatial.shape={tuple(geo_tensor.shape)} "
                            f"(expected {expected_geospatial_shape}), "
                            f"temporal.shape={tuple(window_temporal.shape)} "
                            f"(expected {expected_temporal_shape}), "
                            f"spatiotemporal.shape={tuple(window_spatiotemporal.shape)} "
                            f"(expected {expected_spatiotemporal_shape})",
                            flush=True,
                        )
                        continue

                    window_input = model.FloodModel.Input(
                        geospatial=geo_tensor,
                        temporal=window_temporal,
                        spatiotemporal=window_spatiotemporal,
                    )
                    # generate sample
                    if include_labels:
                        if n_future_steps == 1:
                            yield window_input, labels_tensor[t]
                        else:
                            yield window_input, labels_tensor[t : t + n_future_steps]
                    else:
                        empty_label = tf.zeros(
                            (patch_size, patch_size), dtype=tf.float32
                        )
                        yield window_input, empty_label

    dataset = tf.data.Dataset.from_generator(
        generator=sample_generator,
        output_signature=(
            dict(
                geospatial=tf.TensorSpec(
                    shape=(patch_size, patch_size, constants.GEO_FEATURES),
                    dtype=tf.float32,
                ),
                temporal=tf.TensorSpec(
                    shape=(
                        (n_future_steps, n_flood_maps, m_rainfall)
                        if n_future_steps > 1
                        else (n_flood_maps, m_rainfall)
                    ),
                    dtype=tf.float32,
                ),
                spatiotemporal=tf.TensorSpec(
                    shape=(n_flood_maps, patch_size, patch_size, 1),
                    dtype=tf.float32,
                ),
            ),
            tf.TensorSpec(
                shape=(
                    (n_future_steps, patch_size, patch_size)
                    if n_future_steps > 1
                    else (patch_size, patch_size)
                ),
                dtype=tf.float32,
            ),
        ),
    )

    dataset = dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return dataset


if __name__ == "__main__":

    FILECACHE_DIR = pathlib.Path("/scratch/hw4402/climateiq_filecache_us")
    OUTPUT_DIR = pathlib.Path("/scratch/hw4402/climateiq_output")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_NAME = "09_23_VariedLength_6_steps_loss_no_peak"

    PATCH_SIZE = 256
    PATCH_STRIDE = 128
    BATCH_SIZE = 8   # reverted from 6 — batch 4 had 16x better MAE (0.012 vs 0.197)
    EPOCHS = 100  # ROUND 5 ABLATION — V3 loss only, validate before long run
    N_FLOOD_MAPS = 5
    M_RAINFALL = 6
    N_FUTURE_STEPS = 6  # 13 → 0 windows (last_t = T - n_future = 13 - 13 = 0); 12 gives last_t=1 per chunk
    DEPTH_CAP = 4.0  # metres — raised from 2.5; covers Atlanta peak (~4.4m capped at 4m)
                    # Manhattan 5-7m ponding in closed canyons above this are outliers
    DRY_TIMESTEP_FRACTION = 0.12  # calibrated from prior best run to avoid dry-step dominance

    old_sims = [
        # Atlanta (3 scenarios)
        "Atlanta-Atlanta_config/Rainfall_Data_1.txt",
        "Atlanta-Atlanta_config/Rainfall_Data_2.txt",
        "Atlanta-Atlanta_config/Rainfall_Data_5.txt",
        # Phoenix SM (4 scenarios)
        "Phoenix_SM-PHX_SM/Rainfall_Data_1.txt",
        "Phoenix_SM-PHX_SM/Rainfall_Data_5.txt",
        "Phoenix_SM-PHX_SM/Rainfall_Data_15.txt",
        "Phoenix_SM-PHX_SM/Rainfall_Data_16.txt",
        # Manhattan (8 scenarios; 7 and 15 reserved for test)
        "Manhattan-Manhattan_config/Rainfall_Data_1.txt",
        "Manhattan-Manhattan_config/Rainfall_Data_2.txt",
        "Manhattan-Manhattan_config/Rainfall_Data_5.txt",
        "Manhattan-Manhattan_config/Rainfall_Data_6.txt",
        "Manhattan-Manhattan_config/Rainfall_Data_10.txt",
        "Manhattan-Manhattan_config/Rainfall_Data_13.txt",
        "Manhattan-Manhattan_config/Rainfall_Data_16.txt",
        "Manhattan-Manhattan_config/Rainfall_Data_19.txt",
        # Phoenix PV (2 scenarios)
        "Phoenix_PV-PHX_PV/Rainfall_Data_1.txt",
        "Phoenix_PV-PHX_PV/Rainfall_Data_5.txt",]

    # ── New cities (never seen by model) ─────────────────────────────────
    new_sims = [
            # Denton TX (13 timesteps, 1 chunk each)
        "Denton_TX-Denton_config/Rainfall_Data_1.txt",
        "Denton_TX-Denton_config/Rainfall_Data_2.txt",
        "Denton_TX-Denton_config/Rainfall_Data_5.txt",
        # Lafayette LA (13 timesteps, 1 chunk each)
        "Lafayette_LA-Lafayette_config/Rainfall_Data_1.txt",
        "Lafayette_LA-Lafayette_config/Rainfall_Data_2.txt",
        "Lafayette_LA-Lafayette_config/Rainfall_Data_5.txt",
        # Boise ID (13 timesteps, 1 chunk each)
        "Boise_ID-Boise_config/Rainfall_Data_1.txt",
        "Boise_ID-Boise_config/Rainfall_Data_2.txt",
        "Boise_ID-Boise_config/Rainfall_Data_5.txt",
        # Navarre FL (13 timesteps, 1-2 chunks each)
        "Navarre_FL-Navarre_config/Rainfall_Data_1.txt",
        "Navarre_FL-Navarre_config/Rainfall_Data_2.txt",
        "Navarre_FL-Navarre_config/Rainfall_Data_5.txt",
        # New Orleans
        "New_Orleans-New_Orleans_config/Rainfall_Data_1.txt",
        "New_Orleans-New_Orleans_config/Rainfall_Data_2.txt",
        "New_Orleans-New_Orleans_config/Rainfall_Data_5.txt",]

    # BASELINE TRAINING: Only old cities for clean comparison
    # Filter to only existing sims
    old_sims = [s for s in old_sims if (FILECACHE_DIR / s).exists()]
    new_sims = [s for s in new_sims if (FILECACHE_DIR / s).exists()]
    sim_names = old_sims + new_sims  # All cities — learning new flow channels
    # print(f"BASELINE TRAINING: {len(sim_names)} old city simulations (batch_size={BATCH_SIZE})")
    # for s in sim_names:
    #     print(f"  {s}")

    split = "train"

    ds = load_dataset_windowed_patches(
        filecache_dir=FILECACHE_DIR,
        sim_names=sim_names,
        dataset_split="train",
        patch_size=PATCH_SIZE,
        stride=PATCH_STRIDE,
        batch_size=BATCH_SIZE,
        n_flood_maps=N_FLOOD_MAPS,
        m_rainfall=M_RAINFALL,
        max_patches_per_chunk=None,
        min_flood_fraction=0.12,    # ≥1% flooded pixels — excludes all-dry patches
        min_max_depth=0.05,         # lowered from 0.1 — include early onset patches
        min_label_max_depth=0.001,  # lowered from 0.01 — include very early flood onset
        n_future_steps=N_FUTURE_STEPS,
        dry_timestep_fraction=DRY_TIMESTEP_FRACTION if split == "train" else 0.0,
        shuffle=True,
        temporal_feature_version=2,
    )

    i = 0
    for d in ds:
        print("printing the sample: ", i)
        geo = d[0]["geospatial"]
        temp = d[0]["temporal"]
        spatemp = d[0]["spatiotemporal"]
        print("geofeature shape is:", geo.shape)
        print("temporal feature shape is:", temp.shape)
        print("spatiotemporal feature shape is:", spatemp.shape)
        i+=1
        # print(geo[..., 0])
        # break

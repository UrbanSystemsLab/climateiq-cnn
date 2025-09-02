"""tf.data.Datasets for training FloodML model on CityCAT data."""   

import logging
import pathlib
import random
from pathlib import Path
from typing import Any, Iterator, Tuple

import numpy as np
import tensorflow as tf

from google.cloud import firestore  # type:ignore[attr-defined]
from google.cloud import storage  # type:ignore[attr-defined]

from usl_models.flood_ml import constants, metastore, model
from usl_models.shared import downloader
import re
from . import constants



# Fallback names if not already defined elsewhere
TEMPORAL_FILENAME = globals().get("TEMPORAL_FILENAME", "temporal.npy")
FEATURE_DIRNAME = globals().get("FEATURE_DIRNAME", "geospatial")
LABEL_DIRNAME = globals().get("LABELS_DIRNAME", "labels")
SPATIOTEMP_DIR = globals().get("SPATIOTEMP_DIR", "spatiotemporal")
# add near your imports


_STEM_RE = re.compile(r"^(scaled_|scaled_chunk_)", re.IGNORECASE)

def _norm_stem(name: str) -> str:
    """
    Normalize any feature filename to 'x_y' stem.
    Examples:
      'scaled_chunk_3_5' -> '3_5'
      'scaled_7_2'       -> '7_2'
      '1_4'              -> '1_4'
    """
    stem = _STEM_RE.sub("", name)
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
        return f"{parts[-2]}_{parts[-1]}"
    return stem



def _decode_once(name: str) -> str:
    return name.replace("%2F", "/", 1) if "%2F" in name else name


def _is_prediction_name(name: str) -> bool:
    """Only check the sim family (before the rainfall filename)."""
    base = name.split("%2F", 1)[0].split("/", 1)[0]
    return "prediction" in base.lower()


def _np_f32(p: Path) -> np.ndarray:
    return np.load(str(p)).astype(np.float32, copy=False)


def _temporal_matrix(
    temporal_vec: np.ndarray, m_rainfall: int, n_flood_maps: int
) -> tf.Tensor:
    """
    Return [n_flood_maps, m_rainfall].
    If temporal_vec has length T != n_flood_maps, we pad/truncate on the rows.
    """
    if temporal_vec.ndim > 1:
        temporal_vec = temporal_vec.reshape(-1)
    t = tf.convert_to_tensor(temporal_vec, dtype=tf.float32)  # [T]
    t = tf.tile(tf.reshape(t, (-1, 1)), [1, m_rainfall])  # [T, m_rainfall]
    t = t[:n_flood_maps, :]  # trim to n_flood_maps rows
    rows = tf.shape(t)[0]
    need = n_flood_maps - rows
    # Pad rows if needed (dynamic-safe)
    t = tf.cond(
        need > 0,
        lambda: tf.concat([t, tf.zeros((need, m_rainfall), tf.float32)], axis=0),
        lambda: t,
    )
    return t  # shape is [n_flood_maps, m_rainfall]


def _conform_geo(arr: np.ndarray) -> np.ndarray:
    """Return [MAP_H, MAP_W, GEO_FEATURES], cropping/padding H/W/C as needed."""
    H, W, C = constants.MAP_HEIGHT, constants.MAP_WIDTH, constants.GEO_FEATURES
    if arr.ndim == 2:
        arr = arr[..., None]  # [H,W,1]
    out = np.zeros((H, W, arr.shape[2]), np.float32)
    hh, ww = min(H, arr.shape[0]), min(W, arr.shape[1])
    out[:hh, :ww, : arr.shape[2]] = arr[:hh, :ww, :]
    if out.shape[2] > C:
        out = out[:, :, :C]
    elif out.shape[2] < C:
        pad = np.zeros((H, W, C - out.shape[2]), np.float32)
        out = np.concatenate([out, pad], axis=-1)
    return out


def _maybe_spatiotemporal(sim_dir: Path, split: str, n_flood_maps: int) -> tf.Tensor:
    """Optional cached spatiotemporal slices; otherwise zeros."""
    sdir = sim_dir / split / SPATIOTEMP_DIR
    if not sdir.exists():
        return tf.zeros(
            (n_flood_maps, constants.MAP_HEIGHT, constants.MAP_WIDTH, 1), tf.float32
        )
    files = sorted(sdir.glob("*.npy"))
    if not files:
        return tf.zeros(
            (n_flood_maps, constants.MAP_HEIGHT, constants.MAP_WIDTH, 1), tf.float32
        )
    mats = [_np_f32(p) for p in files]
    mats = [m if m.ndim == 3 else m[..., None] for m in mats]  # [H,W,1+] each
    sp = np.stack(mats, axis=0)  # [N,H,W,C]
    # conform to (n_flood_maps, H, W, 1)
    if sp.shape[-1] != 1:  # collapse extra channels if present
        sp = sp[..., :1]
    if sp.shape[0] > n_flood_maps:
        sp = sp[:n_flood_maps]
    elif sp.shape[0] < n_flood_maps:
        pad = np.zeros((n_flood_maps - sp.shape[0],) + sp.shape[1:], np.float32)
        sp = np.concatenate([sp, pad], axis=0)
    # H/W crop/pad if needed
    H, W = constants.MAP_HEIGHT, constants.MAP_WIDTH
    if sp.shape[1] != H or sp.shape[2] != W:
        fixed = np.zeros((sp.shape[0], H, W, 1), np.float32)
        hh, ww = min(H, sp.shape[1]), min(W, sp.shape[2])
        fixed[:, :hh, :ww, :] = sp[:, :hh, :ww, :1]
        sp = fixed
    return tf.convert_to_tensor(sp, tf.float32)


def load_dataset(
    sim_names: list[str],
    dataset_split: str,
    batch_size: int = 4,
    n_flood_maps: int = constants.N_FLOOD_MAPS,
    m_rainfall: int = constants.M_RAINFALL,
    max_chunks: int | None = None,
    firestore_client: firestore.Client = None,
    storage_client: storage.Client | None = None,
) -> tf.data.Dataset:
    """Creates a dataset which generates chunks for the flood model.

    This dataset produces the input for `model.call_n`.
    For training with teacher-forcing, `load_dataset_windowed` should be used instead.
    The examples are generated from multiple simulations.
    The dataset iteratively yields examples read from Google Cloud Storage to avoid
    pulling all examples into memory at once.

    Args:
      sim_names: The simulation names, e.g. ["Manhattan-config_v1/Rainfall_Data_1.txt"]
      dataset_split: Which dataset split to load: train, val, and/or test.
      batch_size: Size of batches yielded by the dataset. Approximate memory
                  usage is 10GB * batch_size during training.
      n_flood_maps: The number of flood maps in each example.
      m_rainfall: The width of the temporal rainfall tensor.
      max_chunks: The maximum number of examples to yield from the dataset.
                  If `None` (default) yields all examples from the simulations.
      firestore_client: The client to use when interacting with Firestore.
      storage_client: The client to use when interacting with Cloud Storage.
    """
    firestore_client = firestore_client or firestore.Client()
    storage_client = storage_client or storage.Client()

    def generator():
        """Generator for producing full inputs and labels."""
        for sim_name in sim_names:
            for model_input, labels in _iter_model_inputs(
                firestore_client,
                storage_client,
                sim_name,
                n_flood_maps,
                m_rainfall,
                max_chunks,
                dataset_split,
            ):
                yield model_input, labels

    # Create the dataset for this simulation
    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_signature=(
            dict(
                geospatial=_geospatial_dataset_signature(),
                temporal=_temporal_dataset_signature(m_rainfall),
                spatiotemporal=_spatiotemporal_dataset_signature(n_flood_maps),
            ),
            tf.TensorSpec(
                shape=(None, constants.MAP_HEIGHT, constants.MAP_WIDTH),
                dtype=tf.float32,
            ),
        ),
    )
    # If no batch specified, do not batch the dataset, which is required
    # for generating data for batch prediction in VertexAI.
    if batch_size:
        dataset = dataset.batch(batch_size)
    return dataset


def load_dataset_windowed(
    sim_names: list[str],
    dataset_split: str,
    batch_size: int = 4,
    n_flood_maps: int = constants.N_FLOOD_MAPS,
    m_rainfall: int = constants.M_RAINFALL,
    max_chunks: int | None = None,
    firestore_client: firestore.Client | None = None,
    storage_client: storage.Client | None = None,
) -> tf.data.Dataset:
    """Creates a dataset which generates chunks for flood model training.

    This dataset produces the input for `model.call` and should only be used
    for training, since it uses labels.
    For getting data to input into `model.call_n`, use `load_dataset` instead.
    The examples are generated from multiple simulations.
    They are windowed for training on next-map prediction.
    The dataset iteratively yields examples read from Google Cloud Storage to avoid
    pulling all examples into memory at once.

    Args:
      sim_names: The simulation names, e.g. ["Manhattan-config_v1/Rainfall_Data_1.txt"]
      dataset_split: Which dataset split to load: train, val, and/or test.
      batch_size: Size of batches yielded by the dataset. Approximate memory
                  usage is 10GB * batch_size during training.
      n_flood_maps: The number of flood maps in each example.
      m_rainfall: The width of the temporal rainfall tensor.
      max_chunks: The maximum number of examples to yield from the dataset.
                  If `None` (default) yields all examples from the simulations.
      firestore_client: The client to use when interacting with Firestore.
      storage_client: The client to use when interacting with Cloud Storage.
    """
    firestore_client = firestore_client or firestore.Client()
    storage_client = storage_client or storage.Client()

    def generator():
        """Windowed generator for teacher-forcing training."""
        for sim_name in sim_names:
            for model_input, labels in _iter_model_inputs(
                firestore_client,
                storage_client,
                sim_name,
                n_flood_maps,
                m_rainfall,
                max_chunks,
                dataset_split,
            ):
                for window_input, window_label in _generate_windows(
                    model_input, labels, n_flood_maps
                ):
                    yield (window_input, window_label)

    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_signature=(
            dict(
                geospatial=tf.TensorSpec(
                    shape=(
                        constants.MAP_HEIGHT,
                        constants.MAP_WIDTH,
                        constants.GEO_FEATURES,
                    ),
                    dtype=tf.float32,
                ),
                temporal=tf.TensorSpec(
                    shape=(n_flood_maps, m_rainfall),
                    dtype=tf.float32,
                ),
                spatiotemporal=tf.TensorSpec(
                    shape=(n_flood_maps, constants.MAP_HEIGHT, constants.MAP_WIDTH, 1),
                    dtype=tf.float32,
                ),
            ),
            tf.TensorSpec(
                shape=(constants.MAP_HEIGHT, constants.MAP_WIDTH), dtype=tf.float32
            ),
        ),
    )
    # If no batch specified, do not batch the dataset, which is required
    # for generating data for batch prediction in VertexAI.
    if batch_size:
        dataset = dataset.batch(batch_size)
    return dataset


def load_prediction_dataset(
    study_area: str,
    city_cat_config: str,
    batch_size: int = 4,
    n_flood_maps: int = constants.N_FLOOD_MAPS,
    m_rainfall: int = constants.M_RAINFALL,
    max_chunks: int | None = None,
    firestore_client: firestore.Client | None = None,
    storage_client: storage.Client | None = None,
) -> tf.data.Dataset:
    """Creates a prediction dataset which generates chunks for flood model prediction.

    This dataset produces the input for `model.call_n`.
    For training with teacher-forcing, `load_dataset_windowed` should be used instead.
    The examples are generated from multiple simulations.
    The dataset iteratively yields examples read from Google Cloud Storage to avoid
    pulling all examples into memory at once.

    Args:
      study_area: The study area to build geospatial tensors for.
      city_cat_config: The config to build a temporal tensors for.
      batch_size: Size of batches yielded by the dataset. Approximate memory
                  usage is 10GB * batch_size during training.
      n_flood_maps: The number of flood maps in each example.
      m_rainfall: The width of the temporal rainfall tensor.
      max_chunks: The maximum number of examples to yield from the dataset.
                  If `None` (default) yields all examples from the simulations.
      firestore_client: The client to use when interacting with Firestore.
      storage_client: The client to use when interacting with Cloud Storage.
    """
    firestore_client = firestore_client or firestore.Client()
    storage_client = storage_client or storage.Client()

    def generator():
        """Generator for producing full inputs from study area."""
        for model_input, metadata in _iter_model_inputs_for_prediction(
            firestore_client,
            storage_client,
            city_cat_config,
            study_area,
            n_flood_maps,
            m_rainfall,
            max_chunks,
        ):
            yield model_input, metadata

    # Create the dataset for this simulation
    prediction_dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_signature=(
            dict(
                geospatial=_geospatial_dataset_signature(),
                temporal=_temporal_dataset_signature(m_rainfall),
                spatiotemporal=_spatiotemporal_dataset_signature(n_flood_maps),
            ),
            dict(
                feature_chunk=tf.TensorSpec(shape=(), dtype=tf.string),
                rainfall=tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        ),
    )
    # If no batch specified, do not batch the dataset, which is required
    # for generating data for batch prediction in VertexAI.
    if batch_size:
        prediction_dataset = prediction_dataset.batch(batch_size)
    return prediction_dataset


def _generate_windows(
    model_input: model.FloodModel.Input, labels: tf.Tensor, n_flood_maps: int
) -> Iterator[Tuple[model.FloodModel.Input, tf.Tensor]]:
    """Generate inputs for a sliding time window of length n_flood_maps timesteps."""
    (T_max, H, W, *_) = labels.shape
    for t in range(T_max):
        window_input = model.FloodModel.Input(
            geospatial=model_input["geospatial"],
            temporal=_extract_temporal(t, n_flood_maps, model_input["temporal"]),
            spatiotemporal=_extract_spatiotemporal(t, n_flood_maps, labels),
        )
        yield window_input, labels[t]


def _extract_temporal(t: int, n: int, temporal: tf.Tensor) -> tf.Tensor:
    """Generate inputs for a sliding time window of length `n`."""
    (_, D) = temporal.shape
    zeros = tf.zeros(shape=(max(n - t, 0), D))
    data = temporal[max(t - n, 0) : t]
    return tf.concat([zeros, data], axis=0)


def _extract_spatiotemporal(t: int, n: int, labels: tf.Tensor) -> tf.Tensor:
    """Extracts spatiotemporal tensor from labeled data.

    Args:
      t: The current timestep.
      n: The window size.
      labels: The training labels.

    Returns:
      The slice labels[t-n: t] with zero padding so the output tensor is always
      of shape (n, H, W, 1).
    """
    (_, H, W, *_) = labels.shape
    zeros = tf.zeros(shape=(max(n - t, 0), H, W), dtype=tf.float32)
    data = labels[max(t - n, 0) : t]
    return tf.expand_dims(tf.concat([zeros, data], axis=0), axis=-1)


def _iter_model_inputs(
    firestore_client: firestore.Client,
    storage_client: storage.Client,
    sim_name: str,
    n_flood_maps: int,
    m_rainfall: int,
    max_chunks: int | None,
    dataset_split: str,
) -> Iterator[Tuple[model.FloodModel.Input, tf.Tensor]]:
    """Yields model inputs for each spatial chunk in the simulation."""
    temporal, _ = _generate_temporal_tensor(
        metastore.get_temporal_feature_metadata(firestore_client, sim_name),
        storage_client,
        sim_name,
        m_rainfall,
    )
    feature_label_gen = _iter_geo_feature_label_tensors(
        firestore_client, storage_client, sim_name, dataset_split
    )

    for i, (geospatial, labels) in enumerate(feature_label_gen):
        if max_chunks is not None and i >= max_chunks:
            return

        model_input = model.FloodModel.Input(
            temporal=temporal,
            geospatial=geospatial,
            spatiotemporal=tf.zeros(
                shape=(
                    n_flood_maps,
                    constants.MAP_HEIGHT,
                    constants.MAP_WIDTH,
                    1,
                )
            ),
        )
        yield model_input, labels


def _generate_temporal_tensor(
    temporal_metadata: dict[str, Any],
    storage_client: storage.Client,
    sim_name: str,
    m_rainfall: int,
) -> tuple[tf.Tensor, int]:
    """Creates a temporal tensor from the numpy array stored in GCS."""
    gcs_url = temporal_metadata["as_vector_gcs_uri"]
    logging.info("Retrieving temporal features from %s.", gcs_url)

    temporal_vector = downloader.download_as_tensor(storage_client, gcs_url)
    temporal_vector = tf.transpose(
        tf.tile(tf.reshape(temporal_vector, (1, len(temporal_vector))), [m_rainfall, 1])
    )
    return temporal_vector, temporal_metadata["rainfall_duration"]


def _iter_geo_feature_label_tensors(
    firestore_client: firestore.Client,
    storage_client: storage.Client,
    sim_name: str,
    dataset_split: str,
) -> Iterator[tuple[tf.Tensor, tf.Tensor]]:
    """Yields feature and label tensors from chunks stored in GCS."""
    feature_label_metadata = metastore.get_spatial_feature_and_label_chunk_metadata(
        firestore_client, sim_name, dataset_split
    )

    random.shuffle(feature_label_metadata)

    for feature_metadata, label_metadata in feature_label_metadata:
        feature_url = feature_metadata["feature_matrix_path"]
        label_url = label_metadata["gcs_uri"]

        logging.info(
            "Retrieving features from %s and labels from %s", feature_url, label_url
        )
        feature_tensor = downloader.download_as_tensor(storage_client, feature_url)
        label_tensor = downloader.download_as_tensor(storage_client, label_url)

        reshaped_label_tensor = tf.transpose(label_tensor, perm=[2, 0, 1])
        yield feature_tensor, reshaped_label_tensor


def _iter_model_inputs_for_prediction(
    firestore_client: firestore.Client,
    storage_client: storage.Client,
    city_cat_config: str,
    study_area_name: str,
    n_flood_maps: int,
    m_rainfall: int,
    max_chunks: int | None,
) -> Iterator[Tuple[model.FloodModel.Input, dict]]:
    """Yields model inputs for each spatial chunk in the simulation."""
    temporal, rainfall = _generate_temporal_tensor(
        metastore.get_temporal_feature_metadata_for_prediction(
            firestore_client, city_cat_config
        ),
        storage_client,
        city_cat_config,
        m_rainfall,
    )

    for feature_tensor, chunk_name in _iter_study_area_tensors(
        firestore_client, storage_client, study_area_name
    ):
        metadata = {"feature_chunk": chunk_name, "rainfall": rainfall}

        model_input = model.FloodModel.Input(
            temporal=temporal,
            geospatial=feature_tensor,
            spatiotemporal=tf.zeros(
                shape=(
                    n_flood_maps,
                    constants.MAP_HEIGHT,
                    constants.MAP_WIDTH,
                    1,
                )
            ),
        )
        yield model_input, metadata


def _iter_study_area_tensors(
    firestore_client: firestore.Client,
    storage_client: storage.Client,
    study_area_name: str,
) -> Iterator[Tuple[tf.Tensor, str]]:
    """Yields feature tensors from chunks stored in GCS."""
    all_feature_metadata = metastore.get_spatial_feature_chunk_metadata_for_prediction(
        firestore_client, study_area_name
    )

    random.shuffle(all_feature_metadata)

    for feature_metadata in all_feature_metadata:
        feature_url = feature_metadata["feature_matrix_path"]
        chunk_name = feature_url.split("/")[-1]

        logging.info("Retrieving features from %s ", feature_url)
        feature_tensor = downloader.download_as_tensor(storage_client, feature_url)
        yield feature_tensor, chunk_name


def _geospatial_dataset_signature() -> tf.TensorSpec:
    return tf.TensorSpec(
        shape=(
            constants.MAP_HEIGHT,
            constants.MAP_WIDTH,
            constants.GEO_FEATURES,
        ),
        dtype=tf.float32,
    )


def _temporal_dataset_signature(m_rainfall: int) -> tf.TensorSpec:
    return tf.TensorSpec(
        shape=(constants.MAX_RAINFALL_DURATION, m_rainfall),
        dtype=tf.float32,
    )


def _spatiotemporal_dataset_signature(n_flood_maps: int) -> tf.TensorSpec:
    return tf.TensorSpec(
        shape=(
            n_flood_maps,
            constants.MAP_HEIGHT,
            constants.MAP_WIDTH,
            1,
        ),
        dtype=tf.float32,
    )


def download_dataset(
    sim_names: list[str],
    output_path: pathlib.Path,
    firestore_client: firestore.Client | None = None,
    storage_client: storage.Client | None = None,
    dataset_splits: list[str] | None = None,
) -> None:
    """Download simulations from GCS to a local filecache.

    - Regular sims (train/val/test): temporal + (features, labels) per split.
    - Sims whose study-area contains 'Prediction': temporal + features-only per split,
      with normalized 'x_y.npy' filenames.
    """
    firestore_client = firestore_client or firestore.Client()
    storage_client = storage_client or storage.Client()
    dataset_splits = dataset_splits or ["train"]

    for sim_name in sim_names:
        sim_path = output_path / sim_name
        sim_path.mkdir(parents=True, exist_ok=True)

        # Parse "<StudyArea>-<Config>/Rainfall_Data_X.txt"
        left, right = (sim_name.split("/", 1) + [""])[:2]
        if "-" in left:
            study_area, city_cat_config = left.split("-", 1)
        else:
            study_area, city_cat_config = left, ""

        is_prediction = "prediction" in study_area.lower()

        if is_prediction:
            # ---- temporal.npy ----
            base_study = study_area.replace("Prediction", "").rstrip("_-")
            base_sim_for_temporal = f"{base_study}-{city_cat_config}"
            if right:
                base_sim_for_temporal = f"{base_sim_for_temporal}/{right}"

            temporal_meta = metastore.get_temporal_feature_metadata(
                firestore_client, base_sim_for_temporal
            )
            temporal_array = downloader.download_as_array(
                storage_client, temporal_meta["as_vector_gcs_uri"]
            )
            np.save(sim_path / TEMPORAL_FILENAME, temporal_array)

            # ---- geospatial features (no labels), normalized stem per split ----
            feature_metas = list(
                metastore.get_spatial_feature_chunk_metadata_for_prediction(
                    firestore_client, study_area
                )
            )
            for split in dataset_splits:
                features_dir = sim_path / split / FEATURE_DIRNAME  # 'geospatial'
                features_dir.mkdir(parents=True, exist_ok=True)
                for fm in feature_metas:
                    arr = downloader.download_as_array(
                        storage_client, fm["feature_matrix_path"]
                    )
                    raw_stem = pathlib.Path(fm["feature_matrix_path"]).stem
                    stem = _norm_stem(raw_stem)  # <-- normalize to x_y
                    np.save(features_dir / f"{stem}.npy", arr)

            print(
                f"  ✓ [Prediction] Saved temporal.npy and {len(feature_metas)} normalized feature chunks "
                f"to each of splits {dataset_splits} for {sim_name} (no labels)."
            )
            continue

        # =========================
        # TRAIN / VAL / TEST (features + labels)
        # =========================
        temporal_meta = metastore.get_temporal_feature_metadata(
            firestore_client, sim_name
        )
        temporal_array = downloader.download_as_array(
            storage_client, temporal_meta["as_vector_gcs_uri"]
        )
        np.save(sim_path / TEMPORAL_FILENAME, temporal_array)

        for split in dataset_splits:
            feature_label_metadata = (
                metastore.get_spatial_feature_and_label_chunk_metadata(
                    firestore_client, sim_name, split
                )
            )

            features_dir = sim_path / split / FEATURE_DIRNAME  # 'geospatial'
            labels_dir = sim_path / split / LABEL_DIRNAME  # 'labels'
            features_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            for feature_meta, label_meta in feature_label_metadata:
                stem = f"{feature_meta['x_index']}_{feature_meta['y_index']}"
                feature_arr = downloader.download_as_array(
                    storage_client, feature_meta["feature_matrix_path"]
                )
                label_arr = downloader.download_as_array(
                    storage_client, label_meta["gcs_uri"]
                )
                np.save(features_dir / f"{stem}.npy", feature_arr)
                np.save(labels_dir / f"{stem}.npy", label_arr)



def _iter_model_inputs_cached(
    sim_dir: pathlib.Path,
    dataset_split: str,
    n_flood_maps: int,
    m_rainfall: int,
    max_chunks: int | None,
    shuffle: bool = True,
) -> Iterator[Tuple[model.FloodModel.Input, tf.Tensor]]:
    """Yields model inputs from arrays cached on disk."""

    temporal_path = sim_dir / TEMPORAL_FILENAME
    temporal_vec = np.load(temporal_path)
    temporal_tensor = tf.transpose(
        tf.tile(
            tf.reshape(tf.convert_to_tensor(temporal_vec, dtype=tf.float32), (1, -1)),
            [m_rainfall, 1],
        )
    )

    feature_dir = sim_dir / dataset_split / FEATURE_DIRNAME
    label_dir = sim_dir / dataset_split / LABEL_DIRNAME

    feature_files = {f.stem: f for f in feature_dir.glob("*.npy")}
    label_files = {f.stem: f for f in label_dir.glob("*.npy")}
    stems = sorted(set(feature_files) & set(label_files))
    if shuffle:
        random.shuffle(stems)

    for i, stem in enumerate(stems):
        if max_chunks is not None and i >= max_chunks:
            return
        geospatial = tf.convert_to_tensor(
            np.load(feature_files[stem]), dtype=tf.float32
        )
        label_arr = np.load(label_files[stem])
        label_tensor = tf.transpose(
            tf.convert_to_tensor(label_arr, dtype=tf.float32), perm=[2, 0, 1]
        )
        model_input = model.FloodModel.Input(
            temporal=temporal_tensor,
            geospatial=geospatial,
            spatiotemporal=tf.zeros(
                shape=(n_flood_maps, constants.MAP_HEIGHT, constants.MAP_WIDTH, 1)
            ),
        )
        yield model_input, label_tensor


def load_dataset_cached(
    filecache_dir: Path,
    sim_names: list[str],
    dataset_split: str,  # 'train' | 'val' | 'test'
    batch_size: int = 4,
    n_flood_maps: int = constants.N_FLOOD_MAPS,
    m_rainfall: int = constants.M_RAINFALL,
    max_chunks: int | None = None,
    strict_labels: bool = False,  # True: raise if labels missing or multi-channel
) -> tf.data.Dataset:
    """Load cached dataset; train/val yields (inputs, labels), test yields inputs.
       Feature↔label matching uses normalized stems so 'scaled_chunk_3_5.npy' matches '3_5.npy'."""
    include_labels = dataset_split.lower() in ("train", "val")

    def gen():
        for sim_name in sim_names:
            sim_dir = filecache_dir / (
                sim_name.replace("%2F", "/", 1) if "%2F" in sim_name else sim_name
            )
            if not sim_dir.exists():
                raise FileNotFoundError(f"Simulation folder not found: {sim_dir}")

            # ---- temporal -> [n_flood_maps, m_rainfall]
            t_path = sim_dir / TEMPORAL_FILENAME
            if not t_path.exists():
                raise FileNotFoundError(f"Missing {TEMPORAL_FILENAME} at {t_path}")
            t_vec = np.load(str(t_path))
            if t_vec.ndim > 1:
                t_vec = t_vec.reshape(-1)
            t_tf = tf.convert_to_tensor(t_vec, tf.float32)
            t_tf = tf.tile(tf.reshape(t_tf, (-1, 1)), [1, m_rainfall])[:n_flood_maps, :]
            if int(t_tf.shape[0]) < n_flood_maps:
                t_tf = tf.concat(
                    [t_tf, tf.zeros((n_flood_maps - int(t_tf.shape[0]), m_rainfall), tf.float32)],
                    axis=0,
                )
            t_tf = tf.ensure_shape(t_tf, (n_flood_maps, m_rainfall))

            # ---- geospatial chunks
            feat_dir = sim_dir / dataset_split / FEATURE_DIRNAME
            if not feat_dir.exists():
                raise FileNotFoundError(f"Missing features dir: {feat_dir}")
            feat_files = sorted(feat_dir.glob("*.npy"))
            if not feat_files:
                raise FileNotFoundError(f"No *.npy feature files under: {feat_dir}")

            # ---- labels (by normalized stem)
            label_map = {}
            labels_dir = sim_dir / dataset_split / "labels"
            if include_labels and labels_dir.exists():
                label_map = {_norm_stem(p.stem): p for p in labels_dir.glob("*.npy")}
            elif include_labels and strict_labels:
                raise FileNotFoundError(f"Missing labels dir: {labels_dir}")

            # ---- spatiotemporal (optional) -> [n_flood_maps,H,W,1]
            sp = None
            sp_dir = sim_dir / dataset_split / "spatiotemporal"
            if sp_dir.exists():
                sfiles = sorted(sp_dir.glob("*.npy"))
                if sfiles:
                    mats = [np.load(str(p)).astype(np.float32, copy=False) for p in sfiles]
                    mats = [m if m.ndim == 3 else m[..., None] for m in mats]  # [H,W,1+]
                    sp_np = np.stack(mats, axis=0)  # [N,H,W,C]
                    H, W = constants.MAP_HEIGHT, constants.MAP_WIDTH
                    if sp_np.shape[-1] != 1:
                        sp_np = sp_np[..., :1]
                    if sp_np.shape[0] > n_flood_maps:
                        sp_np = sp_np[:n_flood_maps]
                    elif sp_np.shape[0] < n_flood_maps:
                        pad = np.zeros((n_flood_maps - sp_np.shape[0],) + sp_np.shape[1:], np.float32)
                        sp_np = np.concatenate([sp_np, pad], axis=0)
                    if sp_np.shape[1] != H or sp_np.shape[2] != W:
                        fixed = np.zeros((sp_np.shape[0], H, W, 1), np.float32)
                        hh, ww = min(H, sp_np.shape[1]), min(W, sp_np.shape[2])
                        fixed[:, :hh, :ww, :] = sp_np[:, :hh, :ww, :1]
                        sp_np = fixed
                    sp = tf.convert_to_tensor(sp_np, tf.float32)
            if sp is None:
                sp = tf.zeros((n_flood_maps, constants.MAP_HEIGHT, constants.MAP_WIDTH, 1), tf.float32)

            for i, fpath in enumerate(feat_files):
                if max_chunks is not None and i >= max_chunks:
                    break

                # geospatial -> [H,W,G]
                geo = np.load(str(fpath)).astype(np.float32, copy=False)
                if geo.ndim == 2:
                    geo = geo[..., None]
                H, W, G = constants.MAP_HEIGHT, constants.MAP_WIDTH, constants.GEO_FEATURES
                out = np.zeros((H, W, geo.shape[2]), np.float32)
                hh, ww = min(H, geo.shape[0]), min(W, geo.shape[1])
                out[:hh, :ww, : geo.shape[2]] = geo[:hh, :ww, :]
                if out.shape[2] > G:
                    out = out[:, :, :G]
                elif out.shape[2] < G:
                    pad = np.zeros((H, W, G - out.shape[2]), np.float32)
                    out = np.concatenate([out, pad], axis=-1)
                geo_tf = tf.convert_to_tensor(out, tf.float32)
                geo_tf = tf.ensure_shape(geo_tf, (H, W, G))

                inputs = {"temporal": t_tf, "geospatial": geo_tf, "spatiotemporal": sp}

                if include_labels:
                    key = _norm_stem(fpath.stem)  # <-- normalized lookup
                    lp = label_map.get(key)
                    if lp is None and strict_labels:
                        raise FileNotFoundError(f"No label for chunk '{key}' under {labels_dir}")
                    if lp is not None:
                        lab = np.load(str(lp)).astype(np.float32, copy=False)
                        if lab.ndim == 3:
                            if lab.shape[2] == 1:
                                lab = lab[:, :, 0]
                            else:
                                if strict_labels:
                                    raise ValueError(
                                        f"Label has {lab.shape[2]} channels; expected single-channel."
                                    )
                                lab = lab.max(axis=2)  # union across channels
                        elif lab.ndim != 2:
                            raise ValueError(f"Unexpected label ndim={lab.ndim}; expected 2 or 3.")
                    else:
                        lab = np.zeros((H, W), np.float32)

                    lab_fixed = np.zeros((H, W), np.float32)
                    hh, ww = min(H, lab.shape[0]), min(W, lab.shape[1])
                    lab_fixed[:hh, :ww] = lab[:hh, :ww]
                    lab_tf = tf.convert_to_tensor(lab_fixed, tf.float32)
                    lab_tf = tf.ensure_shape(lab_tf, (H, W))
                    yield inputs, lab_tf
                else:
                    yield inputs

    input_sig = dict(
        geospatial=_geospatial_dataset_signature(),
        temporal=tf.TensorSpec((n_flood_maps, m_rainfall), tf.float32),
        spatiotemporal=_spatiotemporal_dataset_signature(n_flood_maps),
    )
    output_sig = (
        (input_sig, tf.TensorSpec((constants.MAP_HEIGHT, constants.MAP_WIDTH), tf.float32))
        if include_labels
        else input_sig
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_sig)
    return ds.batch(batch_size) if batch_size else ds



def load_prediction_dataset_cached(
    filecache_dir: Path,
    sim_names: list[str],
    dataset_split: str = "predict",  # 'train' | 'val' | 'test' | 'predict'
    batch_size: int = 4,
    n_flood_maps: int = constants.N_FLOOD_MAPS,
    m_rainfall: int = constants.M_RAINFALL,
    max_chunks: int | None = None,
) -> tf.data.Dataset:
    """Load cached prediction dataset: features only, no labels. Returns (inputs, metadata).
       Normalizes feature chunk IDs in metadata to 'x_y'."""
    def gen():
        for sim_name in sim_names:
            sim_dir = filecache_dir / (
                sim_name.replace("%2F", "/", 1) if "%2F" in sim_name else sim_name
            )
            if not sim_dir.exists():
                raise FileNotFoundError(f"Simulation folder not found: {sim_dir}")

            # temporal -> [n_flood_maps, m_rainfall]
            t_path = sim_dir / TEMPORAL_FILENAME
            if not t_path.exists():
                raise FileNotFoundError(f"Missing {TEMPORAL_FILENAME} at {t_path}")
            t_vec = np.load(t_path)
            if t_vec.ndim > 1:
                t_vec = t_vec.reshape(-1)
            t_tf = tf.convert_to_tensor(t_vec, tf.float32)
            t_tf = tf.tile(tf.reshape(t_tf, (-1, 1)), [1, m_rainfall])[:n_flood_maps, :]
            if int(t_tf.shape[0]) < n_flood_maps:
                t_tf = tf.concat(
                    [t_tf, tf.zeros((n_flood_maps - int(t_tf.shape[0]), m_rainfall), tf.float32)],
                    axis=0,
                )
            t_tf = tf.ensure_shape(t_tf, (n_flood_maps, m_rainfall))

            # geospatial chunks
            feat_dir = sim_dir / dataset_split / FEATURE_DIRNAME
            if not feat_dir.exists():
                raise FileNotFoundError(f"Missing features dir: {feat_dir}")
            feat_files = sorted(feat_dir.glob("*.npy"))
            if not feat_files:
                raise FileNotFoundError(f"No *.npy feature files under: {feat_dir}")

            # optional spatiotemporal
            sp = None
            sp_dir = sim_dir / dataset_split / "spatiotemporal"
            if sp_dir.exists():
                sfiles = sorted(sp_dir.glob("*.npy"))
                if sfiles:
                    mats = [np.load(str(p)).astype(np.float32, copy=False) for p in sfiles]
                    mats = [m if m.ndim == 3 else m[..., None] for m in mats]
                    sp_np = np.stack(mats, axis=0)
                    H, W = constants.MAP_HEIGHT, constants.MAP_WIDTH
                    if sp_np.shape[-1] != 1:
                        sp_np = sp_np[..., :1]
                    if sp_np.shape[0] > n_flood_maps:
                        sp_np = sp_np[:n_flood_maps]
                    elif sp_np.shape[0] < n_flood_maps:
                        pad = np.zeros((n_flood_maps - sp_np.shape[0],) + sp_np.shape[1:], np.float32)
                        sp_np = np.concatenate([sp_np, pad], axis=0)
                    if sp_np.shape[1] != H or sp_np.shape[2] != W:
                        fixed = np.zeros((sp_np.shape[0], H, W, 1), np.float32)
                        hh, ww = min(H, sp_np.shape[1]), min(W, sp_np.shape[2])
                        fixed[:, :hh, :ww, :] = sp_np[:, :hh, :ww, :1]
                        sp_np = fixed
                    sp = tf.convert_to_tensor(sp_np, tf.float32)
            if sp is None:
                sp = tf.zeros((n_flood_maps, constants.MAP_HEIGHT, constants.MAP_WIDTH, 1), tf.float32)

            for i, fpath in enumerate(feat_files):
                if max_chunks is not None and i >= max_chunks:
                    break
                geo = np.load(str(fpath)).astype(np.float32, copy=False)
                if geo.ndim == 2:
                    geo = geo[..., None]
                H, W, G = constants.MAP_HEIGHT, constants.MAP_WIDTH, constants.GEO_FEATURES
                out = np.zeros((H, W, geo.shape[2]), np.float32)
                hh, ww = min(H, geo.shape[0]), min(W, geo.shape[1])
                out[:hh, :ww, : geo.shape[2]] = geo[:hh, :ww, :]
                if out.shape[2] > G:
                    out = out[:, :, :G]
                elif out.shape[2] < G:
                    pad = np.zeros((H, W, G - out.shape[2]), np.float32)
                    out = np.concatenate([out, pad], axis=-1)
                geo_tf = tf.convert_to_tensor(out, tf.float32)
                geo_tf = tf.ensure_shape(geo_tf, (H, W, G))

                inputs = {"temporal": t_tf, "geospatial": geo_tf, "spatiotemporal": sp}
                meta = {
                    "feature_chunk": tf.convert_to_tensor(_norm_stem(fpath.stem), tf.string),  # normalized
                    "rainfall": tf.convert_to_tensor(int(t_vec.shape[0]), tf.int32),
                }
                yield inputs, meta

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            dict(
                geospatial=_geospatial_dataset_signature(),
                temporal=tf.TensorSpec((n_flood_maps, m_rainfall), tf.float32),
                spatiotemporal=_spatiotemporal_dataset_signature(n_flood_maps),
            ),
            dict(
                feature_chunk=tf.TensorSpec((), tf.string),
                rainfall=tf.TensorSpec((), tf.int32),
            ),
        ),
    )
    return ds.batch(batch_size) if batch_size else ds


def load_dataset_cached_auto(
    *,
    filecache_dir: Path,
    sim_names: list[str],
    dataset_split: str,
    batch_size: int = 4,
    n_flood_maps: int = constants.N_FLOOD_MAPS,
    m_rainfall: int = constants.M_RAINFALL,
    max_chunks: int | None = None,
    strict_labels: bool = False,
):
    """
    Auto-select:
      - ALL names contain 'prediction' → load_prediction_dataset_cached (features-only)
      - NONE contain 'prediction'      → load_dataset_cached (features + labels for train/val)
      - MIXED                          → raise ValueError (avoid mismatched outputs)
    """
    if not sim_names:
        raise ValueError("sim_names is empty.")

    flags = [_is_prediction_name(s) for s in sim_names]
    any_pred, all_pred = any(flags), all(flags)

    if all_pred:
        return load_prediction_dataset_cached(
            filecache_dir=filecache_dir,
            sim_names=sim_names,
            dataset_split=dataset_split,
            batch_size=batch_size,
            n_flood_maps=n_flood_maps,
            m_rainfall=m_rainfall,
            max_chunks=max_chunks,
        )

    if not any_pred:
        return load_dataset_cached(
            filecache_dir=filecache_dir,
            sim_names=sim_names,
            dataset_split=dataset_split,
            batch_size=batch_size,
            n_flood_maps=n_flood_maps,
            m_rainfall=m_rainfall,
            max_chunks=max_chunks,
            strict_labels=strict_labels,
        )

    raise ValueError(
        "Mixed prediction and non-prediction sim_names. "
        "Split the list and call separately so outputs have consistent structure."
    )


def load_dataset_windowed_cached(
    filecache_dir: pathlib.Path,
    sim_names: list[str],
    dataset_split: str,
    batch_size: int = 4,
    n_flood_maps: int = constants.N_FLOOD_MAPS,
    m_rainfall: int = constants.M_RAINFALL,
    max_chunks: int | None = None,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Creates a windowed dataset from locally cached simulations."""

    def generator():
        for sim_name in sim_names:
            sim_dir = filecache_dir / sim_name
            for model_input, labels in _iter_model_inputs_cached(
                sim_dir,
                dataset_split,
                n_flood_maps,
                m_rainfall,
                max_chunks,
                shuffle,
            ):
                for window_input, window_label in _generate_windows(
                    model_input, labels, n_flood_maps
                ):
                    yield window_input, window_label

    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_signature=(
            dict(
                geospatial=_geospatial_dataset_signature(),
                temporal=tf.TensorSpec(
                    shape=(n_flood_maps, m_rainfall), dtype=tf.float32
                ),
                spatiotemporal=tf.TensorSpec(
                    shape=(n_flood_maps, constants.MAP_HEIGHT, constants.MAP_WIDTH, 1),
                    dtype=tf.float32,
                ),
            ),
            tf.TensorSpec(
                shape=(constants.MAP_HEIGHT, constants.MAP_WIDTH), dtype=tf.float32
            ),
        ),
    )
    if batch_size:
        dataset = dataset.batch(batch_size)
    return dataset



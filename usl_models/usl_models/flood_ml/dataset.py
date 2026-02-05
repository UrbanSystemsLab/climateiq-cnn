"""tf.data.Datasets for training FloodML model on CityCAT data."""

import logging
import random
import pathlib
import numpy as np
from typing import Any, Iterator, Tuple

from google.cloud import firestore  # type:ignore[attr-defined]
from google.cloud import storage  # type:ignore[attr-defined]
import tensorflow as tf

from usl_models.flood_ml import constants
from usl_models.flood_ml import metastore
from usl_models.flood_ml import model
from usl_models.shared import downloader

TEMPORAL_FILENAME = "temporal.npy"
FEATURE_DIRNAME = "geospatial"
LABEL_DIRNAME = "labels"


def _engineer_temporal_features(rain_1d: np.ndarray, m_rainfall: int) -> np.ndarray:
    """Engineers m_rainfall temporal features from a 1D rainfall signal.

    The raw rainfall vector is a single channel.  Tiling it m_rainfall times
    (the previous approach) produces identical columns with zero additional
    information.  Instead we derive distinct temporal features so the ConvLSTM
    can learn richer temporal dynamics.

    Features produced (in order up to m_rainfall):
      0: Raw rainfall intensity
      1: Normalised cumulative rainfall (fraction of total up to t)
      2: Rainfall rate-of-change (finite difference)
      3: Rolling mean rainfall (5-step / 25-min window)
      4: Remaining rainfall (fraction of total from t onward)
      5: Binary rain indicator (1 where rainfall > 0)

    Args:
        rain_1d: 1D array of shape (T_MAX,) — raw rainfall per timestep.
        m_rainfall: Number of temporal feature columns to produce.

    Returns:
        Array of shape (T_MAX, m_rainfall).
    """
    features: list[np.ndarray] = []

    # 0: raw rainfall
    features.append(rain_1d.copy())

    # 1: cumulative rainfall, normalised by total
    cum = np.cumsum(rain_1d)
    total = cum[-1] if cum[-1] > 0 else 1.0
    features.append(cum / total)

    # 2: rate-of-change (first difference)
    diff = np.zeros_like(rain_1d)
    diff[1:] = rain_1d[1:] - rain_1d[:-1]
    features.append(diff)

    # 3: rolling mean (window = 5 timesteps)
    features.append(
        np.convolve(rain_1d, np.ones(5, dtype=rain_1d.dtype) / 5, mode="same")
    )

    # 4: remaining rainfall (reverse cumsum, normalised)
    rem = np.cumsum(rain_1d[::-1])[::-1].copy()
    rem_total = rem[0] if rem[0] > 0 else 1.0
    features.append(rem / rem_total)

    # 5: binary indicator
    features.append((rain_1d > 0).astype(rain_1d.dtype))

    return np.stack(features[:m_rainfall], axis=-1)


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
      sim_names:  The simulation names, e.g. ["Manhattan-config_v1/Rainfall_Data_1.txt"]
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
    """Creates a temporal tensor with engineered features from GCS."""
    gcs_url = temporal_metadata["as_vector_gcs_uri"]
    logging.info("Retrieving temporal features from %s.", gcs_url)

    temporal_vector = downloader.download_as_tensor(storage_client, gcs_url).numpy()
    temporal_features = _engineer_temporal_features(temporal_vector, m_rainfall)
    temporal_tensor = tf.convert_to_tensor(temporal_features, dtype=tf.float32)
    return temporal_tensor, temporal_metadata["rainfall_duration"]


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
    include_labels: bool = True,
    rainfall_sim_name: str | None = None,
    allow_missing_sim: bool = False,
) -> None:
    """Download simulations from GCS to a local filecache.

    Args:
      sim_names: For training: simulation names like "City-Config/Rainfall_Data_22.txt".
                 For prediction: study area names like "Atlanta_Prediction".
      output_path: Directory where the cached files should be stored.
      firestore_client: Client used to query Firestore metadata.
      storage_client: Client used to download objects from GCS.
      dataset_splits: Dataset splits to download (train/val/test).
      include_labels: If False (prediction), download features only.
      rainfall_sim_name: Required when include_labels=False.
      allow_missing_sim: If T, sim_names that aren't found in Firestore (study areas).
    """
    firestore_client = firestore_client or firestore.Client()
    storage_client = storage_client or storage.Client()

    if include_labels:
        dataset_splits = dataset_splits or ["train"]
    else:
        dataset_splits = []  # Not needed for prediction

    for sim_name in sim_names:
        if include_labels:
            # TRAINING MODE
            sim_path = output_path / sim_name
            sim_path.mkdir(parents=True, exist_ok=True)

            # Download temporal vector
            try:
                temporal_meta = metastore.get_temporal_feature_metadata(
                    firestore_client, sim_name
                )
            except ValueError as e:
                if not allow_missing_sim:
                    raise ValueError(f"No such simulation {sim_name} found.") from e
                continue

            temporal_array = downloader.download_as_array(
                storage_client, temporal_meta["as_vector_gcs_uri"]
            )
            np.save(sim_path / TEMPORAL_FILENAME, temporal_array)

            # Download feature + label chunks
            for split in dataset_splits:
                feature_label_metadata = (
                    metastore.get_spatial_feature_and_label_chunk_metadata(
                        firestore_client, sim_name, split
                    )
                )
                features_dir = sim_path / split / FEATURE_DIRNAME
                labels_dir = sim_path / split / LABEL_DIRNAME
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

        else:
            # PREDICTION MODE
            if not rainfall_sim_name:
                raise ValueError("rainfall_sim_name must be provided for prediction")

            # Preserve `.txt` extension for rainfall sim
            rainfall_stem = pathlib.Path(rainfall_sim_name).name
            sim_path = output_path / sim_name / rainfall_stem
            sim_path.mkdir(parents=True, exist_ok=True)

            # Download temporal vector from rainfall sim
            temporal_meta = metastore.get_temporal_feature_metadata(
                firestore_client, rainfall_sim_name
            )
            temporal_array = downloader.download_as_array(
                storage_client, temporal_meta["as_vector_gcs_uri"]
            )
            np.save(sim_path / TEMPORAL_FILENAME, temporal_array)

            # Download features from study area
            features_dir = sim_path / FEATURE_DIRNAME
            features_dir.mkdir(parents=True, exist_ok=True)

            feature_metadata_list = (
                metastore.get_spatial_feature_chunk_metadata_for_prediction(
                    firestore_client, sim_name
                )
            )

            for feature_meta in feature_metadata_list:
                stem = (
                    f"{feature_meta['x_index']}_{feature_meta['y_index']}"
                    if "x_index" in feature_meta and "y_index" in feature_meta
                    else pathlib.PurePosixPath(feature_meta["feature_matrix_path"]).stem
                )
                feature_arr = downloader.download_as_array(
                    storage_client, feature_meta["feature_matrix_path"]
                )
                np.save(features_dir / f"{stem}.npy", feature_arr)


def _iter_model_inputs_cached(
    sim_dir: pathlib.Path,
    dataset_split: str,
    n_flood_maps: int,
    m_rainfall: int,
    max_chunks: int | None,
    include_labels: bool = True,
    shuffle: bool = True,
) -> Iterator[Tuple[model.FloodModel.Input, tf.Tensor | None, str]]:
    """Yields model inputs, optional labels, chunk name from arrays cached on disk."""
    temporal_path = sim_dir / TEMPORAL_FILENAME
    temporal_vec = np.load(temporal_path)
    temporal_tensor = tf.convert_to_tensor(
        _engineer_temporal_features(temporal_vec, m_rainfall), dtype=tf.float32
    )

    feature_dir = sim_dir / dataset_split / FEATURE_DIRNAME
    feature_files = {f.stem: f for f in feature_dir.glob("*.npy")}

    if include_labels:
        label_dir = sim_dir / dataset_split / LABEL_DIRNAME
        label_files = {f.stem: f for f in label_dir.glob("*.npy")}
        stems = sorted(set(feature_files) & set(label_files))
    else:
        stems = sorted(feature_files)

    if shuffle:
        random.shuffle(stems)

    for i, stem in enumerate(stems):
        if max_chunks is not None and i >= max_chunks:
            return

        geospatial = tf.convert_to_tensor(
            np.load(feature_files[stem]), dtype=tf.float32
        )
        model_input = model.FloodModel.Input(
            temporal=temporal_tensor,
            geospatial=geospatial,
            spatiotemporal=tf.zeros(
                shape=(n_flood_maps, constants.MAP_HEIGHT, constants.MAP_WIDTH, 1),
                dtype=tf.float32,
            ),
        )

        if include_labels:
            label_arr = np.load(label_files[stem])
            label_tensor = tf.transpose(
                tf.convert_to_tensor(label_arr, dtype=tf.float32), perm=[2, 0, 1]
            )
            yield model_input, label_tensor, stem
        else:
            yield model_input, None, stem


def load_dataset_cached(
    filecache_dir: pathlib.Path,
    sim_names: list[str],
    dataset_split: str | None = "train",
    batch_size: int = 4,
    n_flood_maps: int = constants.N_FLOOD_MAPS,
    m_rainfall: int = constants.M_RAINFALL,
    max_chunks: int | None = None,
    shuffle: bool = True,
    include_labels: bool = True,
    rainfall_sim_name: str | None = None,
) -> tf.data.Dataset:
    """Loads data from local filecache for training or prediction.

    Always yields (model_input, label_tensor, metadata_dict).

    - Training mode (include_labels=True): real labels (T,H,W) and metadata.
    - Prediction mode (include_labels=False): dummy zero labels and metadata.
    """

    def generator():
        for sim_name in sim_names:
            if include_labels:
                # === TRAINING MODE ===
                sim_dir = filecache_dir / sim_name
                for model_input, labels, chunk_name in _iter_model_inputs_cached(
                    sim_dir=sim_dir,
                    dataset_split=dataset_split or "train",
                    n_flood_maps=n_flood_maps,
                    m_rainfall=m_rainfall,
                    max_chunks=max_chunks,
                    include_labels=True,
                    shuffle=shuffle,
                ):
                    metadata = {
                        "feature_chunk": tf.convert_to_tensor(
                            chunk_name, dtype=tf.string
                        ),
                        "rainfall": tf.convert_to_tensor(
                            labels.shape[0], dtype=tf.int32
                        ),
                    }
                    yield model_input, labels, metadata

            else:
                # === PREDICTION MODE ===
                if rainfall_sim_name is None:
                    raise ValueError("Missing rainfall_sim_name for prediction")

                sim_dir = (
                    filecache_dir / sim_name / pathlib.Path(rainfall_sim_name).name
                )

                temporal_path = sim_dir / TEMPORAL_FILENAME
                temporal_vec = np.load(temporal_path)
                rainfall = int(temporal_vec.shape[0])  # simple metadata

                temporal_tensor = tf.convert_to_tensor(
                    _engineer_temporal_features(temporal_vec, m_rainfall),
                    dtype=tf.float32,
                )

                feature_dir = sim_dir / FEATURE_DIRNAME
                feature_files = sorted(feature_dir.glob("*.npy"))
                if shuffle:
                    random.shuffle(feature_files)

                for i, f in enumerate(feature_files):
                    if max_chunks is not None and i >= max_chunks:
                        return

                    geospatial = tf.convert_to_tensor(np.load(f), dtype=tf.float32)
                    model_input = model.FloodModel.Input(
                        temporal=temporal_tensor,
                        geospatial=geospatial,
                        spatiotemporal=tf.zeros(
                            shape=(
                                n_flood_maps,
                                constants.MAP_HEIGHT,
                                constants.MAP_WIDTH,
                                1,
                            ),
                            dtype=tf.float32,
                        ),
                    )

                    metadata = {
                        "feature_chunk": tf.convert_to_tensor(f.stem, dtype=tf.string),
                        "rainfall": tf.convert_to_tensor(rainfall, dtype=tf.int32),
                    }

                    # Dummy zero labels for consistency
                    dummy_label = tf.zeros(
                        (constants.MAP_HEIGHT, constants.MAP_WIDTH), dtype=tf.float32
                    )
                    yield model_input, dummy_label, metadata

    # === Dataset signature (updated to support sequence labels) ===
    if include_labels:
        label_signature = tf.TensorSpec(
            shape=(None, constants.MAP_HEIGHT, constants.MAP_WIDTH),
            dtype=tf.float32,
        )
    else:
        label_signature = tf.TensorSpec(
            shape=(constants.MAP_HEIGHT, constants.MAP_WIDTH),
            dtype=tf.float32,
        )

    output_signature = (
        dict(
            geospatial=_geospatial_dataset_signature(),
            temporal=_temporal_dataset_signature(m_rainfall),
            spatiotemporal=_spatiotemporal_dataset_signature(n_flood_maps),
        ),
        label_signature,
        dict(
            feature_chunk=tf.TensorSpec(shape=(), dtype=tf.string),
            rainfall=tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_signature=output_signature,
    )

    if batch_size:
        dataset = dataset.batch(batch_size)
    return dataset


def load_dataset_windowed_cached(
    filecache_dir: pathlib.Path,
    sim_names: list[str],
    dataset_split: str,
    batch_size: int = 4,
    n_flood_maps: int = constants.N_FLOOD_MAPS,
    m_rainfall: int = constants.M_RAINFALL,
    max_chunks: int | None = None,
    include_labels: bool = True,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Memory-efficient streaming dataset loader.

    Loads windowed flood simulations lazily from the local cache, one
    chunk at a time.
    Only keeps ~`batch_size` examples in memory at once.
    Shuffle happens at the (sim_name, chunk) key level before data loading.
    include_labels respected for training / prediction use cases.
    """

    def sample_generator():
        # Step 1. Gather all chunk keys (sim_dir, stem)
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

        # Step 2. Shuffle keys before loading data
        if shuffle:
            random.shuffle(all_keys)

        # Step 3. Iterate over shuffled keys lazily
        for sim_dir, stem in all_keys:
            temporal_path = sim_dir / TEMPORAL_FILENAME
            if not temporal_path.exists():
                continue

            temporal_vec = np.load(temporal_path)
            temporal_tensor = tf.convert_to_tensor(
                _engineer_temporal_features(temporal_vec, m_rainfall),
                dtype=tf.float32,
            )

            feature_path = sim_dir / dataset_split / FEATURE_DIRNAME / f"{stem}.npy"
            if not feature_path.exists():
                continue

            geospatial = tf.convert_to_tensor(np.load(feature_path), dtype=tf.float32)

            if include_labels:
                label_path = sim_dir / dataset_split / LABEL_DIRNAME / f"{stem}.npy"
                if not label_path.exists():
                    continue
                label_arr = np.load(label_path)
                labels = tf.transpose(
                    tf.convert_to_tensor(label_arr, dtype=tf.float32),
                    perm=[2, 0, 1],
                )
            else:
                labels = tf.zeros(
                    (constants.MAP_HEIGHT, constants.MAP_WIDTH),
                    dtype=tf.float32,
                )

            model_input = model.FloodModel.Input(
                temporal=temporal_tensor,
                geospatial=geospatial,
                spatiotemporal=tf.zeros(
                    shape=(n_flood_maps, constants.MAP_HEIGHT, constants.MAP_WIDTH, 1),
                    dtype=tf.float32,
                ),
            )

            for window_input, window_label in _generate_windows(
                model_input, labels, n_flood_maps
            ):
                yield window_input, window_label

    # Step 4. Wrap in tf.data.Dataset
    dataset = tf.data.Dataset.from_generator(
        generator=sample_generator,
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
                shape=(constants.MAP_HEIGHT, constants.MAP_WIDTH),
                dtype=tf.float32,
            ),
        ),
    )

    # Step 6. Batch + Prefetch
    dataset = dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    # Optionally: cache small metadata if you repeatedly iterate over the same dataset
    # dataset = dataset.cache()  # use only if memory allows
    return dataset


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
    include_labels: bool = True,
    shuffle: bool = True,
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
        include_labels: Whether to load labels (True for training).
        shuffle: Whether to shuffle patches.

    Returns:
        tf.data.Dataset yielding (inputs, labels) tuples with patch-sized tensors.
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

        # Step 3. Process each chunk and extract patches
        for sim_dir, stem in all_keys:
            temporal_path = sim_dir / TEMPORAL_FILENAME
            if not temporal_path.exists():
                continue

            temporal_vec = np.load(temporal_path)
            temporal_features = _engineer_temporal_features(temporal_vec, m_rainfall)

            feature_path = sim_dir / dataset_split / FEATURE_DIRNAME / f"{stem}.npy"
            if not feature_path.exists():
                continue

            geospatial_full = np.load(feature_path)  # (H, W, C)

            if include_labels:
                label_path = sim_dir / dataset_split / LABEL_DIRNAME / f"{stem}.npy"
                if not label_path.exists():
                    continue
                label_arr = np.load(label_path)  # (H, W, T)
                labels_full = np.transpose(label_arr, (2, 0, 1))  # (T, H, W)
            else:
                labels_full = None

            # Step 4. Sliding window to find valid patches
            valid_patches = []
            for y0 in range(0, H - patch_size + 1, stride):
                for x0 in range(0, W - patch_size + 1, stride):
                    if include_labels:
                        y1, x1 = y0 + patch_size, x0 + patch_size
                        labels_patch = labels_full[:, y0:y1, x0:x1]
                        if _is_valid_patch(labels_patch):
                            valid_patches.append((y0, x0))
                    else:
                        # Without labels, include all patches
                        valid_patches.append((y0, x0))

            # Limit patches per chunk if specified
            exceeds_limit = (
                max_patches_per_chunk is not None
                and len(valid_patches) > max_patches_per_chunk
            )
            if exceeds_limit:
                if shuffle:
                    random.shuffle(valid_patches)
                valid_patches = valid_patches[:max_patches_per_chunk]

            # Step 5. Yield temporal windows for each valid patch
            for y0, x0 in valid_patches:
                geo_patch = _extract_patch(geospatial_full, y0, x0, is_3d=False)
                geo_tensor = tf.convert_to_tensor(geo_patch, dtype=tf.float32)

                if include_labels:
                    y1, x1 = y0 + patch_size, x0 + patch_size
                    labels_patch = labels_full[:, y0:y1, x0:x1]
                    labels_tensor = tf.convert_to_tensor(labels_patch, dtype=tf.float32)
                else:
                    labels_tensor = tf.zeros((patch_size, patch_size), dtype=tf.float32)

                temporal_tensor = tf.convert_to_tensor(
                    temporal_features, dtype=tf.float32
                )

                # Generate temporal windows (same as _generate_windows but for patches)
                num_timesteps = labels_tensor.shape[0] if include_labels else 1
                for t in range(num_timesteps):
                    window_temporal = _extract_temporal(
                        t, n_flood_maps, temporal_tensor
                    )
                    window_spatiotemporal = _extract_spatiotemporal_patch(
                        t, n_flood_maps, labels_tensor, patch_size
                    )

                    window_input = model.FloodModel.Input(
                        geospatial=geo_tensor,
                        temporal=window_temporal,
                        spatiotemporal=window_spatiotemporal,
                    )

                    if include_labels:
                        yield window_input, labels_tensor[t]
                    else:
                        empty_label = tf.zeros(
                            (patch_size, patch_size), dtype=tf.float32
                        )
                        yield window_input, empty_label

    # Build dataset
    dataset = tf.data.Dataset.from_generator(
        generator=sample_generator,
        output_signature=(
            dict(
                geospatial=tf.TensorSpec(
                    shape=(patch_size, patch_size, constants.GEO_FEATURES),
                    dtype=tf.float32,
                ),
                temporal=tf.TensorSpec(
                    shape=(n_flood_maps, m_rainfall), dtype=tf.float32
                ),
                spatiotemporal=tf.TensorSpec(
                    shape=(n_flood_maps, patch_size, patch_size, 1),
                    dtype=tf.float32,
                ),
            ),
            tf.TensorSpec(
                shape=(patch_size, patch_size),
                dtype=tf.float32,
            ),
        ),
    )

    dataset = dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return dataset


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

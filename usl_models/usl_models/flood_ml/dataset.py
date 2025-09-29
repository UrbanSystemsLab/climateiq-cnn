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
    include_labels: bool = True,
    rainfall_sim_name: str | None = None,
) -> None:
    """Download simulations from GCS to a local filecache.

    Args:
      sim_names: For training: simulation names like "City-Config/Rainfall_Data_22.txt".
                 For prediction: **study area** names like "Atlanta_Prediction".
      output_path: Directory where the cached files should be stored.
      firestore_client: Client used to query Firestore metadata.
      storage_client: Client used to download objects from GCS.
      dataset_splits: Dataset splits to download (train/val/test). If None,
                      only the "train" split is downloaded (ignored in prediction mode).
      include_labels: If False (prediction), download features only.
      rainfall_sim_name: When include_labels=False, explicit rainfall sim to fetch
                    the temporal vector from, e.g., "City/Config/Rainfall_Data_22.txt".
    """
    firestore_client = firestore_client or firestore.Client()
    storage_client = storage_client or storage.Client()

    if include_labels:
        dataset_splits = dataset_splits or ["train"]
    else:
        dataset_splits = []  # ignored in prediction mode

    for sim_name in sim_names:
        if include_labels:
            # ---------------- TRAINING ----------------
            sim_path = output_path / sim_name
            sim_path.mkdir(parents=True, exist_ok=True)

            # 1. Download temporal vector
            temporal_meta = metastore.get_temporal_feature_metadata(
                firestore_client, sim_name
            )
            temporal_array = downloader.download_as_array(
                storage_client, temporal_meta["as_vector_gcs_uri"]
            )
            np.save(sim_path / TEMPORAL_FILENAME, temporal_array)

            # 2. Download geospatial features + labels
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
            # ---------------- PREDICTION ----------------
            if not rainfall_sim_name:
                raise ValueError(
                    "Prediction mode requires `rainfall_sim_name`, "
                    "e.g. 'Atlanta-Atlanta_config/Rainfall_Data_22.txt'."
                )

            # Derive rainfall filename stem (e.g., 'Rainfall_Data_22')
            rainfall_stem = pathlib.Path(rainfall_sim_name).stem

            # Folder path: e.g., filecache/Atlanta_Prediction/Rainfall_Data_22
            sim_path = output_path / sim_name / rainfall_stem
            sim_path.mkdir(parents=True, exist_ok=True)

            # 1. Download temporal vector from rainfall simulation
            temporal_meta = metastore.get_temporal_feature_metadata(
                firestore_client, rainfall_sim_name
            )
            temporal_array = downloader.download_as_array(
                storage_client, temporal_meta["as_vector_gcs_uri"]
            )
            np.save(sim_path / TEMPORAL_FILENAME, temporal_array)

            # 2. Download geospatial features from the study area
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
                    if ("x_index" in feature_meta and "y_index" in feature_meta)
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
    filecache_dir: pathlib.Path,
    sim_names: list[str],
    dataset_split: str,
    batch_size: int = 4,
    n_flood_maps: int = constants.N_FLOOD_MAPS,
    m_rainfall: int = constants.M_RAINFALL,
    max_chunks: int | None = None,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """Creates a dataset from locally cached simulations."""

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
                yield model_input, labels

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

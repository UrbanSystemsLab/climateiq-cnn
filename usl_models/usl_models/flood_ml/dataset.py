"""tf.data.Datasets for training FloodML model on CityCAT data."""

import dataclasses
import logging
import random
from typing import Any, Iterator, Tuple
import urllib.parse

from google.cloud import firestore  # type:ignore[attr-defined]
from google.cloud import storage  # type:ignore[attr-defined]
import numpy
from numpy.typing import NDArray
import tensorflow as tf

from usl_models.flood_ml import constants
from usl_models.flood_ml import metastore
from usl_models.flood_ml import model


@dataclasses.dataclass
class Filters:
    """Dataset filter settings."""

    # Minimum mean value of a label to apply for filtering out sparse data.
    min_mean_value: float = 0.0
    # Ratio of samples to skip below the min_mean_value threshold [0.0, 1.0].
    min_mean_value_dropout_rate: float = 0.0

    def filter(self, label: tf.Tensor) -> bool:
        """Returns true if the sample should be filtered."""
        return (tf.reduce_mean(label) < self.min_mean_value) and (
            random.random() < self.min_mean_value_dropout_rate
        )


def geospatial_dataset_signature() -> tf.TensorSpec:
    return tf.TensorSpec(
        shape=(
            constants.MAP_HEIGHT,
            constants.MAP_WIDTH,
            constants.GEO_FEATURES,
        ),
        dtype=tf.float32,
    )


def temporal_dataset_signature(m_rainfall: int) -> tf.TensorSpec:
    return tf.TensorSpec(
        shape=(constants.MAX_RAINFALL_DURATION, m_rainfall),
        dtype=tf.float32,
    )


def spatiotemporal_dataset_signature(n_flood_maps: int) -> tf.TensorSpec:
    return tf.TensorSpec(
        shape=(
            n_flood_maps,
            constants.MAP_HEIGHT,
            constants.MAP_WIDTH,
            1,
        ),
        dtype=tf.float32,
    )


def load_dataset(
    sim_names: list[str],
    dataset_split: str,
    batch_size: int = 4,
    n_flood_maps: int = constants.N_FLOOD_MAPS,
    m_rainfall: int = constants.M_RAINFALL,
    max_chunks: int | None = None,
    firestore_client: firestore.Client = None,
    storage_client: storage.Client | None = None,
    filters: Filters | None = None,
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
    filters = filters or Filters()
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
                if filters.filter(labels):
                    continue
                yield model_input, labels

    # Create the dataset for this simulation
    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_signature=(
            dict(
                geospatial=geospatial_dataset_signature(),
                temporal=temporal_dataset_signature(m_rainfall),
                spatiotemporal=spatiotemporal_dataset_signature(n_flood_maps),
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
    filters: Filters | None = None,
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
    filters = filters or Filters()
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
                    if filters.filter(window_label):
                        continue
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

    temporal_vector = _download_as_tensor(storage_client, gcs_url)
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
        feature_tensor = _download_as_tensor(storage_client, feature_url)
        label_tensor = _download_as_tensor(storage_client, label_url)

        reshaped_label_tensor = tf.transpose(label_tensor, perm=[2, 0, 1])
        yield feature_tensor, reshaped_label_tensor


def _download_as_array(client: storage.Client, gcs_url: str) -> NDArray:
    """Retrieves the contents at `gcs_url` from GCS as a numpy array."""
    parsed = urllib.parse.urlparse(gcs_url)
    bucket = client.bucket(parsed.netloc)
    blob = bucket.blob(parsed.path.lstrip("/"))

    with blob.open("rb") as fd:
        return numpy.load(fd)


def _download_as_tensor(client: storage.Client, gcs_url: str) -> tf.Tensor:
    """Retrieves the contents at `gcs_url` from GCS as a tf tensor."""
    return tf.convert_to_tensor(
        _download_as_array(client, gcs_url),
        dtype=tf.float32,
    )

"""tf.data.Datasets for training FloodML model on CityCAT data."""

import functools
import logging
import random
from typing import Optional
import urllib.parse

from google.cloud import firestore
from google.cloud import storage

import numpy
from numpy.typing import NDArray
import tensorflow as tf

from usl_models.flood_ml import constants
from usl_models.flood_ml import metastore
from usl_models.flood_ml import model as flood_model


def load_dataset(
    sim_names: list[str],
    batch_size: int = 4,
    n_flood_maps: int = constants.N_FLOOD_MAPS,
    m_rainfall: int = constants.M_RAINFALL,
    max_examples: Optional[int] = None,
    firestore_client: Optional[firestore.Client] = None,
    storage_client: Optional[storage.Client] = None,
) -> tf.data.Dataset:
    """Creates a dataset which generates chunks for flood model inference.

    This dataset produces the input for `model.call_n`.
    For training with teacher-forcing, `load_dataset_windowed` should be used instead.
    The examples are generated from multiple simulations.
    The dataset iteratively yields examples read from Google Cloud Storage to avoid
    pulling all examples into memory at once.

    Args:
      sim_names: The simulation labels to use for training,
                 e.g. ["Manhattan-config_v1/Rainfall_Data_1.txt"]
      batch_size: Size of batches yielded by the dataset. Approximate memory
                  usage is 10GB * batch_size during training.
      n_flood_maps: The number of flood maps in each example.
      m_rainfall: The width of the temporal rainfall tensor.
      max_examples: The maximum number of examples to yield from the dataset.
                    If `None` (default) yields all examples from the simulations.
      firestore_client: The client to use when interacting with Firestore.
      storage_client: The client to use when interacting with Cloud Storage.
    """
    firestore_client = firestore_client or firestore.Client()
    storage_client = storage_client or storage.Client()

    def generator():
        """Generator for producing full inputs and labels."""
        # Flat list of metadata describing where to download the real tensors from.
        sim_metadata = _get_sims_metadata(
            firestore_client, sim_names, max_examples=max_examples
        )
        while sim_metadata:
            _sim_name, temporal_meta, geo_feature_meta, label_meta = sim_metadata.pop()
            yield _load_example(
                storage_client,
                temporal_meta,
                geo_feature_meta,
                label_meta,
                n_flood_maps,
            )

    # Create the dataset for this simulation
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
                    shape=(constants.MAX_RAINFALL_DURATION, m_rainfall),
                    dtype=tf.float32,
                ),
                spatiotemporal=tf.TensorSpec(
                    shape=(
                        n_flood_maps,
                        constants.MAP_HEIGHT,
                        constants.MAP_WIDTH,
                        1,
                    ),
                    dtype=tf.float32,
                ),
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
    batch_size: int = 4,
    n_flood_maps: int = constants.N_FLOOD_MAPS,
    m_rainfall: int = constants.M_RAINFALL,
    max_examples: Optional[int] = None,
    shuffle: bool = False,
    firestore_client: Optional[firestore.Client] = None,
    storage_client: Optional[storage.Client] = None,
) -> tf.data.Dataset:
    """Creates a dataset which generates chunks for flood model training.

    This dataset produces the input for `model.call`.
    For getting data to input into `model.call_n`, use `load_dataset` instead.
    The examples are generated from multiple simulations.
    They are windowed for training on next-map prediction.
    The dataset iteratively yields examples read from Google Cloud Storage to avoid
    pulling all examples into memory at once.

    Args:
      sim_names: The simulation labels to use for training,
                 e.g. ["Manhattan-config_v1/Rainfall_Data_1.txt"]
      batch_size: Size of batches yielded by the dataset. Approximate memory
                  usage is 10GB * batch_size during training.
      n_flood_maps: The number of flood maps in each example.
      m_rainfall: The width of the temporal rainfall tensor.
      max_examples: The maximum number of examples to yield from the datam_rainfallset.
                    If `None` (default) yields all examples from the simulations.
      firestore_client: The client to use when interacting with Firestore.
      storage_client: The client to use when interacting with Cloud Storage.
    """
    firestore_client = firestore_client or firestore.Client()
    storage_client = storage_client or storage.Client()

    def generator():
        """Windowed generator for teacher-forcing training."""
        # Flat list of metadata describing where to download the real tensors from.
        sim_metadata = _get_sims_metadata_windowed(
            firestore_client, sim_names, shuffle=shuffle, max_examples=max_examples
        )
        while sim_metadata:
            _sim_name, temporal_meta, geo_feature_meta, label_meta, t = (
                sim_metadata.pop()
            )
            yield _load_example_window(
                storage_client,
                temporal_meta,
                geo_feature_meta,
                label_meta,
                t,
                n_flood_maps,
                m_rainfall,
            )

    # Create the dataset for this simulation
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


def _load_example(
    storage_client: storage.Client,
    temporal_meta: metastore.TemporalMetadata,
    geo_feature_meta: metastore.GeoFeatureMetadata,
    label_meta: metastore.LabelMetadata,
    n_flood_maps: int = constants.N_FLOOD_MAPS,
    m_rainfall: int = constants.M_RAINFALL,
) -> tuple[flood_model.FloodModel.Input, tf.Tensor]:
    """Downloads the example described by the given metadata."""
    logging.info(
        "Retrieving features from %s and labels from %s",
        geo_feature_meta["feature_matrix_path"],
        label_meta["gcs_uri"],
    )
    return flood_model.FloodModel.Input(
        temporal=_load_temporal_tensor(storage_client, m_rainfall, temporal_meta),
        geospatial=_download_as_tensor(
            storage_client, geo_feature_meta["feature_matrix_path"]
        ),
        spatiotemporal=tf.zeros(
            shape=(
                n_flood_maps,
                constants.MAP_HEIGHT,
                constants.MAP_WIDTH,
                1,
            )
        ),
    ), _load_label_tensor(storage_client, label_meta)


def _load_example_window(
    storage_client: storage.Client,
    temporal_meta: metastore.TemporalMetadata,
    geo_feature_meta: metastore.GeoFeatureMetadata,
    label_meta: metastore.LabelMetadata,
    t: int,
    n_flood_maps: int = constants.N_FLOOD_MAPS,
    m_rainfall: int = constants.M_RAINFALL,
) -> tuple[flood_model.FloodModel.Input, tf.Tensor]:
    """Loads a windowed example described by the given metadata at timestep t."""
    input, labels = _load_example(
        storage_client,
        temporal_meta,
        geo_feature_meta,
        label_meta,
        n_flood_maps,
        m_rainfall,
    )
    window_input = flood_model.FloodModel.Input(
        geospatial=input["geospatial"],
        temporal=_extract_temporal(t, n_flood_maps, input["temporal"]),
        spatiotemporal=_extract_spatiotemporal(t, n_flood_maps, labels),
    )
    return window_input, labels[t]


def _load_temporal_tensor(
    storage_client: storage.Client,
    m_rainfall: int,
    temporal_meta: metastore.TemporalMetadata,
) -> tf.Tensor:
    gcs_url = temporal_meta["as_vector_gcs_uri"]
    logging.info("Retrieving temporal features from %s.", gcs_url)
    temporal_vector = _download_as_tensor(storage_client, gcs_url)
    return tf.transpose(
        tf.tile(tf.reshape(temporal_vector, (1, len(temporal_vector))), [m_rainfall, 1])
    )


def _load_label_tensor(
    storage_client: storage.Client, label_meta: metastore.LabelMetadata
) -> tf.Tensor:
    """Loads a label tensor from GCloud Storage."""
    label_tensor = _download_as_tensor(storage_client, label_meta["gcs_uri"])
    max_label = tf.math.reduce_max(label_tensor)
    print("max_label:", max_label)
    return tf.transpose(label_tensor, perm=[2, 0, 1])


def _get_sims_metadata(
    firestore_client: firestore.Client,
    sim_names: list[str],
    shuffle: bool = False,
    max_examples: Optional[int] = None,
) -> list[
    tuple[
        str,
        metastore.TemporalMetadata,
        metastore.GeoFeatureMetadata,
        metastore.LabelMetadata,
    ]
]:
    """Returns a flat list of metadata for all (sim_names, chunk) pairs."""
    flat_metadata: list[
        tuple[
            str,
            metastore.TemporalMetadata,
            metastore.GeoFeatureMetadata,
            metastore.LabelMetadata,
        ]
    ] = []
    for sim_name in sim_names:
        temporal_meta = metastore.get_temporal_feature_metadata(
            firestore_client, sim_name
        )
        for (
            geo_feature_meta,
            label_meta,
        ) in metastore.get_spatial_feature_and_label_chunk_metadata(
            firestore_client, sim_name
        ):
            if max_examples is not None and len(flat_metadata) >= max_examples:
                break
            flat_metadata.append(
                (sim_name, temporal_meta, geo_feature_meta, label_meta)
            )

    if shuffle:
        random.shuffle(flat_metadata)

    return flat_metadata


def _get_sims_metadata_windowed(
    firestore_client: firestore.Client,
    sim_names: list[str],
    shuffle: bool = False,
    max_examples: Optional[int] = None,
) -> list[
    tuple[
        str,
        metastore.TemporalMetadata,
        metastore.GeoFeatureMetadata,
        metastore.LabelMetadata,
        int,
    ]
]:
    """Returns a flat list of metadata for all (sim_names, chunk) pairs."""
    flat_metadata: list[
        tuple[
            str,
            metastore.TemporalMetadata,
            metastore.GeoFeatureMetadata,
            metastore.LabelMetadata,
            int,
        ]
    ] = []
    for sim_name in sim_names:
        temporal_meta = metastore.get_temporal_feature_metadata(
            firestore_client, sim_name
        )
        for (
            geo_feature_meta,
            label_meta,
        ) in metastore.get_spatial_feature_and_label_chunk_metadata(
            firestore_client, sim_name
        ):
            if max_examples is not None and len(flat_metadata) >= max_examples:
                break
            for t in range(temporal_meta["rainfall_duration"]):
                flat_metadata.append(
                    (sim_name, temporal_meta, geo_feature_meta, label_meta, t)
                )

    if shuffle:
        random.shuffle(flat_metadata)

    return flat_metadata


def _download_as_array(
    client: storage.Client,
    gcs_url: str,
) -> NDArray:
    """Retrieves the contents at `gcs_url` from GCS as a numpy array."""
    parsed = urllib.parse.urlparse(gcs_url)
    bucket = client.bucket(parsed.netloc)
    filename = parsed.path.lstrip("/")
    blob = bucket.blob(filename)

    with blob.open("rb") as fd:
        return numpy.load(fd)


@functools.lru_cache(maxsize=64)
def _download_as_tensor(
    client: storage.Client,
    gcs_url: str,
) -> tf.Tensor:
    """Retrieves the contents at `gcs_url` from GCS as a tf tensor."""
    return tf.convert_to_tensor(
        _download_as_array(client, gcs_url),
        dtype=tf.float32,
    )

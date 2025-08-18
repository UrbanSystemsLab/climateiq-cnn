"""tf.data.Datasets for training FloodML model on CityCAT data."""

import logging
import random
from typing import Any, Iterator, Tuple

from google.cloud import firestore  # type:ignore[attr-defined]
from google.cloud import storage  # type:ignore[attr-defined]
import tensorflow as tf

from usl_models.flood_ml import constants
from usl_models.flood_ml import metastore
from usl_models.flood_ml import model
from usl_models.shared import downloader


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
    # Collect metadata about available sims/chunks
    sim_chunks = metastore.get_all_chunks(
        sim_names=sim_names,
        split=dataset_split,
        max_chunks=max_chunks,
        firestore_client=firestore_client,
    )
    logging.info("Found %d chunks for %s split", len(sim_chunks), dataset_split)

    def generator() -> Iterator[Tuple[dict[str, tf.Tensor], tf.Tensor]]:
        """Yield (inputs, labels) one example at a time to keep RAM low."""
        for chunk in sim_chunks:
            try:
                # Download rainfall input
                rain_np = downloader.try_download_tensor(
                    storage_client.bucket(constants.RAINFALL_BUCKET),
                    f"{chunk.sim_name}/rainfall/{chunk.chunk_id}.npy",
                )
                if rain_np is None:
                    continue

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
                # Download flood maps (labels)
                flood_np = downloader.try_download_tensor(
                    storage_client.bucket(constants.FLOOD_BUCKET),
                    f"{chunk.sim_name}/flood/{chunk.chunk_id}.npy",
                )
                if flood_np is None:
                    continue

                # Convert/reshape/cast to expected shapes & dtypes
                rainfall = tf.convert_to_tensor(rain_np)
                flood = tf.convert_to_tensor(flood_np)

                rainfall = tf.reshape(
                    rainfall,
                    (m_rainfall, constants.MAP_HEIGHT, constants.MAP_WIDTH),
                )
                flood = tf.reshape(
                    flood, (n_flood_maps, constants.MAP_HEIGHT, constants.MAP_WIDTH)
                )

                rainfall = tf.cast(rainfall, tf.float32)
                flood = tf.cast(flood, tf.float32)

                yield {"rainfall": rainfall}, flood

            except Exception as e:
                logging.warning("Skipping chunk %s due to error: %s", getattr(chunk, "chunk_id", "?"), e)
                continue

    # Dataset signature
    output_signature = (
        {
            "rainfall": tf.TensorSpec(
                shape=(m_rainfall, constants.MAP_HEIGHT, constants.MAP_WIDTH),
                dtype=tf.float32,
            ),
            )
        },
        tf.TensorSpec(
            shape=(n_flood_maps, constants.MAP_HEIGHT, constants.MAP_WIDTH),
            dtype=tf.float32,
        ),
    )
    # If no batch specified, do not batch the dataset, which is required
    # for generating data for batch prediction in VertexAI.
    if batch_size:
        dataset = dataset.batch(batch_size)

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

    # Shuffle only if this is a training split; tune buffer if needed
    if dataset_split.lower() == "train":
        dataset = dataset.shuffle(buffer_size=32, reshuffle_each_iteration=True)

    # Batch + prefetch to keep memory bounded and throughput high
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

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
    """Creates a streaming, windowed dataset for teacher‑forcing training.

    Produces inputs for `model.call`. Keeps memory bounded by yielding
    windows directly from GCS-backed generators (no full in-memory loads).
    """
    firestore_client = firestore_client or firestore.Client()
    storage_client = storage_client or storage.Client()

    def generator():
        """Yield (window_input, window_label) one window at a time."""
        for sim_name in sim_names:
            # Stream one chunk at a time; do not accumulate in memory.
            for model_input, labels in _iter_model_inputs(
                firestore_client,
                storage_client,
                sim_name,
                n_flood_maps,
                m_rainfall,
                max_chunks,
                dataset_split,
            ):
                # Generate sliding windows on-the-fly per chunk
                for window_input, window_label in _generate_windows(
                    model_input, labels, n_flood_maps
                ):
                    # Defensive casts to keep dtypes stable and avoid silent upcasts
                    window_input = dict(
                        geospatial=tf.cast(window_input["geospatial"], tf.float32),
                        temporal=tf.cast(window_input["temporal"], tf.float32),
                        spatiotemporal=tf.cast(window_input["spatiotemporal"], tf.float32),
                    )
                    window_label = tf.cast(window_label, tf.float32)
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

    options = tf.data.Options()
    options.experimental_deterministic = False
    dataset = dataset.with_options(options)

    # Shuffle only for training; buffer tunes RAM vs. randomness
    if dataset_split.lower() == "train":
        dataset = dataset.shuffle(buffer_size=512, reshuffle_each_iteration=True)

    if batch_size:
        dataset = dataset.batch(batch_size, drop_remainder=False)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
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

    temporal_vector = tf.cast(downloader.download_as_tensor(storage_client, gcs_url), tf.float32)
    temporal_vector = tf.transpose(
        tf.tile(tf.reshape(temporal_vector, (1, tf.shape(temporal_vector)[0])), [m_rainfall, 1])
    )
    return temporal_vector, int(temporal_metadata["rainfall_duration"])



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

        logging.info("Retrieving features from %s and labels from %s", feature_url, label_url)
        feature_tensor = tf.cast(downloader.download_as_tensor(storage_client, feature_url), tf.float32)
        label_tensor = tf.cast(downloader.download_as_tensor(storage_client, label_url), tf.float32)

        reshaped_label_tensor = tf.transpose(label_tensor, perm=[2, 0, 1])  # (T, H, W)
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

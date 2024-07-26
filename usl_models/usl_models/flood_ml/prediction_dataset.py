"""tf.data.Datasets for training FloodML model on CityCAT data."""

import logging
import random
from typing import Iterator, Optional, Tuple
import urllib.parse

from google.cloud import firestore  # type:ignore[attr-defined]
from google.cloud import storage  # type:ignore[attr-defined]
import numpy
from numpy.typing import NDArray
import tensorflow as tf

from usl_models.flood_ml import constants
from usl_models.flood_ml import metastore
from usl_models.flood_ml import model


def load_prediction_dataset(
    study_area: str,
    city_cat_config: str,
    batch_size: int = 4,
    n_flood_maps: int = constants.N_FLOOD_MAPS,
    m_rainfall: int = constants.M_RAINFALL,
    max_chunks: Optional[int] = None,
    firestore_client: Optional[firestore.Client] = None,
    storage_client: Optional[storage.Client] = None,
) -> tf.data.Dataset:
    """Creates a prediction dataset which generates chunks for flood model prediction.

    This dataset produces the input for `model.call_n`.
    For training with teacher-forcing, `load_dataset_windowed` should be used instead.
    The examples are generated from multiple simulations.
    The dataset iteratively yields examples read from Google Cloud Storage to avoid
    pulling all examples into memory at once.

    Args:

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
        """Generator for producing full inputs from study area"""
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
            dict(
                feature_chunk=tf.TensorSpec(shape=(), dtype=tf.string),
                rainfall=tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        ),
    )
    # If no batch specified, do not batch the dataset, which is required
    # for generating data for batch prediction in VertexAI.
    if batch_size:
        print("batch: ", batch_size)
        dataset = dataset.batch(batch_size)
    return dataset


def _generate_temporal_tensor(
    firestore_client: firestore.Client,
    storage_client: storage.Client,
    config_name: str,
    m_rainfall: int,
) -> tuple[tf.Tensor, str]:
    """Creates a temporal tensor from the numpy array stored in GCS."""
    temporal_metadata = metastore.get_temporal_feature_metadata_for_prediction(
        firestore_client, config_name
    )
    gcs_url = temporal_metadata["as_vector_gcs_uri"]
    rainfall = temporal_metadata["rainfall_duration"]

    logging.info("Retrieving temporal features from %s.", gcs_url)

    temporal_vector = _download_as_tensor(storage_client, gcs_url)
    return (
        tf.transpose(
            tf.tile(
                tf.reshape(temporal_vector, (1, len(temporal_vector))), [m_rainfall, 1]
            )
        ),
        rainfall,
    )


def _iter_model_inputs_for_prediction(
    firestore_client: firestore.Client,
    storage_client: storage.Client,
    city_cat_config: str,
    study_area_name: str,
    n_flood_maps: int,
    m_rainfall: int,
    max_chunks: Optional[int],
) -> Iterator[Tuple[model.Input, dict]]:
    """Yields model inputs for each spatial chunk in the simulation."""
    temporal, rainfall = _generate_temporal_tensor(
        firestore_client, storage_client, city_cat_config, m_rainfall
    )

    for feature_tensor, chunk_name in _iter_study_area_tensors(
        firestore_client, storage_client, study_area_name
    ):
        metadata = {"feature_chunk": chunk_name, "rainfall": rainfall}

        model_input = model.Input(
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
    feature_metadata = metastore.get_spatial_feature_chunk_metadata_for_prediction(
        firestore_client, study_area_name
    )

    random.shuffle(feature_metadata)

    for feature_metadata in feature_metadata:
        feature_url = feature_metadata["feature_matrix_path"]
        chunk_name = feature_url.split("/")[-1]

        logging.info("Retrieving features from %s ", feature_url)
        feature_tensor = _download_as_tensor(storage_client, feature_url)
        yield feature_tensor, chunk_name


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

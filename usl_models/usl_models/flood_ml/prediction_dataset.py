"""tf.data.Datasets for prediction using FloodML model trained on CityCAT data."""

import logging
from typing import Dict, Iterator, Optional, TypedDict

from google.cloud import firestore, storage  # type:ignore[attr-defined]
import numpy
import tensorflow as tf

from usl_models.flood_ml import constants, dataset


class PredictionInput(TypedDict):
    """Input tensors dictionary used for prediction."""

    geospatial: tf.Tensor
    temporal: tf.Tensor
    spatiotemporal: tf.Tensor
    metadata: tf.Tensor


def load_prediction_dataset(
    rainfall_config_gcs_uri: str,
    study_area_name: str,
    batch_size: int = 4,
    n_flood_maps: int = constants.N_FLOOD_MAPS,
    m_rainfall: int = constants.M_RAINFALL,
    firestore_client: Optional[firestore.Client] = None,
    storage_client: Optional[storage.Client] = None,
) -> tf.data.Dataset:
    """Creates a dataset which generates chunk inputs for the flood model prediction.

    This dataset produces the input for `model.call_n`.
    The dataset iteratively yields chunk examples read from Google Cloud Storage to
    avoid pulling all examples into memory at once.

    Args:
        rainfall_config_gcs_uri: The GCS path for rainfall config numpy matrix.
        study_area_name: Name of study area used to load chunks for.
        batch_size: Size of batches yielded by the dataset.
        n_flood_maps: The number of flood maps in each example.
        m_rainfall: The width of the temporal rainfall tensor.
        storage_client: The client to use when interacting with Cloud Storage.
    """
    firestore_client = firestore_client or firestore.Client()
    storage_client = storage_client or storage.Client()

    def generator():
        """Generator for producing inputs."""
        for prediction_input in _iter_prediction_inputs(
            storage_client,
            rainfall_config_gcs_uri,
            _load_chunk_id_to_uri_map(firestore_client, study_area_name),
            n_flood_maps,
            m_rainfall,
        ):
            yield prediction_input

    # Create the dataset for a given rainfall scenario
    return tf.data.Dataset.from_generator(
        generator=generator,
        output_signature=dict(
            geospatial=dataset.geospatial_generator_signature(),
            temporal=dataset.temporal_generator_signature(m_rainfall),
            spatiotemporal=dataset.spatiotemporal_generator_signature(n_flood_maps),
            metadata=tf.TensorSpec(
                shape=(1),
                dtype=tf.string,
            ),
        ),
    ).batch(batch_size)


def _load_chunk_id_to_uri_map(
    db: firestore.Client, study_area_name: str
) -> Dict[str, str]:
    """Lists chunk IDs from metadata with associated GCS-paths to feature matrices."""
    return {
        ref.id: ref.get().to_dict()["feature_matrix_path"]
        for ref in (
            db.collection("study_areas")
            .document(study_area_name)
            .collection("chunks")
            .list_documents()
        )
    }


def _iter_prediction_inputs(
    storage_client: storage.Client,
    rainfall_config_gcs_uri: str,
    chunk_id_to_gcs_uri_map: Dict[str, str],
    n_flood_maps: int,
    m_rainfall: int,
) -> Iterator[PredictionInput]:
    """Yields model inputs for each spatial chunk in the simulation."""
    temporal = dataset.generate_temporal_tensor_from_gcs_uri(
        storage_client, rainfall_config_gcs_uri, m_rainfall
    )
    feature_metadata_pairs = _iter_geo_feature_tensors_with_metadata(
        storage_client, chunk_id_to_gcs_uri_map
    )
    for geospatial, metadata in feature_metadata_pairs:
        yield PredictionInput(
            temporal=temporal,
            geospatial=geospatial,
            spatiotemporal=dataset.spatiotemporal_zeros(n_flood_maps),
            metadata=metadata,
        )


def _iter_geo_feature_tensors_with_metadata(
    storage_client: storage.Client,
    chunk_id_to_gcs_uri_map: Dict[str, str],
) -> Iterator[tuple[tf.Tensor, tf.Tensor]]:
    """Yields feature and metadata tensors from chunks stored in GCS."""
    for chunk_id, gcs_uri in chunk_id_to_gcs_uri_map.items():
        logging.info("Retrieving feature matrix from %s", gcs_uri)
        feature_tensor = dataset.download_as_tensor(storage_client, gcs_uri)
        yield feature_tensor, tf.convert_to_tensor(
            numpy.array([chunk_id]),
            dtype=tf.string,
        ),

import datetime
from typing import Any, Sequence, TypeVar
import urllib.parse
import logging

from google.cloud import firestore  # type:ignore[attr-defined]


def write_model_metadata(
    db: firestore.Client,
    gcs_model_dir: str,
    sim_names: Sequence[str],
    model_params: dict[str, Any],
    epochs: int,
    model_name: str,
) -> str:
    """Writes information on a trained model to the metastore.

    Args:
      db: The firestore client to use when retrieving metadata.
      gcs_model_dir: The location of the saved model in GCS.
      sim_names: The simulations on which the model was trained.
      model_params: The parameters used to train the model.
      epochs: The number of epochs the model was trained for.
      model_name: A name to associate with the model.

    Returns:
      The document ID of the written model metadata document.
    """
    # Use the GCS location of the model as a unique value for the ID.
    # URL-escape the ID, as characters such as / are not allowed in document IDs.
    model_id = urllib.parse.quote(gcs_model_dir, safe=())
    db.collection("models").document(model_id).set(
        {
            "trained_on": [_get_simulation_doc(db, sim_name) for sim_name in sim_names],
            "trained_at_utc": datetime.datetime.utcnow(),
            "gcs_model_dir": gcs_model_dir,
            "epochs": epochs,
            "model_params": model_params,
            "model_name": model_name,
        }
    )
    return model_id


def get_temporal_feature_metadata(
    db: firestore.Client, sim_name: str
) -> dict[str, Any]:
    """Retrieves metadata stating the location of temporal features in GCS.

    Args:
      db: The firestore client to use when retrieving metadata.
      sim_name: The simulation for which to retrieve metadata.

    Returns:
      A dictionary with keys 'as_vector_gcs_uri' and 'rainfall_duration' which state the
      GCS location of the temporal feature vector and the duration of the rainfall
      represented by the vector.

    Raises:
      ValueError: If a simulation `sim_name` can not be found.
    """
    sim = _get_simulation_doc(db, sim_name).get().to_dict()
    if sim is None:
        raise ValueError(f"No such simulation {sim_name} found.")

    rainfall_config = sim["configuration"]
    return rainfall_config.get().to_dict()


def get_spatial_feature_chunk_metadata(
    db: firestore.Client, sim_name: str
) -> list[dict[str, Any]]:
    """Retrieves metadata stating the location of spatial features in GCS.

    Args:
      db: The firestore client to use when retrieving metadata.
      sim_name: The simulation for which to retrieve metadata.

    Returns:
      A sequence of dictionaries with key 'feature_matrix_path' stating the location in
      GCS of the feature tensor.

    Raises:
      ValueError: If a simulation `sim_name` can not be found.
    """
    sim = _get_simulation_doc(db, sim_name).get().to_dict()
    if sim is None:
        raise ValueError(f"No such simulation {sim_name} found.")

    study_area_ref = sim["study_area"]
    return [doc.to_dict() for doc in study_area_ref.collection("chunks").stream()]


def get_spatial_feature_and_label_chunk_metadata(
    db: firestore.Client, sim_name: str, dataset_split: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Retrieves metadata for the location of (feature, label) pairs in GCS.

    Args:
      db: The firestore client to use when retrieving metadata.
      sim_name: The simulation for which to retrieve metadata.
      dataset_split: Which dataset to retrieve: train, val, or test.

    Returns:
      A sequence of (feature, label) tuples, where `feature` is a dictionary with key
      'feature_matrix_path' stating the location in GCS of the feature tensor and
      `label` is a dictionary with key 'gcs_uri' stating the location in GCS of the
      accompanying label tensor.

    Raises:
      ValueError: If a simulation `sim_name` can not be found.
      AssertionError: If the labels and spatial features for the simulation do not
                      contain the same set of chunks.
    """
    feature_metadata = get_spatial_feature_chunk_metadata(db, sim_name)
    logging.info("feature metadata: %s", str(feature_metadata))

    # Retrieve all label chunks for the simulation.
    label_chunks_collection = (
        _get_simulation_doc(db, sim_name)
        .collection("label_chunks")
        .where(filter=firestore.FieldFilter("dataset", "==", dataset_split))
    )
    label_metadata = [doc.to_dict() for doc in label_chunks_collection.stream()]

    # Map the features and labels by their chunk indices.
    features_by_chunk_index = {
        (feature["x_index"], feature["y_index"]): feature
        for feature in feature_metadata
    }
    labels_by_chunk_index = {
        (label["x_index"], label["y_index"]): label for label in label_metadata
    }

    # Ensure we have the same chunk indices for features and labels.
    missing_features = _missing_keys(labels_by_chunk_index, features_by_chunk_index)
    if missing_features:
        raise AssertionError(
            "Features and label chunks do not line up. "
            f'Indices missing from features: {", ".join(map(str, missing_features))}'
        )

    # Return feature & matching label metadata associated with the same indices.
    return [
        (features_by_chunk_index[index], label)
        for index, label in labels_by_chunk_index.items()
    ]


def get_temporal_feature_metadata_for_prediction(
    db: firestore.Client, config_name: str
) -> dict[str, Any]:
    """Retrieves metadata for temporal features in GCS.

    Args:
      db: The firestore client to use when retrieving metadata.
      config_name: The name of the configuration for which to retrieve metadata.

    Returns:
      A dictionary with keys 'as_vector_gcs_uri' and 'rainfall_duration' which state the
      GCS location of the temporal feature vector and the duration of the rainfall
      represented by the vector.

    Raises:
      ValueError: If a configuration `config_name` cannot be found.
    """
    config_ref = db.collection("city_cat_rainfall_configs").document(config_name)
    config = config_ref.get()

    if not config.exists:
        raise ValueError(f"No such config {config_name} found.")

    return config.to_dict()


def get_spatial_feature_chunk_metadata_for_prediction(
    db: firestore.Client, study_area: str
) -> list[dict[str, Any]]:
    """Retrieves metadata for chunks in the given study area.

    Args:
      db: The firestore client to use when retrieving metadata.
      study_area: The name of the study_area, e.g. NYC

    Returns:
      A list of dictionaries, each representing a chunk's metadata.

    Raises:
      ValueError: If the study area can not be found or it has no chunks.
    """
    doc_ref = db.collection("study_areas").document(study_area)
    doc = doc_ref.get()

    if not doc.exists:
        raise ValueError(f"No such study area '{study_area}' found.")

    chunks = doc_ref.collection("chunks").stream()
    chunk_metadata = [chunk.to_dict() for chunk in chunks]

    if not chunk_metadata:
        raise ValueError(f"No chunks found in study area '{study_area}'.")

    return chunk_metadata


_T = TypeVar("_T")


def _missing_keys(d1: dict[_T, Any], d2: dict[_T, Any]) -> Sequence[_T]:
    """Returns dictionary keys present in d1 but not d2."""
    return [key for key in d1.keys() if key not in d2]


def _get_simulation_doc(
    db: firestore.Client, sim_name: str
) -> firestore.DocumentReference:
    """Retrieves the firestore document for the simulation with the given name."""
    return db.collection("simulations").document(
        # Quote the name to avoid characters not allowed in IDs such as slashes.
        urllib.parse.quote(
            # Unquote to support being passed both quoted & unquoted simulation names.
            urllib.parse.unquote(sim_name),
            safe=(),
        )
    )

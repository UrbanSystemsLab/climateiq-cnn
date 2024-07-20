from typing import Any, Sequence, TypeVar
import urllib.parse

from google.cloud import firestore  # type:ignore[attr-defined]


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
    db: firestore.Client, sim_name: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Retrieves metadata for the location of (feature, label) pairs in GCS.

    Args:
      db: The firestore client to use when retrieving metadata.
      sim_name: The simulation for which to retrieve metadata.

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

    # Retrieve all label chunks for the simulation.
    label_chunks_collection = _get_simulation_doc(db, sim_name).collection(
        "label_chunks"
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
    missing_labels = _missing_keys(features_by_chunk_index, labels_by_chunk_index)
    missing_features = _missing_keys(labels_by_chunk_index, features_by_chunk_index)
    if missing_labels or missing_features:
        raise AssertionError(
            "Features and label chunks do not line up. "
            f'Indices missing from labels: {", ".join(map(str, missing_labels))} '
            f'Indices missing from features: {", ".join(map(str, missing_features))}'
        )

    # Return feature & matching label metadata associated with the same indices.
    return [
        (feature, labels_by_chunk_index[index])
        for index, feature in features_by_chunk_index.items()
    ]


def get_temporal_feature_metadata_for_prediction(
    db: firestore.Client, config_name: str
) -> dict[str, Any]:
    """Retrieves metadata stating the location of temporal features in GCS.

    Args:
      db: The firestore client to use when retrieving metadata.
      config_name: The name of the city cat configuration for which to retrieve metadata.

    Returns:
      A dictionary with keys 'as_vector_gcs_uri' and 'rainfall_duration' which state the
      GCS location of the temporal feature vector and the duration of the rainfall
      represented by the vector.

    Raises:
      ValueError: If a configuration `config_name` cannot be found.
    """
    collection_name = "city_cat_rainfall_configs"
    config_ref = db.collection(collection_name).document(config_name)
    config = config_ref.get()

    if not config.exists:
        raise ValueError(f"No such config {config_name} found.")

    config_data = config.to_dict()
   
    return config_data


def get_spatial_feature_chunk_metadata_for_prediction(
    db: firestore.Client, study_area: str
) -> list[dict[str, Any]]:
    """Retrieves metadata for chunks in the NYC study area.

    Args:
      db: The firestore client to use when retrieving metadata.
      study_area: The name of the study_area, ex: NYC

    Returns:
      A list of dictionaries, each representing a chunk with its metadata.

    Raises:
      ValueError: If the NYC document or its chunks collection cannot be found.
    """
    # Get the reference to the NYC document in the study_area collection
    doc_ref = db.collection("study_areas").document(study_area)
    doc = doc_ref.get()

    if not doc.exists:
        raise ValueError("NYC document not found in study_area collection.")

    # Get the chunks collection under the NYC document
    chunks_collection = doc_ref.collection("chunks")
    
    # Retrieve all chunks
    chunks = chunks_collection.stream()
    
    # Convert each chunk document to a dictionary and store in a list
    chunk_metadata = [chunk.to_dict() for chunk in chunks]
    
    if not chunk_metadata:
        raise ValueError(f"No chunks found in the {doc_ref} document.")

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
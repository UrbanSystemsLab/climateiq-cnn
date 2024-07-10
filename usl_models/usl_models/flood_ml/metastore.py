from typing import Any, Sequence, TypeVar
import urllib.parse

from google.cloud import firestore


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


_T = TypeVar("_T")


def _missing_keys(d1: dict[_T, Any], d2: dict[_T, Any]) -> Sequence[_T]:
    """Returns dictionary keys present in d1 but not d2."""
    return [key for key in d1.keys() if key not in d2]


def _get_simulation_doc(
    db: firestore.Client, sim_name: str
) -> firestore.DocumentReference:
    """Retrieves the firestore document for the simulation with the given name."""
    # Escape the name to avoid characters not allowed in IDs such as slashes.
    return db.collection("simulations").document(urllib.parse.quote(sim_name, safe=()))

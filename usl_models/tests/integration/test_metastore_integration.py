import os

from google.cloud import firestore
import pytest
import requests

from usl_models.flood_ml import metastore

_GCP_PROJECT_ID = "integation-test"


@pytest.fixture
def firestore_db() -> firestore.Client:
    """Creates a firestore DB client and clears the test database."""
    try:
        emulator_host = os.environ["FIRESTORE_EMULATOR_HOST"]
    except KeyError:
        raise RuntimeError(
            "To run these tests, you must start a firestore emulator and set the "
            "FIRESTORE_EMULATOR_HOST environmental variable. "
            "See the README file for details. "
            "You can skip these tests by running: pytest -k 'not integration'"
        )

    requests.delete(
        f"http://{emulator_host}/emulator/v1/projects/{_GCP_PROJECT_ID}/"
        "databases/(default)/documents"
    )

    return firestore.Client(project=_GCP_PROJECT_ID)


def test_get_temporal_feature_metadata(firestore_db) -> None:
    """Ensures we can retrieve temporal metadata for a simulation."""
    firestore_db.collection("simulations").document("sim_name").set(
        {
            "configuration": firestore_db.collection(
                "city_cat_rainfall_configs"
            ).document("config_name")
        }
    )
    firestore_db.collection("city_cat_rainfall_configs").document("config_name").set(
        {"as_vector_gcs_uri": "gs://bucket/path.npy"}
    )

    assert metastore.get_temporal_feature_metadata(firestore_db, "sim_name") == {
        "as_vector_gcs_uri": "gs://bucket/path.npy"
    }


def test_get_temporal_feature_metadata_raises_on_missing_sim(firestore_db) -> None:
    """Ensures we raise an error for missing simulations."""
    with pytest.raises(ValueError):
        metastore.get_temporal_feature_metadata(firestore_db, "sim_name")


def test_get_spatial_feature_chunk_metadata(firestore_db) -> None:
    """Ensures we can retrieve metadata for spatial features."""
    firestore_db.collection("simulations").document("sim_name").set(
        {
            "study_area": firestore_db.collection("study_areas").document(
                "study_area_name"
            )
        }
    )

    firestore_db.collection("study_areas").document("study_area_name").collection(
        "chunks"
    ).document("chunk_0_0").set({"feature_matrix_path": "gs://bucket/chunk_0_0.npy"})

    firestore_db.collection("study_areas").document("study_area_name").collection(
        "chunks"
    ).document("chunk_0_1").set({"feature_matrix_path": "gs://bucket/chunk_0_1.npy"})

    assert metastore.get_spatial_feature_chunk_metadata(firestore_db, "sim_name") == [
        {"feature_matrix_path": "gs://bucket/chunk_0_0.npy"},
        {"feature_matrix_path": "gs://bucket/chunk_0_1.npy"},
    ]


def test_get_spatial_feature_chunk_metadata_raises_on_missing_sim(firestore_db) -> None:
    """Ensures we raise an error for missing simulations."""
    with pytest.raises(ValueError):
        metastore.get_spatial_feature_chunk_metadata(firestore_db, "sim_name")


def test_get_spatial_feature_and_label_chunk_metadata(firestore_db) -> None:
    """Ensures we can retrieve metadata for combined spatial features & labels."""
    firestore_db.collection("simulations").document("sim_name").set(
        {
            "study_area": firestore_db.collection("study_areas").document(
                "study_area_name"
            )
        }
    )

    # Insert features.
    firestore_db.collection("study_areas").document("study_area_name").collection(
        "chunks"
    ).document("chunk_0_0").set(
        {"feature_matrix_path": "gs://bucket/chunk_0_0.npy", "x_index": 0, "y_index": 0}
    )

    firestore_db.collection("study_areas").document("study_area_name").collection(
        "chunks"
    ).document("chunk_0_1").set(
        {"feature_matrix_path": "gs://bucket/chunk_0_1.npy", "x_index": 0, "y_index": 1}
    )

    # Insert matching labels.
    firestore_db.collection("simulations").document("sim_name").collection(
        "label_chunks"
    ).document("0_0").set(
        {"gcs_uri": "gs://bucket/0_0.npy", "x_index": 0, "y_index": 0}
    )

    firestore_db.collection("simulations").document("sim_name").collection(
        "label_chunks"
    ).document("0_1").set(
        {"gcs_uri": "gs://bucket/0_1.npy", "x_index": 0, "y_index": 1}
    )

    assert metastore.get_spatial_feature_and_label_chunk_metadata(
        firestore_db, "sim_name"
    ) == [
        (
            {
                "feature_matrix_path": "gs://bucket/chunk_0_0.npy",
                "x_index": 0,
                "y_index": 0,
            },
            {"gcs_uri": "gs://bucket/0_0.npy", "x_index": 0, "y_index": 0},
        ),
        (
            {
                "feature_matrix_path": "gs://bucket/chunk_0_1.npy",
                "x_index": 0,
                "y_index": 1,
            },
            {"gcs_uri": "gs://bucket/0_1.npy", "x_index": 0, "y_index": 1},
        ),
    ]


def test_get_spatial_feature_and_label_chunk_metadata_raises_on_missing_sim(
    firestore_db,
) -> None:
    """Ensures we raise an error for missing simulations."""
    with pytest.raises(ValueError):
        metastore.get_spatial_feature_and_label_chunk_metadata(firestore_db, "sim_name")


def test_get_label_chunk_metadata_missing_features(firestore_db) -> None:
    """Ensures we raise an error if labels do not match up with features."""
    firestore_db.collection("simulations").document("sim_name").set(
        {
            "study_area": firestore_db.collection("study_areas").document(
                "study_area_name"
            )
        }
    )

    # Insert 1 feature.
    firestore_db.collection("study_areas").document("study_area_name").collection(
        "chunks"
    ).document("chunk_0_0").set(
        {"feature_matrix_path": "gs://bucket/chunk_0_0.npy", "x_index": 0, "y_index": 0}
    )

    # Insert 2 labels.
    firestore_db.collection("simulations").document("sim_name").collection(
        "label_chunks"
    ).document("0_0").set(
        {"gcs_uri": "gs://bucket/0_0.npy", "x_index": 0, "y_index": 0}
    )

    firestore_db.collection("simulations").document("sim_name").collection(
        "label_chunks"
    ).document("0_1").set(
        {"gcs_uri": "gs://bucket/0_1.npy", "x_index": 0, "y_index": 1}
    )

    with pytest.raises(AssertionError) as excinfo:
        metastore.get_spatial_feature_and_label_chunk_metadata(firestore_db, "sim_name")

    assert "Indices missing from labels:  Indices missing from features: (0, 1)" in str(
        excinfo.value
    )


def test_get_label_chunk_metadata_missing_labels(firestore_db) -> None:
    """Ensures we raise an error if features do not match up with labels."""
    firestore_db.collection("simulations").document("sim_name").set(
        {
            "study_area": firestore_db.collection("study_areas").document(
                "study_area_name"
            )
        }
    )

    # Insert 2 features.
    firestore_db.collection("study_areas").document("study_area_name").collection(
        "chunks"
    ).document("chunk_0_0").set(
        {"feature_matrix_path": "gs://bucket/chunk_0_0.npy", "x_index": 0, "y_index": 0}
    )

    firestore_db.collection("study_areas").document("study_area_name").collection(
        "chunks"
    ).document("chunk_0_1").set(
        {"feature_matrix_path": "gs://bucket/chunk_0_1.npy", "x_index": 0, "y_index": 1}
    )

    # Insert 1 label.
    firestore_db.collection("simulations").document("sim_name").collection(
        "label_chunks"
    ).document("0_0").set(
        {"gcs_uri": "gs://bucket/0_0.npy", "x_index": 0, "y_index": 0}
    )

    with pytest.raises(AssertionError) as excinfo:
        metastore.get_spatial_feature_and_label_chunk_metadata(firestore_db, "sim_name")

    assert "Indices missing from labels: (0, 1) Indices missing from features:" in str(
        excinfo.value
    )

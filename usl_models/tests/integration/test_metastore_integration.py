import os
import unittest

from google.cloud import firestore  # type:ignore[attr-defined]
import pytest
import requests

from usl_models.flood_ml import metastore
from usl_models.flood_ml import model_params

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
    firestore_db.collection("simulations").document("dir%2Fsim_name").set(
        {
            "configuration": firestore_db.collection(
                "city_cat_rainfall_configs"
            ).document("config_name")
        }
    )
    firestore_db.collection("city_cat_rainfall_configs").document("config_name").set(
        {"as_vector_gcs_uri": "gs://bucket/path.npy"}
    )

    assert metastore.get_temporal_feature_metadata(firestore_db, "dir%2Fsim_name") == {
        "as_vector_gcs_uri": "gs://bucket/path.npy"
    }


def test_get_temporal_feature_metadata_unquoted_name(firestore_db) -> None:
    """Ensures we can retrieve temporal metadata for unquoted simulation names."""
    firestore_db.collection("simulations").document("dir%2Fsim_name").set(
        {
            "configuration": firestore_db.collection(
                "city_cat_rainfall_configs"
            ).document("config_name")
        }
    )
    firestore_db.collection("city_cat_rainfall_configs").document("config_name").set(
        {"as_vector_gcs_uri": "gs://bucket/path.npy"}
    )

    assert metastore.get_temporal_feature_metadata(firestore_db, "dir/sim_name") == {
        "as_vector_gcs_uri": "gs://bucket/path.npy"
    }


def test_get_temporal_feature_metadata_raises_on_missing_sim(firestore_db) -> None:
    """Ensures we raise an error for missing simulations."""
    with pytest.raises(ValueError):
        metastore.get_temporal_feature_metadata(firestore_db, "dir%2Fsim_name")


def test_get_spatial_feature_chunk_metadata(firestore_db) -> None:
    """Ensures we can retrieve metadata for spatial features."""
    firestore_db.collection("simulations").document("dir%2Fsim_name").set(
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

    assert metastore.get_spatial_feature_chunk_metadata(
        firestore_db, "dir%2Fsim_name"
    ) == [
        {"feature_matrix_path": "gs://bucket/chunk_0_0.npy"},
        {"feature_matrix_path": "gs://bucket/chunk_0_1.npy"},
    ]


def test_get_spatial_feature_chunk_metadata_unquoted_name(firestore_db) -> None:
    """Ensures we can retrieve metadata for unquoted simulation names."""
    firestore_db.collection("simulations").document("dir%2Fsim_name").set(
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

    assert metastore.get_spatial_feature_chunk_metadata(
        firestore_db, "dir/sim_name"
    ) == [
        {"feature_matrix_path": "gs://bucket/chunk_0_0.npy"},
        {"feature_matrix_path": "gs://bucket/chunk_0_1.npy"},
    ]


def test_get_spatial_feature_chunk_metadata_raises_on_missing_sim(firestore_db) -> None:
    """Ensures we raise an error for missing simulations."""
    with pytest.raises(ValueError):
        metastore.get_spatial_feature_chunk_metadata(firestore_db, "dir%2Fsim_name")


def test_get_spatial_feature_and_label_chunk_metadata(firestore_db) -> None:
    """Ensures we can retrieve metadata for combined spatial features & labels."""
    firestore_db.collection("simulations").document("dir%2Fsim_name").set(
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
    firestore_db.collection("simulations").document("dir%2Fsim_name").collection(
        "label_chunks"
    ).document("0_0").set(
        {
            "gcs_uri": "gs://bucket/0_0.npy",
            "x_index": 0,
            "y_index": 0,
            "dataset": "train",
        }
    )

    firestore_db.collection("simulations").document("dir%2Fsim_name").collection(
        "label_chunks"
    ).document("0_1").set(
        {
            "gcs_uri": "gs://bucket/0_1.npy",
            "x_index": 0,
            "y_index": 1,
            "dataset": "train",
        }
    )

    assert metastore.get_spatial_feature_and_label_chunk_metadata(
        firestore_db, "dir%2Fsim_name", "train"
    ) == [
        (
            {
                "feature_matrix_path": "gs://bucket/chunk_0_0.npy",
                "x_index": 0,
                "y_index": 0,
            },
            {
                "gcs_uri": "gs://bucket/0_0.npy",
                "x_index": 0,
                "y_index": 0,
                "dataset": "train",
            },
        ),
        (
            {
                "feature_matrix_path": "gs://bucket/chunk_0_1.npy",
                "x_index": 0,
                "y_index": 1,
            },
            {
                "gcs_uri": "gs://bucket/0_1.npy",
                "x_index": 0,
                "y_index": 1,
                "dataset": "train",
            },
        ),
    ]


def test_get_spatial_feature_and_label_chunk_metadata_unquoted_name(
    firestore_db,
) -> None:
    """Ensures we can retrieve metadata when given unquoted names."""
    firestore_db.collection("simulations").document("dir%2Fsim_name").set(
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
        {
            "feature_matrix_path": "gs://bucket/chunk_0_0.npy",
            "x_index": 0,
            "y_index": 0,
            "dataset": "train",
        }
    )

    # Insert matching labels.
    firestore_db.collection("simulations").document("dir%2Fsim_name").collection(
        "label_chunks"
    ).document("0_0").set(
        {
            "gcs_uri": "gs://bucket/0_0.npy",
            "x_index": 0,
            "y_index": 0,
            "dataset": "train",
        }
    )

    assert metastore.get_spatial_feature_and_label_chunk_metadata(
        firestore_db, "dir/sim_name", "train"
    ) == [
        (
            {
                "feature_matrix_path": "gs://bucket/chunk_0_0.npy",
                "x_index": 0,
                "y_index": 0,
                "dataset": "train",
            },
            {
                "gcs_uri": "gs://bucket/0_0.npy",
                "x_index": 0,
                "y_index": 0,
                "dataset": "train",
            },
        ),
    ]


def test_get_spatial_feature_and_label_chunk_metadata_raises_on_missing_sim(
    firestore_db,
) -> None:
    """Ensures we raise an error for missing simulations."""
    with pytest.raises(ValueError):
        metastore.get_spatial_feature_and_label_chunk_metadata(
            firestore_db, "dir%2Fsim_name", "train"
        )


def test_get_label_chunk_metadata_missing_features(firestore_db) -> None:
    """Ensures we raise an error if labels do not match up with features."""
    firestore_db.collection("simulations").document("dir%2Fsim_name").set(
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
    firestore_db.collection("simulations").document("dir%2Fsim_name").collection(
        "label_chunks"
    ).document("0_0").set(
        {
            "gcs_uri": "gs://bucket/0_0.npy",
            "x_index": 0,
            "y_index": 0,
            "dataset": "train",
        }
    )

    firestore_db.collection("simulations").document("dir%2Fsim_name").collection(
        "label_chunks"
    ).document("0_1").set(
        {
            "gcs_uri": "gs://bucket/0_1.npy",
            "x_index": 0,
            "y_index": 1,
            "dataset": "train",
        }
    )

    with pytest.raises(AssertionError) as excinfo:
        metastore.get_spatial_feature_and_label_chunk_metadata(
            firestore_db, "dir%2Fsim_name", "train"
        )

    assert "Indices missing from features: (0, 1)" in str(excinfo.value)


def test_write_model_metadata(firestore_db) -> None:
    id_ = metastore.write_model_metadata(
        firestore_db,
        "gs://the/model",
        ["sim-1", "sim-2"],
        model_params.default_params(),
        5,
        "a_model",
    )
    result = firestore_db.collection("models").document(id_).get().to_dict()
    # We won't know what time it is, just make sure it's present.
    assert "trained_at_utc" in result
    assert result["gcs_model_dir"] == "gs://the/model"
    assert result["model_params"] == model_params.default_params()
    assert result["epochs"] == 5
    assert result["model_name"] == "a_model"
    assert result["trained_on"] == [
        firestore_db.collection("simulations").document("sim-1"),
        firestore_db.collection("simulations").document("sim-2"),
    ]


def test_get_temporal_feature_metadata_for_prediction(firestore_db) -> None:
    firestore_db.collection("city_cat_rainfall_configs").document("config_name").set(
        {"as_vector_gcs_uri": "gs://bucket/path.npy"}
    )

    assert metastore.get_temporal_feature_metadata_for_prediction(
        firestore_db, "config_name"
    ) == {"as_vector_gcs_uri": "gs://bucket/path.npy"}


def test_get_temporal_feature_metadata_for_prediction_raises_error_for_config(
    firestore_db,
) -> None:
    with pytest.raises(ValueError):
        metastore.get_temporal_feature_metadata_for_prediction(
            firestore_db, "config_name"
        )


def test_get_spatial_feature_chunk_metadata_for_prediction(firestore_db) -> None:
    firestore_db.collection("study_areas").document("study_area_name").set({})

    firestore_db.collection("study_areas").document("study_area_name").collection(
        "chunks"
    ).document("chunk_0_0").set({"feature_matrix_path": "gs://bucket/chunk_0_0.npy"})

    firestore_db.collection("study_areas").document("study_area_name").collection(
        "chunks"
    ).document("chunk_0_1").set({"feature_matrix_path": "gs://bucket/chunk_0_1.npy"})

    firestore_db.collection("study_areas").document(
        "another_study_area_name"
    ).collection("chunks").document("chunk_0_3").set(
        {"feature_matrix_path": "gs://bucket/chunk_0_3.npy"}
    )

    unittest.TestCase().assertCountEqual(
        metastore.get_spatial_feature_chunk_metadata_for_prediction(
            firestore_db, "study_area_name"
        ),
        [
            {"feature_matrix_path": "gs://bucket/chunk_0_0.npy"},
            {"feature_matrix_path": "gs://bucket/chunk_0_1.npy"},
        ],
    )


def test_get_spatial_feature_chunk_metadata_for_prediction_raises_for_missing(
    firestore_db,
) -> None:
    with pytest.raises(ValueError):
        metastore.get_spatial_feature_chunk_metadata_for_prediction(
            firestore_db, "study_area_name"
        )


def test_get_spatial_feature_chunk_metadata_for_prediction_raises_for_empty(
    firestore_db,
) -> None:
    firestore_db.collection("study_areas").document("study_area_name").set({})

    with pytest.raises(ValueError):
        metastore.get_spatial_feature_chunk_metadata_for_prediction(
            firestore_db, "study_area_name"
        )

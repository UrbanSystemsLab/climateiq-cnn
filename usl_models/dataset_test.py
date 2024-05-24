import pytest
from unittest.mock import MagicMock
import numpy as np
import tensorflow as tf
import json
import io

from usl_models.flood_ml.dataset import IncrementalTrainDataGenerator
from usl_models.flood_ml.data_utils import FloodModelData


@pytest.fixture
def config_file(tmp_path):
    config = {"firestore_collection": "test_collection", "batch_size": 2}
    config_path = tmp_path / "config.json"
    with config_path.open("w") as f:
        json.dump(config, f)
    return config_path


@pytest.fixture
def study_areas():
    return ["A", "B"]


@pytest.fixture
def firestore_client():
    return MagicMock()


@pytest.fixture
def storage_client():
    return MagicMock()


@pytest.fixture
def generator(study_areas, config_file, firestore_client, storage_client):
    return IncrementalTrainDataGenerator(
        study_areas,
        2,
        config_file=config_file,
        firestore_client=firestore_client,
        storage_client=storage_client,
    )


def test_create_firestore_client(generator, firestore_client):
    assert generator.firestore_client == firestore_client


def test_create_storage_client(generator, storage_client):
    assert generator.storage_client == storage_client


def test_get_chunks_for_study_areas(generator, firestore_client):
    firestore_client.collection().where().stream.return_value = [
        MagicMock(
            to_dict=lambda: {
                "chunks": ["gs://bucket/chunk1.npy", "gs://bucket/chunk2.npy"]
            }
        )
    ]
    chunks = generator._get_chunks_for_study_areas()
    assert chunks == ["gs://bucket/chunk1.npy", "gs://bucket/chunk2.npy"]


def test_get_numpy_tensor_from_gcs(generator, storage_client):
    mock_blob = MagicMock()
    tensor_dict = {"data": np.array([1, 2, 3, 4])}

    buffer = io.BytesIO()
    np.savez(buffer, **tensor_dict)
    buffer.seek(0)

    mock_blob.download_as_bytes.return_value = buffer.read()
    storage_client.bucket().blob.return_value = mock_blob

    gcs_url = "gs://bucket/chunk1.npy"
    tensor = generator._get_numpy_tensor_from_gcs(gcs_url)
    assert np.array_equal(tensor["data"], np.array([1, 2, 3, 4]))


def test_convert_to_train_data(generator):
    numpy_arrays = [
        {"storm_duration": np.array([1, 2, 3]), "geospatial": np.array([4, 5, 6])}
    ]
    train_data = generator._convert_to_train_data(numpy_arrays)
    assert isinstance(train_data, dict)
    assert isinstance(train_data["storm_duration"], tf.Tensor)
    assert isinstance(train_data["geospatial"], tf.Tensor)
    assert train_data["storm_duration"].shape == (1, 3)
    assert train_data["geospatial"].shape == (1, 3)


def test_get_next_batch(generator, firestore_client, storage_client):
    firestore_client.collection().where().stream.return_value = [
        MagicMock(
            to_dict=lambda: {
                "chunks": ["gs://bucket/chunk1.npy", "gs://bucket/chunk2.npy"]
            }
        )
    ]

    tensor_dict = {
        "storm_duration": np.array([1, 2, 3]),
        "geospatial": np.array([4, 5, 6]),
        "temporal": np.array([7, 8, 9]),
        "spatiotemporal": np.array([10, 11, 12]),
        "labels": np.array([13, 14, 15]),
    }

    buffer = io.BytesIO()
    np.savez(buffer, **tensor_dict)
    buffer.seek(0)

    mock_blob = MagicMock()
    mock_blob.download_as_bytes.return_value = buffer.read()
    storage_client.bucket().blob.return_value = mock_blob

    batch_data = generator.get_next_batch()
    assert isinstance(batch_data, FloodModelData)
    assert isinstance(batch_data.storm_duration, tf.Tensor)
    assert isinstance(batch_data.geospatial, tf.Tensor)
    assert isinstance(batch_data.temporal, tf.Tensor)
    assert isinstance(batch_data.spatiotemporal, tf.Tensor)
    assert isinstance(batch_data.labels, tf.Tensor)
    assert batch_data.storm_duration.shape == (2, 3)
    assert batch_data.geospatial.shape == (2, 3)
    assert batch_data.temporal.shape == (2, 3)
    assert batch_data.spatiotemporal.shape == (2, 3)
    assert batch_data.labels.shape == (2, 3)
    # Adjust shapes based on actual data dimensions

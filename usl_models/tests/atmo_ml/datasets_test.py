import numpy as np
import io
import pytest
from unittest import mock
from google.cloud import storage  # type: ignore
from usl_models.atmo_ml.datasets import create_atmo_dataset
from usl_models.atmo_ml import constants


@pytest.fixture
@mock.patch("usl_models.atmo_ml.datasets.storage.Client")
def mock_storage_client(mock_storage_client):
    """Fixture for mocking Google Cloud Storage client."""
    return mock_storage_client()


@pytest.fixture
@mock.patch("usl_models.atmo_ml.datasets.firestore.Client")
def mock_firestore_client(mock_firestore_client):
    """Fixture for mocking Firestore client."""
    return mock_firestore_client()


def test_create_atmo_dataset(mock_storage_client, mock_firestore_client):
    """Test creating AtmoML dataset from GCS."""
    bucket_name = "test-bucket"
    spatiotemporal_file_names = [
        "spatiotemporal_data_1.npy",
        "spatiotemporal_data_2.npy",
    ]
    label_file_names = ["label_data_1.npy", "label_data_2.npy"]
    spatial_file_name = "spatial_data.npy"
    lu_index_file_name = "lu_index.npy"
    time_steps_per_day = 4
    batch_size = 2

    # Set up mock data
    mock_spatiotemporal_data = np.random.rand(
        10,
        constants.MAP_HEIGHT,
        constants.MAP_WIDTH,
        constants.num_spatiotemporal_features,
    ).astype(np.float32)

    mock_label_data = np.random.rand(
        10, constants.MAP_HEIGHT, constants.MAP_WIDTH, 1
    ).astype(np.float32)

    mock_spatial_data = np.random.rand(
        constants.MAP_HEIGHT, constants.MAP_WIDTH, constants.num_spatial_features
    ).astype(np.float32)

    mock_lu_index_data = np.random.randint(
        0, 10, size=(constants.MAP_HEIGHT * constants.MAP_WIDTH,)
    ).astype(np.int32)

    def mock_blob_func(blob_name):
        buf = io.BytesIO()
        if "spatiotemporal" in blob_name:
            np.save(buf, mock_spatiotemporal_data)
        elif "label" in blob_name:
            np.save(buf, mock_label_data)
        elif "spatial" in blob_name:
            np.save(buf, mock_spatial_data)
        elif "lu_index" in blob_name:
            np.save(buf, mock_lu_index_data)
        buf.seek(0)
        mock_blob = mock.MagicMock(spec=storage.Blob)
        mock_blob.download_as_bytes.return_value = buf.read()
        return mock_blob

    mock_storage_client().bucket().blob.side_effect = mock_blob_func

    train_dataset, val_dataset, test_dataset = create_atmo_dataset(
        bucket_name=bucket_name,
        spatiotemporal_file_names=spatiotemporal_file_names,
        label_file_names=label_file_names,
        spatial_file_name=spatial_file_name,
        lu_index_file_name=lu_index_file_name,
        time_steps_per_day=time_steps_per_day,
        batch_size=batch_size,
        storage_client=mock_storage_client(),
        firestore_client=mock_firestore_client(),
    )

    # Check dataset shapes
    for data in train_dataset.take(1):
        inputs, labels = data
        assert inputs["spatiotemporal"].shape == (
            batch_size,
            time_steps_per_day,
            constants.MAP_HEIGHT,
            constants.MAP_WIDTH,
            constants.num_spatiotemporal_features,
        )
        assert inputs["spatial"].shape == (
            batch_size,
            constants.MAP_HEIGHT,
            constants.MAP_WIDTH,
            constants.num_spatial_features,
        )
        assert inputs["lu_index"].shape == (
            batch_size,
            constants.MAP_HEIGHT * constants.MAP_WIDTH,
        )
        assert labels.shape == (
            batch_size,
            time_steps_per_day,
            constants.MAP_HEIGHT,
            constants.MAP_WIDTH,
            1,
        )


def test_load_prediction_dataset(mock_storage_client, mock_firestore_client):
    """Test loading prediction dataset from GCS."""
    from usl_models.atmo_ml.datasets import load_prediction_dataset

    sim_names = ["simulation_1", "simulation_2"]
    batch_size = 2

    # Set up mock data
    mock_spatiotemporal_data = np.random.rand(
        10,
        constants.TIME_STEPS_PER_DAY,
        constants.MAP_HEIGHT,
        constants.MAP_WIDTH,
        constants.num_spatiotemporal_features,
    ).astype(np.float32)

    mock_spatial_data = np.random.rand(
        constants.MAP_HEIGHT, constants.MAP_WIDTH, constants.num_spatial_features
    ).astype(np.float32)

    mock_lu_index_data = np.random.randint(
        0, 10, size=(constants.MAP_HEIGHT * constants.MAP_WIDTH,)
    ).astype(np.int32)

    def mock_blob_func(blob_name):
        buf = io.BytesIO()
        if "spatiotemporal" in blob_name:
            np.save(buf, mock_spatiotemporal_data)
        elif "spatial" in blob_name:
            np.save(buf, mock_spatial_data)
        elif "lu_index" in blob_name:
            np.save(buf, mock_lu_index_data)
        buf.seek(0)
        mock_blob = mock.MagicMock(spec=storage.Blob)
        mock_blob.download_as_bytes.return_value = buf.read()
        mock_blob.open.return_value = buf  # Mock the open call to return the buffer
        return mock_blob

    mock_storage_client().bucket().blob.side_effect = mock_blob_func

    dataset = load_prediction_dataset(
        sim_names=sim_names,
        batch_size=batch_size,
        storage_client=mock_storage_client(),
        firestore_client=mock_firestore_client(),
    )

    # Check prediction dataset
    for inputs in dataset.take(1):
        assert inputs["spatiotemporal"].shape == (
            batch_size,
            constants.TIME_STEPS_PER_DAY,
            constants.MAP_HEIGHT,
            constants.MAP_WIDTH,
            constants.num_spatiotemporal_features,
        )
        assert inputs["spatial"].shape == (
            batch_size,
            constants.MAP_HEIGHT,
            constants.MAP_WIDTH,
            constants.num_spatial_features,
        )
        assert inputs["lu_index"].shape == (
            batch_size,
            constants.MAP_HEIGHT * constants.MAP_WIDTH,
        )

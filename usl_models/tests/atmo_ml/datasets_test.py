import unittest
from unittest import mock
import tensorflow as tf
import numpy as np
import io
from unittest.mock import MagicMock
from usl_models.atmo_ml.datasets import create_atmo_dataset, load_labels_from_cloud


class TestAtmoMLDataset(unittest.TestCase):
    @mock.patch("google.cloud.storage.Client")
    @mock.patch("google.cloud.firestore.Client")
    def test_create_atmo_dataset(self, mock_firestore_client, mock_storage_client):
        """Test creating AtmoML dataset from GCS."""
        # Set up mock bucket and blobs
        mock_storage_client_instance = mock_storage_client.return_value
        mock_bucket = MagicMock()
        mock_storage_client_instance.bucket.return_value = mock_bucket

        # Simulate spatial data blob
        mock_spatial_blob = MagicMock()
        mock_bucket.blob.return_value = mock_spatial_blob
        mock_spatial_data = np.random.rand(200, 200, 17).astype(np.float32)
        spatial_data_bytes = io.BytesIO()
        np.save(spatial_data_bytes, mock_spatial_data, allow_pickle=True)
        spatial_data_bytes.seek(0)
        mock_spatial_blob.download_as_bytes.return_value = spatial_data_bytes.getvalue()

        # Simulate spatiotemporal data blobs (6 time steps)
        mock_spatiotemporal_blobs = [MagicMock() for _ in range(3)]
        for i, blob in enumerate(mock_spatiotemporal_blobs):
            spatiotemporal_data = np.random.rand(6, 200, 200, 9).astype(
                np.float32
            )  # 6 time steps
            spatiotemporal_data_bytes = io.BytesIO()
            np.save(spatiotemporal_data_bytes, spatiotemporal_data, allow_pickle=True)
            spatiotemporal_data_bytes.seek(0)
            blob.download_as_bytes.return_value = spatiotemporal_data_bytes.getvalue()

        # Simulate land use index data blob
        mock_lu_index_blob = MagicMock()
        mock_bucket.blob.return_value = mock_lu_index_blob
        lu_index_data = np.random.randint(0, 10, size=(200 * 200,)).astype(np.int32)
        lu_index_data_bytes = io.BytesIO()
        np.save(lu_index_data_bytes, lu_index_data, allow_pickle=True)
        lu_index_data_bytes.seek(0)
        mock_lu_index_blob.download_as_bytes.return_value = (
            lu_index_data_bytes.getvalue()
        )

        # Simulate label data blobs (8 time steps)
        mock_label_blobs = [MagicMock() for _ in range(3)]
        for i, blob in enumerate(mock_label_blobs):
            label_data = np.random.rand(8, 200, 200, 1).astype(
                np.float32
            )  # 8 time steps
            label_data_bytes = io.BytesIO()
            np.save(label_data_bytes, label_data, allow_pickle=True)
            label_data_bytes.seek(0)
            blob.download_as_bytes.return_value = label_data_bytes.getvalue()

        # Mock blob listing behavior to simulate folder structure
        mock_bucket.list_blobs.side_effect = lambda prefix: {
            "spatial": [mock_spatial_blob],
            "spatiotemporal": mock_spatiotemporal_blobs,
            "lu_index": [mock_lu_index_blob],
            "labels": mock_label_blobs,
        }.get(prefix, [])

        # Define bucket names and folder paths
        data_bucket_name = "test-data-bucket"
        label_bucket_name = "test-label-bucket"
        spatiotemporal_folder = "spatiotemporal"
        spatial_folder = "spatial"
        lu_index_folder = "lu_index"
        label_folder = "labels"
        time_steps_per_day = 6  # Adjust as needed for spatiotemporal input
        batch_size = 2

        # Call the function under test
        train_dataset, val_dataset, test_dataset = create_atmo_dataset(
            data_bucket_name=data_bucket_name,
            label_bucket_name=label_bucket_name,
            spatiotemporal_folder=spatiotemporal_folder,
            spatial_folder=spatial_folder,
            lu_index_folder=lu_index_folder,
            label_folder=label_folder,
            time_steps_per_day=time_steps_per_day,
            batch_size=batch_size,
            storage_client=mock_storage_client_instance,
            firestore_client=mock_firestore_client,
        )

        # Check if datasets are created and contain data
        self.assertIsInstance(train_dataset, tf.data.Dataset)
        self.assertIsInstance(val_dataset, tf.data.Dataset)
        self.assertIsInstance(test_dataset, tf.data.Dataset)

        # Iterate over one batch to see if data structure matches the expected signature
        for inputs, labels in train_dataset.take(1):
            self.assertIn("spatiotemporal", inputs)
            self.assertIn("spatial", inputs)
            self.assertIn("lu_index", inputs)
            self.assertEqual(
                inputs["spatiotemporal"].shape, (batch_size, 6, 200, 200, 9)
            )  # 6 time steps
            self.assertEqual(inputs["spatial"].shape, (batch_size, 200, 200, 17))
            self.assertEqual(inputs["lu_index"].shape, (batch_size, 200 * 200))
            self.assertEqual(labels.shape, (batch_size, 8, 200, 200, 1))  # 8 time steps


@mock.patch("google.cloud.storage.Client")
@mock.patch("google.cloud.firestore.Client")
def test_load_labels_from_cloud(mock_firestore_client, mock_storage_client):
    """Test loading labels from GCS and verifying correct data structure and content."""
    # Set up mock storage client instance and bucket
    mock_storage_client_instance = mock_storage_client.return_value
    mock_bucket = MagicMock()
    mock_storage_client_instance.bucket.return_value = mock_bucket

    # Simulate label data blobs with random data
    num_time_steps = 6
    height, width, channels = 200, 200, 1
    num_blobs = 3  # Number of label blobs to simulate
    mock_label_blobs = [MagicMock() for _ in range(num_blobs)]

    for i, blob in enumerate(mock_label_blobs):
        label_data = np.random.rand(num_time_steps, height, width, channels).astype(
            np.float32
        )
        label_data_bytes = io.BytesIO()
        np.save(label_data_bytes, label_data, allow_pickle=True)
        label_data_bytes.seek(0)
        blob.download_as_bytes.return_value = label_data_bytes.getvalue()

    # Mock blob listing behavior to simulate folder structure
    mock_bucket.list_blobs.return_value = mock_label_blobs

    # Define bucket name and folder path for labels
    label_bucket_name = "test-label-bucket"
    label_folder = "labels"

    # Call the function under test
    labels_tensor = load_labels_from_cloud(
        bucket_name=label_bucket_name,
        folder_name=label_folder,
        storage_client=mock_storage_client_instance,
        firestore_client=mock_firestore_client,
    )

    # Validate the type and shape of the returned tensor
    expected_shape = (num_blobs, num_time_steps, height, width, channels)
    assert isinstance(labels_tensor, tf.Tensor)
    assert (
        labels_tensor.shape == expected_shape
    ), f"Expected shape {expected_shape}, but got {labels_tensor.shape}"

    # Verify that the data within the tensor is as expected (basic data integrity check)
    for i in range(num_blobs):
        np.testing.assert_array_almost_equal(
            labels_tensor[i].numpy(),
            np.load(io.BytesIO(mock_label_blobs[i].download_as_bytes())),
        )

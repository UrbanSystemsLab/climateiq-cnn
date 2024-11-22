# TODO: fix this file
import io
import numpy as np
import unittest
from unittest import mock
import tensorflow as tf
from unittest.mock import MagicMock
from usl_models.atmo_ml import dataset


def create_mock_blob(data, dtype=np.float32, allow_pickle=True):
    """Create a mock blob with simulated data and return it."""
    blob = MagicMock()
    data_bytes = io.BytesIO()
    np.save(data_bytes, data.astype(dtype), allow_pickle=allow_pickle)
    data_bytes.seek(0)
    blob.download_as_bytes.return_value = data_bytes.getvalue()
    return blob


class TestAtmoMLDataset(unittest.TestCase):
    @mock.patch("google.cloud.storage.Client")
    def test_load_dataset_structure(self, mock_storage_client):
        """Test creating AtmoML dataset from GCS with expected structure and shapes."""
        # Mock GCS client and bucket
        mock_storage_client_instance = mock_storage_client.return_value
        mock_bucket = MagicMock()
        mock_storage_client_instance.bucket.return_value = mock_bucket

        # Define mock URLs
        # mock_spatial_url = "gs://test-data-bucket/sim1/spatial/mock_spatial_data.npy"
        # mock_spatiotemporal_urls = [
        #     f"gs://test-data-bucket/sim1/spatiotemporal/mock_spatiotemporal_data_{i}.npy"
        #     for i in range(3)
        # ]
        # mock_lu_index_url = "gs://test-data-bucket/sim1/lu_index/mock_lu_index_data.npy"
        # mock_label_urls = [
        #     f"gs://test-label-bucket/sim1/mock_label_data_{i}.npy" for i in range(3)
        # ]

        # Simulate mock blobs for different datasets
        mock_spatial_blob = create_mock_blob(
            np.random.rand(200, 200, 17).astype(np.float32)
        )
        mock_bucket.list_blobs.return_value = [mock_spatial_blob]
        mock_spatiotemporal_blobs = [
            create_mock_blob(np.random.rand(6, 200, 200, 9).astype(np.float32))
            for _ in range(3)
        ]
        mock_lu_index_blob = create_mock_blob(
            np.random.randint(
                0,
                10,
                size=(200, 200),
            ).astype(np.int32)
        )
        mock_bucket.list_blobs.return_value = [mock_lu_index_blob]

        mock_label_blobs = [
            create_mock_blob(np.random.rand(8, 200, 200, 1).astype(np.float32))
            for _ in range(3)
        ]

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
        time_steps_per_day = 6
        batch_size = 2

        # Call the function under test
        ds = dataset.load_dataset(
            data_bucket_name=data_bucket_name,
            label_bucket_name=label_bucket_name,
            sim_names=["sim1"],
            time_steps_per_day=time_steps_per_day,
            storage_client=mock_storage_client_instance,
        )
        ds = ds.batch(batch_size=batch_size)
        train_dataset, val_dataset, test_dataset = dataset.split_dataset(
            ds, train_frac=0.7, val_frac=0.15, test_frac=0.15
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
            )
            self.assertEqual(inputs["spatial"].shape, (batch_size, 200, 200, 17))
            self.assertEqual(inputs["lu_index"].shape, (batch_size, 200, 200))
            self.assertEqual(labels.shape, (batch_size, 8, 200, 200, 1))

    @mock.patch("google.cloud.storage.Client")
    def test_load_labels_from_cloud(self, mock_storage_client):
        """Test loading labels from GCS and verifying correct data structure."""
        # Mock storage and blobs
        mock_storage_client_instance = mock_storage_client.return_value
        mock_bucket = MagicMock()
        mock_storage_client_instance.bucket.return_value = mock_bucket

        # Simulate label data blobs
        num_time_steps = 8
        height, width, channels = 200, 200, 1
        num_blobs = 3
        mock_label_blobs = [
            create_mock_blob(np.random.rand(num_time_steps, height, width, channels))
            for _ in range(num_blobs)
        ]

        # Mock bucket behavior
        mock_bucket.list_blobs.return_value = mock_label_blobs

        # Call the function under test
        labels_tensor = dataset.load_labels_from_cloud(
            bucket_name="test-label-bucket",
            folder_name="labels",
            storage_client=mock_storage_client_instance,
        )

        # Verify the tensor structure
        self.assertIsInstance(labels_tensor, tf.Tensor)
        self.assertEqual(
            labels_tensor.shape, (num_blobs, num_time_steps, height, width, channels)
        )

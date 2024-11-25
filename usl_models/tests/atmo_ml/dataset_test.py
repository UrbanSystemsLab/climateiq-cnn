# TODO: fix this file
import io
import numpy as np
import unittest
from unittest import mock
import tensorflow as tf
import pprint
from unittest.mock import MagicMock
import usl_models.testing
from usl_models.atmo_ml import dataset
from usl_models.atmo_ml import constants


def create_mock_blob(data, dtype=np.float32, allow_pickle=True):
    """Create a mock blob with simulated data and return it."""
    blob = MagicMock()
    data_bytes = io.BytesIO()
    np.save(data_bytes, data.astype(dtype), allow_pickle=allow_pickle)
    data_bytes.seek(0)
    blob.download_as_bytes.return_value = data_bytes.getvalue()
    return blob


class TestAtmoMLDataset(usl_models.testing.TestCase):
    @mock.patch("google.cloud.storage.Client")
    def test_load_dataset_structure(self, mock_storage_client):
        """Test creating AtmoML dataset from GCS with expected structure and shapes."""
        # Mock GCS client and bucket
        mock_storage_client_instance = mock_storage_client.return_value
        mock_bucket = MagicMock()
        mock_storage_client_instance.bucket.return_value = mock_bucket

        num_days = 4
        timesteps_per_day = 6
        num_timesteps = num_days * timesteps_per_day
        batch_size = 2

        B = batch_size
        H, W = constants.MAP_HEIGHT, constants.MAP_WIDTH
        F_S = constants.NUM_SAPTIAL_FEATURES
        F_ST = constants.NUM_SPATIOTEMPORAL_FEATURES
        C = constants.OUTPUT_CHANNELS
        T_I, T_O = constants.INPUT_TIME_STEPS, constants.OUTPUT_TIME_STEPS

        # Simulate mock blobs for datasets
        mock_spatial_blob = create_mock_blob(
            np.random.rand(H, W, F_S).astype(np.float32)
        )
        mock_spatiotemporal_tensor = np.random.rand(H, W, F_ST).astype(np.float32)
        mock_spatiotemporal_blobs = [
            create_mock_blob(mock_spatiotemporal_tensor) for _ in range(num_timesteps)
        ]
        mock_lu_index_blob = create_mock_blob(
            np.random.randint(
                low=0,
                high=10,
                size=(H, W),
            ).astype(np.int32)
        )
        mock_label_blobs = [
            create_mock_blob(np.random.rand(H, W, C).astype(np.float32))
            for _ in range(num_timesteps)
        ]

        # Mock blob listing behavior to simulate folder structure
        mock_bucket.list_blobs.side_effect = lambda prefix: {
            "sim1/spatial": [mock_spatial_blob],
            "sim1/spatiotemporal": mock_spatiotemporal_blobs,
            "sim1/lu_index": [mock_lu_index_blob],
            "sim1": mock_label_blobs,
        }[prefix]

        # Define bucket names and folder paths
        data_bucket_name = "test-data-bucket"
        label_bucket_name = "test-label-bucket"

        # Call the function under test
        ds = dataset.load_dataset(
            data_bucket_name=data_bucket_name,
            label_bucket_name=label_bucket_name,
            sim_names=["sim1"],
            timesteps_per_day=timesteps_per_day,
            storage_client=mock_storage_client_instance,
        )
        ds = ds.batch(batch_size=batch_size)

        inputs, labels = zip(*ds)
        num_batches = num_days // batch_size
        self.assertShapesRecursive(
            list(inputs),
            [
                {
                    "spatiotemporal": (B, T_I, H, W, F_ST),
                    "spatial": (B, H, W, F_S),
                    "lu_index": (B, H, W),
                }
            ] * num_batches,
        )
        self.assertShapesRecursive(
            list(labels),
            [
                (B, T_O, H, W, C),
            ] * num_batches,
        )

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

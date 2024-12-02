# TODO: fix this file
import io

from unittest import mock
from unittest.mock import MagicMock

import numpy as np

import usl_models.testing
from usl_models.atmo_ml import dataset
from usl_models.atmo_ml import constants


def create_mock_blob(data, dtype=np.float32, allow_pickle=True):
    """Create a mock blob with simulated data and return it."""
    blob = MagicMock()
    buf = io.BytesIO()
    np.save(buf, data.astype(dtype), allow_pickle=allow_pickle)
    buf.seek(0)
    blob.open.return_value = buf
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
            ]
            * num_batches,
        )
        self.assertShapesRecursive(
            list(labels),
            [
                (B, T_O, H, W, C),
            ]
            * num_batches,
        )

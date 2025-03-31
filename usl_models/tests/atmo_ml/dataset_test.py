from datetime import datetime, timedelta

import numpy as np

import usl_models.testing
from usl_models.testing import MockBlob, MockStorageClient, MockBucket
from usl_models.atmo_ml import dataset
from usl_models.atmo_ml import constants


# Test constants
B = 2
H, W = constants.MAP_HEIGHT, constants.MAP_WIDTH
F_S = constants.NUM_SAPTIAL_FEATURES
F_ST = constants.NUM_SPATIOTEMPORAL_FEATURES
C = 3
T_I, T_O = constants.INPUT_TIME_STEPS, 2


class TestAtmoMLDataset(usl_models.testing.TestCase):
    def setUp(self):
        """Sets up the dataset in mock GCS."""
        feature_bucket_name = "test-feature-bucket"
        label_bucket_name = "test-label-bucket"
        sim_name = "test-sim"

        num_days = 4
        timestep_hours = 6
        timesteps_per_day = 24 // timestep_hours
        num_timesteps = num_days * timesteps_per_day
        start_time = datetime.strptime("2000-05-24", dataset.DATE_FORMAT)
        timestamps = [
            start_time + timedelta(hours=timestep_hours * i)
            for i in range(-1, num_timesteps + 1)
        ]
        label_timestamps = [
            start_time + timedelta(hours=timestep_hours // 2 * i)
            for i in range(num_timesteps * 2)
        ]

        filenames = [ts.strftime(dataset.FEATURE_FILENAME_FORMAT) for ts in timestamps]
        label_filenames = [
            ts.strftime(dataset.LABEL_FILENAME_FORMAT) for ts in label_timestamps
        ]

        spatiotemporal_tensor = np.random.rand(H, W, F_ST).astype(np.float32)

        feature_bucket = MockBucket().with_blobs(
            {
                f"{sim_name}/lu_index/{filename}": MockBlob()
                .with_npy(np.random.randint(0, 10, size=(H, W), dtype=np.int32))
                .with_path(
                    f"/b/{feature_bucket_name}/o/{sim_name}/lu_index/{filenames}"
                )
                for filename in filenames
            }
            | {
                f"{sim_name}/spatial/{filename}": MockBlob()
                .with_npy(np.random.rand(H, W, F_S).astype(np.float32))
                .with_path(f"/b/{feature_bucket_name}/o/{sim_name}/spatial/{filename}")
                for filename in filenames
            }
            | {
                f"{sim_name}/spatiotemporal/{filename}": MockBlob()
                .with_npy(spatiotemporal_tensor)
                .with_path(
                    f"/b/{feature_bucket_name}/o/{sim_name}/spatiotemporal/{filename}"
                )
                for filename in filenames
            }
        )
        label_bucket = MockBucket().with_blobs(
            {
                f"{sim_name}/{filename}": MockBlob()
                .with_npy(np.random.rand(H, W, C).astype(np.float32))
                .with_path(f"/b/{label_bucket_name}/o/{sim_name}/{filename}")
                for filename in label_filenames
            }
        )
        self.client = MockStorageClient().with_buckets(
            {
                feature_bucket_name: feature_bucket,
                label_bucket_name: label_bucket,
            }
        )
        self.sim_name = sim_name
        self.feature_bucket_name = feature_bucket_name
        self.label_bucket_name = label_bucket_name
        self.num_days = num_days
        return super().setUp()

    def test_load_day(self):
        """Test loading a single example with expected structure and shapes."""
        load_result = dataset.load_day(
            date=datetime.strptime("2000-05-25", dataset.DATE_FORMAT),
            sim_name=self.sim_name,
            feature_bucket=self.client.bucket(self.feature_bucket_name),
            label_bucket=self.client.bucket(self.label_bucket_name),
            config=dataset.Config(),
        )
        self.assertNotEqual(load_result, None)
        inputs, label = load_result
        self.assertShapesRecursive(
            inputs,
            {
                "spatiotemporal": (T_I, H, W, F_ST),
                "spatial": (H, W, F_S),
                "lu_index": (H, W),
                "date": "",
                "sim_name": "",
            },
        )
        self.assertShape(label, (T_O, H, W, C))

    def test_load_dataset(self):
        """Test creating AtmoML dataset from GCS with expected structure and shapes."""
        ds = dataset.load_dataset(
            data_bucket_name=self.feature_bucket_name,
            label_bucket_name=self.label_bucket_name,
            sim_names=[self.sim_name],
            storage_client=self.client,
            config=dataset.Config(),
        ).batch(batch_size=B)

        inputs, labels = zip(*ds)
        num_batches = self.num_days // B
        inputs = list(inputs)
        labels = list(labels)
        self.assertShapesRecursive(
            list(inputs),
            [
                {
                    "spatiotemporal": (B, T_I, H, W, F_ST),
                    "spatial": (B, H, W, F_S),
                    "lu_index": (B, H, W),
                    "date": (B,),
                    "sim_name": (B,),
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

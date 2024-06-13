import os
import pytest
from unittest.mock import patch
import numpy as np

import tensorflow as tf


from usl_models.flood_ml.data_utils import FloodModelData
from usl_models.flood_ml.metastore import FirestoreDataHandler
from usl_models.flood_ml.settings import Settings
from usl_models.flood_ml.geospatial import GeospatialTensor
from usl_models.flood_ml.spatiotemporal import SpatiotemporalTensor
from usl_models.flood_ml.dataset import IncrementalTrainDataGenerator


class TestIncrementalTrainDataGenerator:
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch: pytest.MonkeyPatch):
        # Set up environment variables
        monkeypatch.setenv("LOCAL_NUMPY_DIR", "/tmp/test_numpy")
        monkeypatch.setenv("DEBUG", "1")

        # Create a mock FirestoreDataHandler
        self.mock_firestore_data_handler = FirestoreDataHandler(
            firestore_client=None, settings=Settings()
        )

        # Create a mock GeospatialTensor
        self.mock_geospatial_tensor = GeospatialTensor(settings=Settings())

        # Create a mock SpatiotemporalTensor
        self.mock_spatiotemporal_tensor = SpatiotemporalTensor(settings=Settings())

        # Create an instance of IncrementalTrainDataGenerator
        self.data_generator = IncrementalTrainDataGenerator(
            study_areas=["test-1"],
            batch_size=1,
            firestore_client=None,
            storage_client=None,
            metastore=self.mock_firestore_data_handler,
            geospatial_tensor=self.mock_geospatial_tensor,
            spatial_temporal_tensor=self.mock_spatiotemporal_tensor,
        )

    def test_init(self):
        assert self.data_generator.study_areas == ["test-1"]
        assert self.data_generator.batch_size == 1
        assert self.data_generator.metastore == self.mock_firestore_data_handler
        assert self.data_generator.geospatial_tensor == self.mock_geospatial_tensor
        assert (
            self.data_generator.spatial_temporal_tensor
            == self.mock_spatiotemporal_tensor
        )

    def test_get_next_batch(
        self,
        mock_process_batch,
        mock_create_tfrecord_dataset,
        mock_create_tfrecord_from_numpy,
        mock_download_numpy_files,
    ):
        # Mock data
        mock_process_batch.return_value = [FloodModelData()]
        mock_create_tfrecord_dataset.return_value = None
        mock_create_tfrecord_from_numpy.return_value = [
            tf.train.Example().SerializeToString()
        ]
        mock_download_numpy_files.return_value = None

        # Call the method
        result = self.data_generator.get_next_batch(["test-1"], 1)

        # Assert results
        assert result == {"test-1": [FloodModelData()]}

        # Assert mock calls
        mock_process_batch.assert_called_once_with(
            mock_create_tfrecord_dataset.return_value,
            "test-1",
            1,
            None,
        )
        mock_create_tfrecord_dataset.assert_called_once_with(
            mock_create_tfrecord_from_numpy.return_value,
            {"labels": tf.io.FixedLenFeature([], tf.string)},
        )
        mock_create_tfrecord_from_numpy.assert_called_once_with("labels")
        mock_download_numpy_files.assert_called_once_with(
            self.mock_firestore_data_handler._get_label_chunks_urls("test-1")
        )

    def test_get_next_batch_with_empty_study_areas(
        self,
        mock_process_batch,
        mock_create_tfrecord_dataset,
        mock_create_tfrecord_from_numpy,
        mock_download_numpy_files,
    ):
        # Call the method with empty study_areas
        with pytest.raises(ValueError) as excinfo:
            self.data_generator.get_next_batch([], 1)

        # Assert the error message
        assert "study_areas cannot be empty" in str(excinfo.value)

    def test_get_next_batch_with_invalid_study_areas(
        self,
        mock_process_batch,
        mock_create_tfrecord_dataset,
        mock_create_tfrecord_from_numpy,
        mock_download_numpy_files,
    ):
        # Call the method with invalid study_areas
        with pytest.raises(ValueError) as excinfo:
            self.data_generator.get_next_batch("test-1", 1)

        # Assert the error message
        assert "study_areas must be a list" in str(excinfo.value)

    def test_get_next_batch_with_invalid_batch_size(
        self,
        mock_process_batch,
        mock_create_tfrecord_dataset,
        mock_create_tfrecord_from_numpy,
        mock_download_numpy_files,
    ):
        # Call the method with invalid batch_size
        with pytest.raises(ValueError) as excinfo:
            self.data_generator.get_next_batch(["test-1"], 0)

        # Assert the error message
        assert "batch must be a positive integer greater than or equal to 1" in str(
            excinfo.value
        )

    def test_get_next_batch_with_no_simulation_data(
        self,
        mock_process_batch,
        mock_create_tfrecord_dataset,
        mock_create_tfrecord_from_numpy,
        mock_download_numpy_files,
    ):
        # Mock no simulation data
        self.mock_firestore_data_handler._retrieve_simulation_data.return_value = {}

        # Call the method
        with pytest.raises(ValueError) as excinfo:
            self.data_generator.get_next_batch(["test-1"], 1)

        # Assert the error message
        assert "No simulation data found for the given study areas" in str(
            excinfo.value
        )

    def test_get_next_batch_with_error_processing_data(
        self,
        mock_process_batch,
        mock_create_tfrecord_dataset,
        mock_create_tfrecord_from_numpy,
        mock_download_numpy_files,
    ):
        # Mock error processing data
        mock_process_batch.side_effect = Exception("Error processing data")

        # Call the method
        result = self.data_generator.get_next_batch(["test-1"], 1)

        # Assert the result is empty
        assert result == {}

        # Assert the error message is printed
        assert (
            "Error: Error producing training data for study area:  test-1"
            in self.data_generator._print_log
        )

    def test_download_numpy_files(self):
        # Mock GCS URLs
        gcs_urls = [
            "gs://test-bucket/test_file1.npy",
            "gs://test-bucket/test_file2.npy",
        ]

        # Call the method
        self.data_generator._download_numpy_files(gcs_urls)

        # Assert files are downloaded to LOCAL_NUMPY_DIR
        assert os.path.exists(
            os.path.join(os.environ["LOCAL_NUMPY_DIR"], "test_file1.npy")
        )
        assert os.path.exists(
            os.path.join(os.environ["LOCAL_NUMPY_DIR"], "test_file2.npy")
        )

    def test_download_numpy_files_with_empty_urls(self):
        # Call the method with empty GCS URLs
        with pytest.raises(ValueError) as excinfo:
            self.data_generator._download_numpy_files([])

        # Assert the error message
        assert "GCS URLs are empty." in str(excinfo.value)

    def test_create_tfrecord_from_numpy(self):
        # Mock numpy files
        numpy_files = [
            os.path.join(os.environ["LOCAL_NUMPY_DIR"], "test_file1.npy"),
            os.path.join(os.environ["LOCAL_NUMPY_DIR"], "test_file2.npy"),
        ]

        # Call the method
        result = self.data_generator._create_tfrecord_from_numpy("labels")

        # Assert the result is a list of TFRecord examples
        assert isinstance(result, list)
        assert len(result) == 2

        # Assert the TFRecord examples are valid
        for example in result:
            assert isinstance(example, tf.train.Example)

    def test_create_tfrecord_from_numpy_with_empty_directory(self):
        # Set LOCAL_NUMPY_DIR to an empty directory
        os.environ["LOCAL_NUMPY_DIR"] = "/tmp/empty_dir"

        # Call the method
        result = self.data_generator._create_tfrecord_from_numpy("labels")

        # Assert the result is an empty list
        assert result == []

    def test_create_tfrecord_dataset(
        self, mock_process_batch, mock_create_tfrecord_dataset, mock_create_tfrecord_from_numpy, mock_download_numpy_files
    ):
        # Mock data
        mock_process_batch.return_value = [FloodModelData()]
        mock_create_tfrecord_dataset.return_value = None
        mock_create_tfrecord_from_numpy.return_value = [
            tf.train.Example().SerializeToString()
        ]
        mock_download_numpy_files.return_value = None

        # Call the method
        result = self.data_generator.get_next_batch(["test-1"], 1)

        # Assert results
        assert result == {"test-1": [FloodModelData()]}

        # Assert mock calls
        mock_process_batch.assert_called_once_with(
            mock_create_tfrecord_dataset.return_value,
            "test-1",
            1,
            None,
        )
        mock_create_tfrecord_dataset.assert_called_once_with(
            mock_create_tfrecord_from_numpy.return_value,
            {"labels": tf.io.FixedLenFeature([], tf.string)},
        )
        mock_create_tfrecord_from_numpy.assert_called_once_with("labels")
        mock_download_numpy_files.assert_called_once_with(
            self.mock_firestore_data_handler._get_label_chunks_urls("test_study_area")
        )


    def test_create_tfrecord_dataset_with_empty_examples(self):
        # Call the method with empty serialized examples
        result = self.data_generator._create_tfrecord_dataset(
            [], {"labels": tf.io.FixedLenFeature([], tf.string)}
        )

        # Assert the result is an empty dataset
        assert result.cardinality().numpy() == 0

    def test_parse_tf_example(self):
        # Mock data
        serialized_example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "labels": tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[tf.io.serialize_tensor(np.zeros((1000, 1000, 4))).numpy()]
                        )
                    )
                }
            )
        ).SerializeToString()
        feature_description = {"labels": tf.io.FixedLenFeature([], tf.float32)}

        # Mock tf.io.parse_tensor
        with patch("tensorflow.io.parse_tensor") as mock_parse_tensor:
            mock_parse_tensor.return_value = tf.convert_to_tensor(np.zeros((1000, 1000, 4)))

            # Call the method
            result = self.data_generator._parse_tf_example(
                serialized_example, feature_description
            )

            # Assert results
            assert isinstance(result, dict)
            assert "labels" in result
            assert isinstance(result["labels"], tf.Tensor)
            assert result["labels"].shape == (1000, 1000, 4)
            assert result["labels"].dtype == tf.float32


    def test_process_batch(self):
        # Mock data
        dataset = tf.data.Dataset.from_tensor_slices(
            [
                tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "labels": tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[
                                        tf.io.serialize_tensor(
                                            np.zeros((1000, 1000, 4))
                                        ).numpy()
                                    ]
                                )
                            )
                        }
                    )
                )
            ]
        )
        study_area = "test-1"
        rainfall_duration = 10
        temporal_tensor = tf.convert_to_tensor(np.zeros((864,)))

        # Call the method
        result = self.data_generator._process_batch(
            dataset, study_area, rainfall_duration, temporal_tensor
        )

        # Assert the result is a list of FloodModelData objects
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], FloodModelData)

        # Assert the FloodModelData object has the expected attributes
        assert result[0].storm_duration == rainfall_duration
        assert result[0].geospatial.shape == (32, 1000, 1000, 3)
        assert result[0].temporal.shape == (864,)
        assert result[0].spatiotemporal.shape == (32, 1000, 1000, 1)
        assert result[0].labels.shape == (32, 1000, 1000, 4)

    def test_process_batch_with_empty_dataset(self):
        # Call the method with an empty dataset
        result = self.data_generator._process_batch(
            tf.data.Dataset.from_tensor_slices([]),
            "test-1",
            10,
            tf.convert_to_tensor(np.zeros((32, 1000, 1000, 1))),
        )

        # Assert the result is an empty list
        assert result == []

    def test_generate_geospatial_tensor(self):
        # Mock study_area and data
        study_area = "test-1"
        data = {"test_key": "test_value"}

        # Call the method
        result = self.data_generator._generate_geospatial_tensor(study_area, data)

        # Assert the result is a TensorFlow tensor
        assert isinstance(result, tf.Tensor)

        # Assert the tensor has the expected shape
        assert result.shape == (32, 1000, 1000, 3)

    def test_generate_spatio_temporal_tensor(self):
        # Mock study_area and data
        study_area = "test-1"
        data = {"test_key": "test_value"}

        # Call the method
        result = self.data_generator._generate_spatio_temporal_tensor(study_area, data)

        # Assert the result is a TensorFlow tensor
        assert isinstance(result, tf.Tensor)

        # Assert the tensor has the expected shape
        assert result.shape == (32, 1000, 1000, 1)

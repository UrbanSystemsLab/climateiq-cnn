import io
from unittest import mock
import numpy as np
from google.cloud import storage, firestore  # type: ignore
from usl_models.atmo_ml import datasets
from usl_models.atmo_ml import cnn_inputs_outputs


@mock.patch.object(datasets, "create_atmo_dataset")
@mock.patch("usl_models.atmo_ml.datasets.make_predictions")
def test_atmo_dataset(mock_make_predictions, mock_create_dataset) -> None:
    """Create expected dataset for the atmo model from GCS and Firestore objects."""
    mock_storage_client = mock.MagicMock(spec=storage.Client)
    mock_firestore_client = mock.MagicMock(spec=firestore.Client)

    # Set parameters for the dataset
    bucket_name = "our-bucket-name"
    spatiotemporal_file_names = [
        "spatiotemporal_data_1.npy",
        "spatiotemporal_data_2.npy",
    ]
    label_file_names = ["labels_1.npy", "labels_2.npy"]
    spatial_file_name = "spatial_data.npy"
    lu_index_file_name = "lu_index.npy"
    time_steps_per_day = 6
    batch_size = 4

    # Set up mock data for the GCS
    stinp_1 = np.random.rand(batch_size, 100, 100, 1)
    labelinput_1 = np.random.rand(batch_size, 100, 100, 1)
    stinp_2 = np.random.rand(batch_size, 100, 100, 1)
    labelinput_2 = np.random.rand(batch_size, 100, 100, 1)

    # Call the function to process inputs and labels
    mock_spatiotemporal_data_1, mock_labels_1 = cnn_inputs_outputs.divide_into_days(
        stinp_1, labelinput_1, input_steps_per_day=4, label_steps_per_day=8
    )
    mock_spatiotemporal_data_2, mock_labels_2 = cnn_inputs_outputs.divide_into_days(
        stinp_2, labelinput_2, input_steps_per_day=4, label_steps_per_day=8
    )
    mock_spatial_data = np.random.rand(100, 100, 1)
    mock_lu_index_data = np.random.randint(1, 62, (100, 100))

    def mock_blob_func(blob_name):
        """Returns the mock data depending on the blob requested."""
        if blob_name == "spatiotemporal_data_1.npy":
            mock_data = mock_spatiotemporal_data_1
        elif blob_name == "spatiotemporal_data_2.npy":
            mock_data = mock_spatiotemporal_data_2
        elif blob_name == "labels_1.npy":
            mock_data = mock_labels_1
        elif blob_name == "labels_2.npy":
            mock_data = mock_labels_2
        elif blob_name == "spatial_data.npy":
            mock_data = mock_spatial_data
        elif blob_name == "lu_index.npy":
            mock_data = mock_lu_index_data
        else:
            raise ValueError(f"Unexpected name {blob_name} passed to mock.")

        buf = io.BytesIO()
        np.save(buf, mock_data)
        buf.seek(0)

        mock_blob = mock.MagicMock(spec=storage.Blob)
        mock_blob.open.return_value = buf
        return mock_blob

    # Set the storage client to return the right mock data for each GCS path.
    mock_storage_client.bucket().blob.side_effect = mock_blob_func

    # Mock Firestore interactions
    mock_firestore_client.collection().document().get().to_dict.return_value = {
        "some_firestore_key": "some_value"
    }

    # Create the dataset using mocked data
    dataset = datasets.create_atmo_dataset(
        bucket_name=bucket_name,
        spatiotemporal_file_names=spatiotemporal_file_names,
        label_file_names=label_file_names,
        spatial_file_name=spatial_file_name,
        lu_index_file_name=lu_index_file_name,
        time_steps_per_day=time_steps_per_day,
        batch_size=batch_size,
        storage_client=mock_storage_client,
        firestore_client=mock_firestore_client,  # Add Firestore client here
    )

    # Mock predictions returned by the model
    mock_predictions = np.random.rand(batch_size, time_steps_per_day, 100, 100, 1)
    mock_make_predictions.return_value = mock_predictions

    # Call the prediction function and validate
    predictions = datasets.make_predictions(mock.MagicMock(), dataset)
    assert predictions.shape == mock_predictions.shape, "Predictions shape mismatch"

    # Iterate through the dataset and check the shape
    for data in dataset:
        inputs, labels = data
        assert inputs["spatiotemporal"].shape == (
            batch_size,
            time_steps_per_day,
            100,
            100,
            1,
        ), "Input spatiotemporal shape mismatch"
        assert inputs["spatial"].shape == (100, 100, 1), "Spatial data shape mismatch"
        assert inputs["lu_index"].shape == (100 * 100,), "Land use index shape mismatch"
        assert labels.shape == (
            batch_size,
            time_steps_per_day,
            100,
            100,
            1,
        ), "Label shape mismatch"

        print("Inputs:", inputs)
        print("Labels:", labels)
        print("Predictions:", predictions)


@mock.patch("usl_models.atmo_ml.datasets.storage.Client")
def test_load_prediction_dataset(mock_storage_client) -> None:
    """Test loading and generating batches for predictions."""
    bucket_name = "test-bucket"
    spatiotemporal_file_names = [
        "spatiotemporal_data_1.npy",
        "spatiotemporal_data_2.npy",
    ]
    spatial_file_name = "spatial_data.npy"
    lu_index_file_name = "lu_index.npy"
    batch_size = 4
    time_steps_per_day = 6

    # Mock data for GCS
    spatiotemporal_data_1 = np.random.rand(8, 100, 100, 1)
    spatiotemporal_data_2 = np.random.rand(8, 100, 100, 1)
    spatial_data = np.random.rand(100, 100, 1)
    lu_index_data = np.random.randint(1, 62, (100, 100))

    def mock_blob_func(blob_name):
        """Returns the mock data depending on the blob requested."""
        if blob_name == "spatiotemporal_data_1.npy":
            mock_data = spatiotemporal_data_1
        elif blob_name == "spatiotemporal_data_2.npy":
            mock_data = spatiotemporal_data_2
        elif blob_name == "spatial_data.npy":
            mock_data = spatial_data
        elif blob_name == "lu_index.npy":
            mock_data = lu_index_data
        else:
            raise ValueError(f"Unexpected name {blob_name} passed to mock.")

        buf = io.BytesIO()
        np.save(buf, mock_data)
        buf.seek(0)

        mock_blob = mock.MagicMock(spec=storage.Blob)
        mock_blob.open.return_value = buf
        return mock_blob

    # Set the storage client to return the right mock data for each GCS path.
    mock_storage_client().bucket().blob.side_effect = mock_blob_func

    # Load prediction dataset and iterate through the generator
    dataset_generator = datasets.load_prediction_dataset(
        bucket_name=bucket_name,
        spatiotemporal_file_names=spatiotemporal_file_names,
        spatial_file_name=spatial_file_name,
        lu_index_file_name=lu_index_file_name,
        batch_size=batch_size,
        time_steps_per_day=time_steps_per_day,
        storage_client=mock_storage_client(),
    )

    # Check that the generator yields correctly batched data
    for i, inputs in enumerate(dataset_generator):
        spatiotemporal_batch = inputs["spatiotemporal"]
        spatial_batch = inputs["spatial"]
        lu_index_batch = inputs["lu_index"]

        assert spatiotemporal_batch.shape == (
            batch_size,
            100,
            100,
            1,
        ), "Spatiotemporal shape mismatch"
        assert spatial_batch.shape == (100, 100, 1), "Spatial data shape mismatch"
        assert lu_index_batch.shape == (100 * 100,), "LU index shape mismatch"

        print(f"Batch {i + 1} spatiotemporal shape: {spatiotemporal_batch.shape}")
        print(f"Batch {i + 1} spatial shape: {spatial_batch.shape}")
        print(f"Batch {i + 1} LU index shape: {lu_index_batch.shape}")

import io
from unittest import mock

from google.cloud import storage
import numpy

from usl_models.atmo_ml import datasets
from usl_models.atmo_ml import cnn_inputs_outputs


@mock.patch.object(datasets, "create_atmo_dataset")
def test_atmo_dataset(mock_create_dataset) -> None:
    """Create expected dataset for the atmo model from GCS objects."""
    mock_storage_client = mock.MagicMock(spec=storage.Client)

    # Set parameters for the dataset
    bucket_name = "your-bucket-name"
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
    stinp_1 = numpy.random.rand(batch_size, 100, 100, 1)
    labelinput_1 = numpy.random.rand(batch_size, 100, 100, 1)
    stinp_2 = numpy.random.rand(batch_size, 100, 100, 1)
    labelinput_2 = numpy.random.rand(batch_size, 100, 100, 1)

    # Call the function to process inputs and labels
    mock_spatiotemporal_data_1, mock_labels_1 = cnn_inputs_outputs.divide_into_days(
        stinp_1, labelinput_1, input_steps_per_day=4, label_steps_per_day=8
    )
    mock_spatiotemporal_data_2, mock_labels_2 = cnn_inputs_outputs.divide_into_days(
        stinp_2, labelinput_2, input_steps_per_day=4, label_steps_per_day=8
    )
    mock_spatial_data = numpy.random.rand(100, 100, 1)
    mock_lu_index_data = numpy.random.randint(1, 62, (100, 100))

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
        numpy.save(buf, mock_data)
        buf.seek(0)

        mock_blob = mock.MagicMock(spec=storage.Blob)
        mock_blob.open.return_value = buf
        return mock_blob

    # Set the storage client to return the right mock data for each GCS path.
    mock_storage_client.bucket().blob.side_effect = mock_blob_func

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
    )

    # Iterate through the dataset and check the shape
    for data in dataset:
        inputs, labels = data
        assert inputs.shape == (
            batch_size,
            time_steps_per_day,
            100,
            100,
            1,
        ), "Input shape mismatch"
        assert labels.shape == (
            batch_size,
            time_steps_per_day,
            100,
            100,
            1,
        ), "Label shape mismatch"
        print("Inputs:", inputs)
        print("Labels:", labels)

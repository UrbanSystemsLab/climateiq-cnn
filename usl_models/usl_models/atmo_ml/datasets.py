import tensorflow as tf
import numpy as np
from google.cloud import storage
from usl_models.atmo_ml import constants
from usl_models.atmo_ml import cnn_inputs_outputs


# Load data from Google Cloud Storage instead of local file paths
def load_data_from_cloud(
    bucket_name: str, file_name: str, storage_client: storage.Client
):
    """Load data from Google Cloud Storage."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    downloaded_data = blob.download_as_bytes()
    return np.load(downloaded_data)


def load_spatiotemporal_data_from_cloud(
    bucket_name: str, file_name: str, storage_client: storage.Client
) -> tf.Tensor:
    """Load spatiotemporal data from Google Cloud Storage."""
    data = load_data_from_cloud(bucket_name, file_name, storage_client)
    return tf.convert_to_tensor(data, dtype=tf.float32)


def load_labels_from_cloud(
    bucket_name: str, file_name: str, storage_client: storage.Client
) -> tf.Tensor:
    """Load labels from Google Cloud Storage."""
    labels = load_data_from_cloud(bucket_name, file_name, storage_client)
    return tf.convert_to_tensor(labels, dtype=tf.float32)


def load_lu_index_from_cloud(
    bucket_name: str, file_name: str, storage_client: storage.Client
) -> tf.Tensor:
    """Load land use index (categorical) from Google Cloud Storage."""
    lu_index_data = load_data_from_cloud(bucket_name, file_name, storage_client)
    return tf.convert_to_tensor(lu_index_data, dtype=tf.int32)


def load_spatial_data_from_cloud(
    bucket_name: str, file_name: str, storage_client: storage.Client
) -> tf.Tensor:
    """Load spatial data from Google Cloud Storage."""
    spatial_data = load_data_from_cloud(bucket_name, file_name, storage_client)
    return tf.convert_to_tensor(spatial_data, dtype=tf.float32)


def create_atmo_dataset(
    bucket_name: str,
    spatiotemporal_file_names: list[str],
    label_file_names: list[str],
    spatial_file_name: str,  # Adding the spatial feature file
    lu_index_file_name: str,
    time_steps_per_day: int,
    batch_size: int = 32,
    shuffle: bool = True,
    storage_client: storage.Client = None,
) -> tf.data.Dataset:
    """Creates the dataset for the AtmoML model.

    Args:
        bucket_name: The GCS bucket name.
        spatiotemporal_file_names: List of file names for spatiotemporal in GCS.
        label_file_names: List of file names for corresponding labels in the GCS.
        spatial_file_name: File name for spatial features (static data) in the GCS.
        lu_index_file_name: File name for the land use index data in the GCS.
        time_steps_per_day: Number of time steps to divide per day.
        batch_size: Batch size for the dataset.
        shuffle: Whether to shuffle the dataset.
        storage_client: An instance of Google Cloud Storage client.

    Returns:
        A tf.data.Dataset object yielding input dictionaries compatible with AtmoML.
    """

    def data_generator():
        lu_index_data = load_lu_index_from_cloud(
            bucket_name, lu_index_file_name, storage_client
        )
        spatial_data = load_spatial_data_from_cloud(
            bucket_name, spatial_file_name, storage_client
        )

        for spatiotemporal_file_name, label_file_name in zip(
            spatiotemporal_file_names, label_file_names
        ):
            spatiotemporal_data = load_spatiotemporal_data_from_cloud(
                bucket_name, spatiotemporal_file_name, storage_client
            )
            label_data = load_labels_from_cloud(
                bucket_name, label_file_name, storage_client
            )

            # Divide the spatiotemporal data and labels into days
            inputs, labels = cnn_inputs_outputs.divide_into_days(
                spatiotemporal_data, label_data, time_steps_per_day
            )

            # Yield each day's input and label as a batch
            for day_inputs, day_labels in zip(inputs, labels):
                yield {
                    "spatiotemporal": day_inputs,  # shape: (t,h,w,f)
                    "spatial": spatial_data,  # shape: (height, width, spatial_features)
                    "lu_index": lu_index_data,  # shape: (height * width,)
                }, day_labels  # shape: (time_steps_per_day, height, width, 1)

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            {
                "spatiotemporal": tf.TensorSpec(
                    shape=(
                        time_steps_per_day,
                        constants.MAP_HEIGHT,
                        constants.MAP_WIDTH,
                        constants.num_spatiotemporal_features,
                    ),
                    dtype=tf.float32,
                ),
                "spatial": tf.TensorSpec(
                    shape=(
                        constants.MAP_HEIGHT,
                        constants.MAP_WIDTH,
                        constants.num_spatial_features,
                    ),
                    dtype=tf.float32,
                ),
                "lu_index": tf.TensorSpec(
                    shape=(constants.MAP_HEIGHT * constants.MAP_WIDTH,),
                    dtype=tf.int32,
                ),
            },
            tf.TensorSpec(
                shape=(
                    time_steps_per_day,
                    constants.MAP_HEIGHT,
                    constants.MAP_WIDTH,
                    1,
                ),
                dtype=tf.float32,
            ),
        ),
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(spatiotemporal_file_names))

    dataset = dataset.batch(batch_size)

    return dataset

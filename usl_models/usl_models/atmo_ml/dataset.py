import tensorflow as tf
import numpy as np
import io
from google.cloud import storage, firestore  # type: ignore
from usl_models.atmo_ml import constants, cnn_inputs_outputs


# Load data from Google Cloud Storage with Firestore logging
def load_data_from_cloud(
    bucket_name: str,
    path: str,
    storage_client: storage.Client,
    firestore_client: firestore.Client = None,
    is_folder: bool = False,
):
    """Load data from Google Cloud Storage, with optional Firestore logging.

    Args:
        bucket_name (str): Name of the Google Cloud Storage bucket.
        path (str): Path to the file or folder in the bucket.
        storage_client (storage.Client): Google Cloud Storage client instance.
        firestore_client (firestore.Client, optional): Firestore client for logging.
        is_folder (bool): If True, load all .npy files in the folder.

    Returns:
        Union[np.ndarray, List[np.ndarray]]: Loaded numpy data or list of arrays.
    """
    bucket = storage_client.bucket(bucket_name)

    if is_folder:
        # List all blobs within the folder
        blobs = bucket.list_blobs(prefix=path)
        all_data = []
        for blob in blobs:
            if blob.name.endswith(".npy"):  # Ensure only .npy files are processed
                downloaded_data = blob.download_as_bytes()
                np_data = np.load(io.BytesIO(downloaded_data))
                all_data.append(np_data)
        return all_data  # Return a list of numpy arrays
    else:
        # Load a single file
        blobs = bucket.list_blobs(prefix=path)
        blob = next(blobs)
        downloaded_data = blob.download_as_bytes()
        return np.load(io.BytesIO(downloaded_data))


def load_spatiotemporal_data_from_cloud(
    bucket_name: str,
    folder_name: str,
    storage_client: storage.Client,
    firestore_client: firestore.Client = None,
) -> tf.Tensor:
    """Load all spatiotemporal files from a specified folder in Google Cloud Storage.

    Args:
        bucket_name (str): Name of the Google Cloud Storage bucket.
        folder_name (str): Name of the folder containing time step numpy files.
        storage_client (storage.Client): Google Cloud Storage client instance.
        firestore_client (firestore.Client, optional): Firestore client for logging.

    Returns:
        tf.Tensor: A tensor containing spatiotemporal data across all time steps.
    """
    # Use load_data_from_cloud with is_folder=True to load all .npy files in the folder
    time_step_data = load_data_from_cloud(
        bucket_name=bucket_name,
        path=folder_name,
        storage_client=storage_client,
        firestore_client=firestore_client,
        is_folder=True,
    )

    # Stack all time steps along a new axis to create a tensor
    spatiotemporal_data = tf.convert_to_tensor(
        np.stack(time_step_data), dtype=tf.float32
    )
    return spatiotemporal_data


def load_labels_from_cloud(
    bucket_name: str,
    folder_name: str,
    storage_client: storage.Client,
    firestore_client: firestore.Client = None,
) -> tf.Tensor:
    """Load all label files from a specified folder in GCS.

    Combine the label data into a single tensor.

    Args:
        bucket_name (str): Name of the Google Cloud Storage bucket.
        folder_name (str): Name of the folder containing label numpy files.
        storage_client (storage.Client): Google Cloud Storage client instance.
        firestore_client (firestore.Client, optional): Firestore client for logging,
            if needed.

    Returns:
        tf.Tensor: A tensor containing label data across all time steps.
    """
    # Use load_data_from_cloud with is_folder=True to load all .npy files in the folder
    label_data = load_data_from_cloud(
        bucket_name=bucket_name,
        path=folder_name,
        storage_client=storage_client,
        firestore_client=firestore_client,
        is_folder=True,
    )

    # Stack all label data along a new axis to create a tensor
    labels_tensor = tf.convert_to_tensor(np.stack(label_data), dtype=tf.float32)
    return labels_tensor


def load_lu_index_from_cloud(
    bucket_name: str,
    folder_name: str,
    storage_client: storage.Client,
    firestore_client: firestore.Client = None,
) -> tf.Tensor:
    """Load land use index from Google Cloud Storage with Firestore logging."""
    lu_index_data = load_data_from_cloud(
        bucket_name, folder_name, storage_client, firestore_client
    )
    return tf.convert_to_tensor(lu_index_data, dtype=tf.int32)


def load_spatial_data_from_cloud(
    bucket_name: str,
    folder_name: str,
    storage_client: storage.Client,
    firestore_client: firestore.Client = None,
) -> tf.Tensor:
    """Load spatial data from Google Cloud Storage with Firestore logging."""
    spatial_data = load_data_from_cloud(
        bucket_name, folder_name, storage_client, firestore_client
    )
    return tf.convert_to_tensor(spatial_data, dtype=tf.float32)


def load_dataset(
    data_bucket_name: str,
    label_bucket_name: str,
    sim_names: list[str],
    time_steps_per_day: int,
    batch_size: int = 4,
    shuffle: bool = True,
    storage_client: storage.Client = None,
    firestore_client: firestore.Client = None,
) -> tf.data.Dataset:
    firestore_client = firestore_client or firestore.Client()
    storage_client = storage_client or storage.Client()

    def data_generator():
        for sim_name in sim_names:

            spatiotemporal_folder = f"{sim_name}/spatiotemporal"
            spatial_folder = f"{sim_name}/spatial"
            lu_index_folder = f"{sim_name}/lu_index"
            label_folder = sim_name

            # Load spatial, LU index, spatiotemporal, and label from their folders
            lu_index_data = load_lu_index_from_cloud(
                data_bucket_name, lu_index_folder, storage_client, firestore_client
            )
            spatial_data = load_spatial_data_from_cloud(
                data_bucket_name, spatial_folder, storage_client, firestore_client
            )
            spatiotemporal_data = load_spatiotemporal_data_from_cloud(
                data_bucket_name,
                spatiotemporal_folder,
                storage_client,
                firestore_client,
            )
            label_data = load_labels_from_cloud(
                label_bucket_name, label_folder, storage_client, firestore_client
            )

            # Iterate through each spatiotemporal and label file
            # Divide data into days and apply padding or truncation
            inputs, labels = cnn_inputs_outputs.divide_into_days(
                spatiotemporal_data, label_data, time_steps_per_day
            )
            for day_inputs, day_labels in zip(inputs, labels):
                day_inputs_padded = pad_or_truncate_data(
                    day_inputs.numpy(), time_steps_per_day
                )
                day_labels_padded = pad_or_truncate_data(
                    day_labels.numpy(), time_steps_per_day
                )
                yield {
                    "spatiotemporal": day_inputs_padded,
                    "spatial": spatial_data.numpy(),
                    "lu_index": lu_index_data.numpy(),
                }, day_labels_padded

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
        dataset = dataset.shuffle(buffer_size=1000)  # Use a fixed buffer size

    dataset = dataset.batch(batch_size)
    return dataset


def load_prediction(
    data_bucket_name: str,
    spatiotemporal_folder: str,
    spatial_folder: str,
    lu_index_folder: str,
    time_steps_per_day: int,
    batch_size: int = 4,
    storage_client: storage.Client = None,
    firestore_client: firestore.Client = None,
) -> tf.data.Dataset:

    # Load spatial, LU index, and spatiotemporal data from their respective folders
    lu_index_data = load_lu_index_from_cloud(
        data_bucket_name, lu_index_folder, storage_client, firestore_client
    )
    spatial_data = load_spatial_data_from_cloud(
        data_bucket_name, spatial_folder, storage_client, firestore_client
    )
    spatiotemporal_data_list = load_spatiotemporal_data_from_cloud(
        data_bucket_name, spatiotemporal_folder, storage_client, firestore_client
    )

    def data_generator():
        # Iterate through each spatiotemporal data file
        for spatiotemporal_data in spatiotemporal_data_list:
            # Divide data into days and apply padding or truncation
            # Divide the spatiotemporal data into daily inputs
            inputs, _ = cnn_inputs_outputs.divide_into_days(
                spatiotemporal_data, labels=None
            )
            for day_inputs in inputs:
                day_inputs_padded = pad_or_truncate_data(
                    day_inputs.numpy(), time_steps_per_day
                )
                yield {
                    "spatiotemporal": day_inputs_padded,
                    "spatial": spatial_data.numpy(),
                    "lu_index": lu_index_data.numpy(),
                }

    # Create the dataset using the data generator
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
        ),
    )

    # Batch the dataset
    dataset = dataset.batch(batch_size)
    return dataset


def make_predictions(model: tf.keras.Model, dataset: tf.data.Dataset) -> np.ndarray:
    """Make predictions using the AtmoML model on the provided dataset.

    Args:

        model: The trained AtmoML model.
        dataset: The dataset to predict on.

    Returns:
        A numpy array of predictions.
    """
    predictions = model.predict(dataset)
    return predictions


# Pad or truncate data to a target length along the first axis.
def pad_or_truncate_data(data, target_length, pad_value=0):
    """Pad or truncate data to the target length along the first axis."""
    current_length = data.shape[0]
    if current_length > target_length:
        return data[:target_length]
    elif current_length < target_length:
        pad_shape = [target_length - current_length] + list(data.shape[1:])
        padding = np.full(pad_shape, pad_value, dtype=data.dtype)
        return np.concatenate([data, padding], axis=0)
    return data


def split_dataset(dataset, train_frac=0.7, val_frac=0.15, test_frac=0.15):
    """Splits a tf.data.Dataset into training, validation, and test sets.

    Args:
        dataset (tf.data.Dataset): The dataset to split.
        train_frac (float): Fraction of the dataset to use for training.
        val_frac (float): Fraction of the dataset to use for validation.
        test_frac (float): Fraction of the dataset to use for testing.

    Returns:
        train, validation, and test data.
    """
    assert train_frac + val_frac + test_frac == 1, "Fractions must sum to 1."

    total_size = sum(1 for _ in dataset)  # Calculate the total dataset size
    train_size = int(train_frac * total_size)
    val_size = int(val_frac * total_size)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size)

    return train_dataset, val_dataset, test_dataset
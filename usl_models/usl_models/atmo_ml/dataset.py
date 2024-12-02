import functools

import tensorflow as tf
import numpy as np
from google.cloud import storage  # type: ignore
from usl_models.atmo_ml import constants, cnn_inputs_outputs
from usl_models.shared import downloader


# Load data from Google Cloud Storage
@functools.lru_cache(maxsize=256)
def load_folder_from_cloud(
    bucket_name: str,
    path: str,
    storage_client: storage.Client,
    max_blobs: int | None = None,
) -> tf.Tensor:
    """Download all files in the folder.

    If max_blobs is specified, only returns that many blobs.
    """
    bucket = storage_client.bucket(bucket_name)
    # List all blobs within the folder
    blobs = bucket.list_blobs(prefix=path)
    all_data = []
    blob_count = 0
    for blob in blobs:
        blob_count += 1
        if max_blobs is not None and blob_count > max_blobs:
            break

        if blob.name.endswith(".npy"):  # Ensure only .npy files are processed
            all_data.append(downloader.blob_to_tensor(blob))

    return tf.stack(all_data)


def load_dataset(
    data_bucket_name: str,
    label_bucket_name: str,
    sim_names: list[str],
    timesteps_per_day: int,
    shuffle: bool = True,
    storage_client: storage.Client = None,
    max_blobs: int | None = None,
) -> tf.data.Dataset:
    storage_client = storage_client or storage.Client()

    # Early validation
    assert storage_client.bucket(
        label_bucket_name
    ).exists(), f"Bucket does not exist: {label_bucket_name}"
    assert storage_client.bucket(
        data_bucket_name
    ).exists(), f"Bucket does not exist: {data_bucket_name}"

    label_timesteps = 2 * (timesteps_per_day - 2)

    def data_generator():
        for sim_name in sim_names:

            spatiotemporal_folder = f"{sim_name}/spatiotemporal"
            spatial_folder = f"{sim_name}/spatial"
            lu_index_folder = f"{sim_name}/lu_index"
            label_folder = sim_name

            # Load spatial, LU index, spatiotemporal, and label from their folders
            lu_index_data = load_folder_from_cloud(
                data_bucket_name, lu_index_folder, storage_client, max_blobs=1
            )[0]
            spatial_data = load_folder_from_cloud(
                data_bucket_name, spatial_folder, storage_client, max_blobs=1
            )[0]
            spatiotemporal_data = load_folder_from_cloud(
                data_bucket_name,
                spatiotemporal_folder,
                storage_client,
                max_blobs=max_blobs,
            )
            label_data = load_folder_from_cloud(
                label_bucket_name, label_folder, storage_client,
                max_blobs=max_blobs
            )

            # Iterate through each spatiotemporal and label file
            # Divide data into days and apply padding or truncation
            inputs, labels = cnn_inputs_outputs.divide_into_days(
                spatiotemporal_data, label_data, timesteps_per_day
            )
            for day_inputs, day_labels in zip(inputs, labels):
                day_inputs_padded = pad_or_truncate_data(
                    day_inputs.numpy(), timesteps_per_day
                )
                day_labels_padded = pad_or_truncate_data(
                    day_labels.numpy(), label_timesteps
                )
                yield {
                    "spatiotemporal": day_inputs_padded,
                    "spatial": spatial_data.numpy(),
                    "lu_index": lu_index_data.numpy().reshape(200, 200),
                }, day_labels_padded

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            {
                "spatiotemporal": tf.TensorSpec(
                    shape=(
                        timesteps_per_day,
                        constants.MAP_HEIGHT,
                        constants.MAP_WIDTH,
                        constants.NUM_SPATIOTEMPORAL_FEATURES,
                    ),
                    dtype=tf.float32,
                ),
                "spatial": tf.TensorSpec(
                    shape=(
                        constants.MAP_HEIGHT,
                        constants.MAP_WIDTH,
                        constants.NUM_SAPTIAL_FEATURES,
                    ),
                    dtype=tf.float32,
                ),
                "lu_index": tf.TensorSpec(
                    shape=(
                        constants.MAP_HEIGHT,
                        constants.MAP_WIDTH,
                    ),
                    dtype=tf.float32,
                ),
            },
            tf.TensorSpec(
                shape=(
                    # Inputs have an extra 2 timesteps from the previous day.
                    # Output has double the resolution.
                    label_timesteps,
                    constants.MAP_HEIGHT,
                    constants.MAP_WIDTH,
                    constants.OUTPUT_CHANNELS,
                ),
                dtype=tf.float32,
            ),
        ),
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)  # Use a fixed buffer size

    return dataset


def load_prediction_dataset(
    data_bucket_name: str,
    spatiotemporal_folder: str,
    spatial_folder: str,
    lu_index_folder: str,
    timesteps_per_day: int,
    batch_size: int = 4,
    storage_client: storage.Client = None,
    max_blobs: int | None = None
) -> tf.data.Dataset:

    # Load spatial, LU index, and spatiotemporal data from their respective folders
    lu_index_data = load_folder_from_cloud(
        data_bucket_name, lu_index_folder, storage_client, max_blobs=1
    )[0]
    spatial_data = load_folder_from_cloud(
        data_bucket_name, spatial_folder, storage_client, max_blobs=1
    )[0]
    spatiotemporal_data_list = load_folder_from_cloud(
        data_bucket_name, spatiotemporal_folder, storage_client, max_blobs=max_blobs
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
                    day_inputs.numpy(), timesteps_per_day
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
                        timesteps_per_day,
                        constants.MAP_HEIGHT,
                        constants.MAP_WIDTH,
                        constants.NUM_SPATIOTEMPORAL_FEATURES,
                    ),
                    dtype=tf.float32,
                ),
                "spatial": tf.TensorSpec(
                    shape=(
                        constants.MAP_HEIGHT,
                        constants.MAP_WIDTH,
                        constants.NUM_SAPTIAL_FEATURES,
                    ),
                    dtype=tf.float32,
                ),
                "lu_index": tf.TensorSpec(
                    shape=(
                        constants.MAP_HEIGHT,
                        constants.MAP_WIDTH,
                    ),
                    dtype=tf.float32,
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

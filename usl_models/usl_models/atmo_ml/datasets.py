import tensorflow as tf
import numpy as np
import io
from typing import Iterator
from google.cloud import storage, firestore  # type: ignore
from usl_models.atmo_ml import constants, cnn_inputs_outputs


# Load data from Google Cloud Storage with Firestore logging
def load_data_from_cloud(
    bucket_name: str,
    file_name: str,
    storage_client: storage.Client,
    firestore_client: firestore.Client = None,
):
    """Load data from Google Cloud Storage, with optional Firestore logging."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    downloaded_data = blob.download_as_bytes()
    # Log progress to Firestore (if Firestore client is provided)
    if firestore_client:
        doc_ref = firestore_client.collection("dataset_loading").document(file_name)
        doc_ref.set({"file_name": file_name, "status": "loaded"})
    return np.load(io.BytesIO(downloaded_data))


def load_spatiotemporal_data_from_cloud(
    bucket_name: str,
    file_name: str,
    storage_client: storage.Client,
    firestore_client: firestore.Client = None,
) -> tf.Tensor:
    """Load spatiotemporal data from Google Cloud Storage with Firestore logging."""
    data = load_data_from_cloud(
        bucket_name, file_name, storage_client, firestore_client
    )
    return tf.convert_to_tensor(data, dtype=tf.float32)


def load_labels_from_cloud(
    bucket_name: str,
    file_name: str,
    storage_client: storage.Client,
    firestore_client: firestore.Client = None,
) -> tf.Tensor:
    """Load labels from Google Cloud Storage with Firestore logging."""
    labels = load_data_from_cloud(
        bucket_name, file_name, storage_client, firestore_client
    )
    return tf.convert_to_tensor(labels, dtype=tf.float32)


def load_lu_index_from_cloud(
    bucket_name: str,
    file_name: str,
    storage_client: storage.Client,
    firestore_client: firestore.Client = None,
) -> tf.Tensor:
    """Load land use index from Google Cloud Storage with Firestore logging."""
    lu_index_data = load_data_from_cloud(
        bucket_name, file_name, storage_client, firestore_client
    )
    return tf.convert_to_tensor(lu_index_data, dtype=tf.int32)


def load_spatial_data_from_cloud(
    bucket_name: str,
    file_name: str,
    storage_client: storage.Client,
    firestore_client: firestore.Client = None,
) -> tf.Tensor:
    """Load spatial data from Google Cloud Storage with Firestore logging."""
    spatial_data = load_data_from_cloud(
        bucket_name, file_name, storage_client, firestore_client
    )
    return tf.convert_to_tensor(spatial_data, dtype=tf.float32)


def create_atmo_dataset(
    bucket_name: str,
    spatiotemporal_file_names: list[str],
    label_file_names: list[str],
    spatial_file_name: str,
    lu_index_file_name: str,
    time_steps_per_day: int,
    batch_size: int = 32,
    shuffle: bool = True,
    storage_client: storage.Client = None,
    firestore_client: firestore.Client = None,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Creates the dataset for the AtmoML model with optional Firestore logging.

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
        firestore_client: An optional Firestore client to log data loading status.

    Returns:
        A tuple of tf.data.Dataset objects for train, validation, and test sets.
    """

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

    def data_generator():
        lu_index_data = load_lu_index_from_cloud(
            bucket_name, lu_index_file_name, storage_client, firestore_client
        )
        spatial_data = load_spatial_data_from_cloud(
            bucket_name, spatial_file_name, storage_client, firestore_client
        )
        for spatiotemporal_file_name, label_file_name in zip(
            spatiotemporal_file_names, label_file_names
        ):
            spatiotemporal_data = load_spatiotemporal_data_from_cloud(
                bucket_name, spatiotemporal_file_name, storage_client, firestore_client
            )
            label_data = load_labels_from_cloud(
                bucket_name, label_file_name, storage_client, firestore_client
            )
            # Divide the spatiotemporal data and labels into days
            inputs, labels = cnn_inputs_outputs.divide_into_days(
                spatiotemporal_data, label_data, time_steps_per_day
            )
            # Pad or truncate each day's input and label
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
        dataset = dataset.shuffle(buffer_size=len(spatiotemporal_file_names))
    dataset = dataset.batch(batch_size)
    # Split the dataset into train, validation, and test sets
    total_size = len(spatiotemporal_file_names)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size).take(test_size)
    return train_dataset, val_dataset, test_dataset


def load_prediction_dataset(
    sim_names: list[str],
    batch_size: int = 4,
    max_chunks: int | None = None,
    firestore_client: firestore.Client = None,
    storage_client: storage.Client = None,
) -> tf.data.Dataset:
    """Creates a dataset for predictions without labels."""
    firestore_client = firestore_client or firestore.Client()
    storage_client = storage_client or storage.Client()

    def generator():
        """Generator for producing only inputs (no labels) for predictions."""
        for sim_name in sim_names:
            for model_input in _iter_model_inputs_for_prediction(
                firestore_client,
                storage_client,
                sim_name,
                max_chunks,
            ):
                yield model_input

    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_signature=dict(
            spatiotemporal=_spatiotemporal_dataset_signature(),
            spatial=_spatial_dataset_signature(),
            lu_index=_lu_index_signature(),
        ),
    )
    if batch_size:
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


def _iter_model_inputs_for_prediction(
    firestore_client: firestore.Client,
    storage_client: storage.Client,
    sim_name: str,
    max_chunks: int | None,
) -> Iterator[dict]:
    """Yields model inputs for prediction, no labels."""
    lu_index_data = load_lu_index_from_cloud(
        sim_name, "lu_index.npy", storage_client, firestore_client
    )
    spatial_data = load_spatial_data_from_cloud(
        sim_name, "spatial_data.npy", storage_client, firestore_client
    )
    spatiotemporal_data = load_spatiotemporal_data_from_cloud(
        sim_name, "spatiotemporal_data.npy", storage_client, firestore_client
    )

    # Divide the spatiotemporal data into daily inputs
    spatiotemporal_inputs, _ = cnn_inputs_outputs.divide_into_days(
        spatiotemporal_data, labels=None
    )
    for st_input in spatiotemporal_inputs:
        model_input = {
            "spatiotemporal": st_input,
            "spatial": spatial_data,
            "lu_index": lu_index_data,
        }
        yield model_input


def _spatiotemporal_dataset_signature():
    """Defines the spatiotemporal dataset signature."""
    return tf.TensorSpec(
        shape=(
            constants.TIME_STEPS_PER_DAY,
            constants.MAP_HEIGHT,
            constants.MAP_WIDTH,
            constants.num_spatiotemporal_features,
        ),
        dtype=tf.float32,
    )


def _spatial_dataset_signature():
    """Defines the spatial dataset signature."""
    return tf.TensorSpec(
        shape=(
            constants.MAP_HEIGHT,
            constants.MAP_WIDTH,
            constants.num_spatial_features,
        ),
        dtype=tf.float32,
    )


def _lu_index_signature():
    """Defines the land-use index dataset signature."""
    return tf.TensorSpec(
        shape=(constants.MAP_HEIGHT * constants.MAP_WIDTH,),
        dtype=tf.int32,
    )

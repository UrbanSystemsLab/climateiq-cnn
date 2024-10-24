import tensorflow as tf
import numpy as np
from google.cloud import storage, firestore  # type: ignore
from usl_models.atmo_ml import constants
from usl_models.atmo_ml import cnn_inputs_outputs


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

    return np.load(downloaded_data)


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


def load_prediction_dataset(
    bucket_name: str,
    spatiotemporal_file_names: list,
    spatial_file_name: str,
    lu_index_file_name: str,
    batch_size: int,
    time_steps_per_day: int,
    storage_client: storage.Client,
):
    """Load prediction data from GCS and generate batches for predictions.

    Args:
        bucket_name: Name of the GCS bucket.
        spatiotemporal_file_names: List of GCS paths for spatiotemporal.
        spatial_file_name: GCS path for spatial data.
        lu_index_file_name: GCS path for LU index.
        batch_size: Batch size for prediction.
        time_steps_per_day: Number of time steps per day for spatiotemporal data.
        storage_client: The GCS client.

    Yields:
        A batch of prediction inputs.
    """
    bucket = storage_client.bucket(bucket_name)

    # Load spatial and LU index data
    spatial_blob = bucket.blob(spatial_file_name)
    lu_index_blob = bucket.blob(lu_index_file_name)

    spatial_data = np.load(spatial_blob.open("rb"))
    lu_index_data = np.load(lu_index_blob.open("rb"))

    # Load spatiotemporal data in batches for prediction
    for st_file_name in spatiotemporal_file_names:
        spatiotemporal_blob = bucket.blob(st_file_name)
        spatiotemporal_data = np.load(spatiotemporal_blob.open("rb"))

        # Split the spatiotemporal data into batches
        for i in range(0, spatiotemporal_data.shape[0], batch_size):
            batch_spatiotemporal = spatiotemporal_data[i : i + batch_size]

            # Yield the batch of inputs
            inputs = {
                "spatiotemporal": batch_spatiotemporal,
                "spatial": spatial_data,
                "lu_index": lu_index_data.flatten(),
            }
            yield inputs


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
) -> tf.data.Dataset:
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
        A tf.data.Dataset object yielding input dictionaries compatible with AtmoML.
    """

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

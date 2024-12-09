import functools
import hashlib  # For hashing days
import tensorflow as tf
import numpy as np
from google.cloud import storage  # type: ignore
from usl_models.atmo_ml import cnn_inputs_outputs
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


def hash_day(timestamp: str) -> float:
    """Hash a timestamp into a float between 0 and 1.

    Ensure that all timestamps for the same day hash to the same value.

    Args:
        timestamp (str): A string in the format 'met_em.d03.2000-05-24_00:00:00.npy'
                        or 'wrfout_d03_2000-05-24_00:00:00.npy'.

    Returns:
        float: A hash value between 0 and 1 for the day.
    """
    # Extract the date part (e.g., '2000-05-24') from the timestamp
    date_part = timestamp.split("_")[1]

    # Hash the date part to ensure all timestamps for the same day are consistent
    return int(hashlib.sha256(date_part.encode()).hexdigest(), 16) % (10**8) / (10**8)


def select_days(
    all_days: list[str], train_range=(0.0, 0.8), val_range=(0.8, 1)
) -> tuple[list[str], list[str]]:
    """Select days for training and validation based on hash ranges."""
    train_days = [
        day for day in all_days if train_range[0] <= hash_day(day) < train_range[1]
    ]
    val_days = [
        day for day in all_days if val_range[0] <= hash_day(day) < val_range[1]
    ]
    return train_days, val_days


def get_all_simulation_days(sim_names: list[str], storage_client, bucket_name: str) -> list[str]:
    """Retrieve all simulation days from simulation names."""
    all_days = set()
    bucket = storage_client.bucket(bucket_name)

    for sim_name in sim_names:
        # List blobs under the simulation folder
        blobs = bucket.list_blobs(prefix=f"{sim_name}/")
        for blob in blobs:
            # Extract the date from the filename (e.g., "2000-05-24.npy")
            filename = blob.name.split("/")[-1]
            if filename.endswith(".npy"):
                date_part = filename.split("_")[1]
                all_days.add(date_part)

    return sorted(all_days)


def load_dataset(
    data_bucket_name: str,
    label_bucket_name: str,
    sim_names: list[str],
    timesteps_per_day: int,
    shuffle: bool = True,
    storage_client: storage.Client = None,
    max_blobs: int | None = None,
    train_range=(0.0, 0.5),
    val_range=(0.6, 0.8),
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    storage_client = storage_client or storage.Client()

    # Early validation
    assert storage_client.bucket(
        label_bucket_name
    ).exists(), f"Bucket does not exist: {label_bucket_name}"
    assert storage_client.bucket(
        data_bucket_name
    ).exists(), f"Bucket does not exist: {data_bucket_name}"

    label_timesteps = 2 * (timesteps_per_day - 2)
    all_days = get_all_simulation_days(
    sim_names=sim_names, 
    storage_client=storage_client, 
    bucket_name=data_bucket_name
    )
    train_days, val_days = select_days(all_days, train_range, val_range)

    def data_generator(selected_days):
        for day in selected_days:
            print(f"Processing data for day: {day}")
            # Simulate downloading and processing for a specific day
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
                    label_bucket_name, label_folder, storage_client, max_blobs=max_blobs
                )

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

    def create_dataset(selected_days):
        dataset = tf.data.Dataset.from_generator(
            lambda: data_generator(selected_days),
            output_signature=(...),
        )
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        return dataset

    train_dataset = create_dataset(train_days)
    val_dataset = create_dataset(val_days)

    return train_dataset, val_dataset


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

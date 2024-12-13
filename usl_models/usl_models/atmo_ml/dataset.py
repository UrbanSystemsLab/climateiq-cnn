import logging
import random

import functools
import hashlib  # For hashing days
import tensorflow as tf
import numpy as np
from google.cloud import storage  # type: ignore
from usl_models.atmo_ml import cnn_inputs_outputs, constants
from usl_models.shared import downloader
import re  # To extract dates from filenames


# Load data from Google Cloud Storage
@functools.lru_cache(maxsize=512)
def load_pattern_from_cloud(
    bucket_name: str,
    prefix: str,
    storage_client: storage.Client,
    max_blobs: int | None = None,
) -> tf.Tensor:
    """Download all files in the folder.

    If max_blobs is specified, only returns that many blobs.
    """
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    all_data = []
    blob_count = 0
    for blob in blobs:
        blob_count += 1
        if max_blobs is not None and blob_count > max_blobs:
            break

        if blob.name.endswith(".npy"):  # Ensure only .npy files are processed
            all_data.append(downloader.blob_to_tensor(blob))

    return tf.stack(all_data)


def get_date(filename: str) -> str:
    return filename.split(".")[2].split("_")[0]


def hash_day(sim_name: str, date: str) -> float:
    """Hash a timestamp into a float between 0 and 1.

    Ensure that all timestamps for the same day hash to the same value.

    Args:
        filepath (str): A string in the format 'met_em.d03.2000-05-24_00:00:00.npy'
                        or 'wrfout_d03_2000-05-24_00:00:00.npy'.

    Returns:
        float: A hash value between 0 and 1 for the day.
    """
    # Hash the date part to ensure all timestamps for the same day are consistent
    return (
        int(hashlib.sha256((sim_name + date).encode()).hexdigest(), 16)
        % (10**8)
        / (10**8)
    )


def get_all_simulation_days(
    sim_names: list[str], storage_client, bucket_name: str
) -> list[tuple[str, str]]:
    """Retrieve all simulation days from simulation names.

    Returns: sim_name, date
    """
    all_days = set()
    bucket = storage_client.bucket(bucket_name)

    for sim_name in sim_names:
        # List blobs under the simulation folder
        blobs = bucket.list_blobs(prefix=f"{sim_name}/")
        
        num_blobs = 0
        for blob in blobs:
            num_blobs += 1
            # Extract the date from the filename (e.g., "2000-05-24.npy")

            filename = blob.name.split("/")[-1]
            all_days.add((sim_name, get_date(filename)))
        
        assert num_blobs > 0

    return sorted(all_days)


def load_dataset(
    data_bucket_name: str,
    label_bucket_name: str,
    sim_names: list[str],
    storage_client: storage.Client = None,
    shuffle: bool = True,
    hash_range=(0.0, 0.5),
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    storage_client = storage_client or storage.Client()

    # Early validation
    assert storage_client.bucket(
        label_bucket_name
    ).exists(), f"Bucket does not exist: {label_bucket_name}"
    assert storage_client.bucket(
        data_bucket_name
    ).exists(), f"Bucket does not exist: {data_bucket_name}"

    sim_name_dates = get_all_simulation_days(
        sim_names=sim_names, storage_client=storage_client, bucket_name=data_bucket_name
    )

    if shuffle:
        random.shuffle(sim_name_dates)

    logging.info("sim_name_dates %s" % str(sim_name_dates))
    # for i, (sim_name, day) in enumerate(sim_name_dates):

    #     print("all_days:", sim_name_dates)

    def data_generator():
        missing_files: int = 0
        for sim_name, day in sim_name_dates:
            # Skip days not in this dataset
            if hash_range[0] <= hash_day(sim_name, day) < hash_range[1]:
                continue

            # Load spatial, LU index, spatiotemporal, and label from their folders
            lu_index_data = load_pattern_from_cloud(
                data_bucket_name,
                f"{sim_name}/lu_index",
                storage_client,
                max_blobs=1,
            )[0]
            spatial_data = load_pattern_from_cloud(
                data_bucket_name, f"{sim_name}/spatial", storage_client, max_blobs=1
            )[0]

            load_result = load_day(
                day,
                sim_name,
                bucket_name_inputs=data_bucket_name,
                bucket_name_labels=label_bucket_name,
                storage_client=storage_client,
            )
            if load_result is None:
                missing_files += 1
                logging.warning(
                    "incomplete data!: %s %s #%d" % (day, sim_name, missing_files)
                )
                continue

            spatiotemporal_data, label_data = load_result

            yield {
                "spatiotemporal": spatiotemporal_data,
                "spatial": spatial_data.numpy(),
                "lu_index": lu_index_data.numpy().reshape(200, 200),
            }, label_data

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(),
        output_signature=(
            {
                "spatiotemporal": tf.TensorSpec(
                    shape=(
                        constants.INPUT_TIME_STEPS,
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
                    dtype=tf.int32,
                ),
            },
            tf.TensorSpec(
                shape=(
                    constants.OUTPUT_TIME_STEPS,
                    constants.MAP_HEIGHT,
                    constants.MAP_WIDTH,
                    constants.OUTPUT_CHANNELS,
                ),
                dtype=tf.float32,
            ),
        ),
    )
    return dataset


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


def extract_dates(
    bucket_name: str, path: str, storage_client: storage.Client
) -> list[str]:
    """Extract unique dates from filenames in the given bucket path."""
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=path)

    dates = set()
    for blob in blobs:
        match = re.search(r"\d{4}-\d{2}-\d{2}", blob.name)  # Match YYYY-MM-DD format
        if match:
            dates.add(match.group())
    return sorted(dates)


def load_day(
    day: str,  # 2000-05-24
    sim_name: str,
    bucket_name_inputs: str,
    bucket_name_labels: str,
    storage_client: storage.Client,
) -> tuple[tf.Tensor, tf.Tensor] | None:
    # TODO: mock this out to return random tensor to get the count.

    logging.info("load_day (%s, %s)" % (day, sim_name))
    """Load spatiotemporal data and labels for a given day from GCP."""
    # Extract all available dates dynamically
    base_path = f"{sim_name}/spatiotemporal/"
    all_dates = extract_dates(bucket_name_inputs, base_path, storage_client)

    # Ensure the requested day exists in the list of dates
    if day not in all_dates:
        raise ValueError(f"Day {day} is not found in the dataset!")

    # Determine previous, current, and next days
    day_index = all_dates.index(day)
    previous_day = all_dates[day_index - 1] if day_index > 0 else day
    next_day = all_dates[day_index + 1] if day_index < len(all_dates) - 1 else day

    # Load inputs for previous, current, and next days
    def load_inputs_for_date(date: str, max_blobs: int | None = None):
        input_path = f"{sim_name}/spatiotemporal/met_em.d03.{date}_"
        return load_pattern_from_cloud(
            bucket_name_inputs, input_path, storage_client, max_blobs
        )

    inputs_previous = load_inputs_for_date(previous_day, max_blobs=1)
    inputs_current = load_inputs_for_date(day)
    inputs_next = load_inputs_for_date(next_day, max_blobs=1)

    # Concatenate inputs to form a single tensor
    inputs = tf.concat([inputs_previous, inputs_current, inputs_next], axis=0)

    # Load labels for the day
    label_path = f"{sim_name}/wrfout_d03_{day}_"
    labels = load_pattern_from_cloud(bucket_name_labels, label_path, storage_client)
    num_label_timestamps = labels.shape[0]
    if num_label_timestamps == constants.OUTPUT_TIME_STEPS:
        return inputs, labels

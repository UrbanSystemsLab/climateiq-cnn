import logging
import random
from typing import Iterable
import functools
import itertools
from datetime import datetime, timedelta
import urllib.parse
import hashlib  # For hashing days
import tensorflow as tf
from google.cloud import storage  # type: ignore
from usl_models.atmo_ml import constants
from usl_models.shared import downloader
import re  # To extract dates from filenames

DATE_FORMAT = "%Y-%m-%d"
FEATURE_FILENAME_FORMAT = "met_em.d03.%Y-%m-%d_%H:%M:%S.npy"
LABEL_FILENAME_FORMAT = "wrfout_d03_%Y-%m-%d_%H:%M:%S.npy"


# Load data from Google Cloud Storage
@functools.lru_cache(maxsize=512)
def load_pattern_from_cloud(
    bucket_name: str,
    prefix: str,
    storage_client: storage.Client,
    max_blobs: int | None = None,
    blob_offset: int | None = None,
) -> tf.Tensor:
    """Download all files in the folder.

    If max_blobs is specified, only returns that many blobs.
    """
    logging.warning("prefix: %s", prefix)
    bucket = storage_client.bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix=prefix, max_results=max_blobs))
    if blob_offset:
        blobs = blobs[blob_offset:]

    all_data = []
    for blob in blobs:
        logging.warning("  blob.path: %s", urllib.parse.unquote_plus(blob.path))

        if blob.name.endswith(".npy"):  # Ensure only .npy files are processed
            all_data.append(downloader.blob_to_tensor(blob))
        else:
            logging.error("  Unexpected file extension: %s", blob.name)

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

    Returns: [(sim_name, date), ...]
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
    hash_range=(0.0, 1.0),
    dates: list[str] | None = None,
) -> tf.data.Dataset:
    storage_client = storage_client or storage.Client()

    # Early validation
    assert storage_client.bucket(
        data_bucket_name
    ).exists(), f"Bucket does not exist: {data_bucket_name}"
    assert storage_client.bucket(
        label_bucket_name
    ).exists(), f"Bucket does not exist: {label_bucket_name}"

    if dates is not None:
        sim_name_dates = list(itertools.product(sim_names, dates))
    else:
        sim_name_dates = get_all_simulation_days(
            sim_names=sim_names,
            storage_client=storage_client,
            bucket_name=data_bucket_name,
        )
    print("sim_name_dates", sim_name_dates)

    if shuffle:
        random.shuffle(sim_name_dates)

    logging.info("Total simulation days before filtering: %d", len(sim_name_dates))

    # Track stats for filtering
    total_days = len(sim_name_dates)
    selected_days = [
        (sim_name, day)
        for sim_name, day in sim_name_dates
        if hash_range[0] <= hash_day(sim_name, day) < hash_range[1]
    ]
    selected_percentage = len(selected_days) / total_days * 100
    logging.info(
        "Selected %d/%d days (%.2f%%) based on hash range %s.",
        len(selected_days),
        total_days,
        selected_percentage,
        hash_range,
    )

    def data_generator() -> Iterable[tuple[dict[str, tf.Tensor], tf.Tensor]]:
        missing_days: int = 0
        generated_count: int = 0

        feature_bucket = storage_client.bucket(data_bucket_name)
        label_bucket = storage_client.bucket(label_bucket_name)
        for sim_name, day in sim_name_dates:
            # Skip days not in this dataset
            if not (hash_range[0] <= hash_day(sim_name, day) < hash_range[1]):
                continue

            load_result = load_day(
                sim_name,
                datetime.strptime(day, DATE_FORMAT),
                feature_bucket=feature_bucket,
                label_bucket=label_bucket,
            )
            if load_result is None:
                missing_days += 1
                continue

            generated_count += 1
            yield load_result

        logging.info("Total generated samples: %d", generated_count)
        if missing_days > 0:
            logging.warning("Total days with missing data: %d", missing_days)

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


def extract_dates(
    bucket: storage.Bucket, path: str, storage_client: storage.Client
) -> list[str]:
    """Extract unique dates from filenames in the given bucket path."""
    blobs = bucket.list_blobs(prefix=path)

    dates = set()
    for blob in blobs:
        match = re.search(r"\d{4}-\d{2}-\d{2}", blob.name)  # Match YYYY-MM-DD format
        if match:
            dates.add(match.group())
    return sorted(dates)


def load_day(
    sim_name: str,
    date: datetime,
    feature_bucket: storage.Bucket,
    label_bucket: storage.Bucket,
) -> tuple[dict[str, tf.Tensor], tf.Tensor] | None:
    """Loads a single example from (sim_name, date)."""

    logging.info("load_day('%s', '%s')" % (sim_name, date.strftime(DATE_FORMAT)))
    start_filename = date.strftime(FEATURE_FILENAME_FORMAT)

    lu_index_data = downloader.try_download_tensor(
        feature_bucket, f"{sim_name}/lu_index/{start_filename}"
    )
    if lu_index_data is None:
        return None

    spatial_data = downloader.try_download_tensor(
        feature_bucket,
        f"{sim_name}/spatial/{start_filename}",
    )
    if spatial_data is None:
        return None

    load_result = load_day_temporal(
        sim_name,
        date,
        feature_bucket,
        label_bucket,
    )
    if load_result is None:
        return None

    spatiotemporal_data, label_data = load_result
    logging.warning("label shape: %s", str(label_data.shape))

    return {
        "spatiotemporal": spatiotemporal_data,
        "spatial": spatial_data,
        "lu_index": tf.reshape(
            lu_index_data, shape=(constants.MAP_HEIGHT, constants.MAP_WIDTH)
        ),
    }, label_data


def load_day_temporal(
    sim_name: str,
    date: datetime,
    feature_bucket: storage.Bucket,
    label_bucket: storage.Bucket,
) -> tuple[tf.Tensor, tf.Tensor] | None:
    """Load spatiotemporal data and labels for a given day from GCP."""

    # Load spatiotemporal tensors.
    spatiotemporal_path = f"{sim_name}/spatiotemporal/"
    timestep_interval = timedelta(hours=6)
    timestamps = [date + timestep_interval * i for i in range(-1, 5)]
    spatiotemporal_tensors = [
        downloader.try_download_tensor(
            feature_bucket, ts.strftime(spatiotemporal_path + FEATURE_FILENAME_FORMAT)
        )
        for ts in timestamps
    ]
    if None in spatiotemporal_tensors:
        logging.warning(
            "Missing feature timestamp(s) for date %s",
            date.strftime(sim_name + "/" + DATE_FORMAT),
        )
        return None

    label_path = f"{sim_name}/"
    label_timestep_interval = timedelta(hours=3)
    label_timestamps = [date + label_timestep_interval * i for i in range(8)]
    label_tensors = [
        downloader.try_download_tensor(
            label_bucket, ts.strftime(label_path + LABEL_FILENAME_FORMAT)
        )
        for ts in label_timestamps
    ]
    if None in label_tensors:
        logging.warning(
            "Missing label timestamp(s) for date %s",
            date.strftime(sim_name + "/" + DATE_FORMAT),
        )
        return None

    spatiotemporal_tensor = tf.concat([spatiotemporal_tensors], axis=0)
    label_tensor = tf.stack(label_tensors)
    return spatiotemporal_tensor, label_tensor

import logging
import random
from typing import Iterable
import pathlib
import itertools
from datetime import datetime, timedelta
import hashlib  # For hashing days
import tensorflow as tf
import numpy as np
from google.cloud import storage  # type: ignore
from google.cloud.storage import transfer_manager  # type: ignore
from usl_models.atmo_ml import constants, vars
from usl_models.shared import downloader


FEATURE_BUCKET_NAME = "climateiq-study-area-feature-chunks"
LABEL_BUCKET_NAME = "climateiq-study-area-label-chunks"
DATE_FORMAT = "%Y-%m-%d"
FEATURE_FILENAME_FORMAT = "met_em.d03.%Y-%m-%d_%H:%M:%S.npy"
FEATURE_FILENAME_FORMAT_NPZ = "met_em.d03.%Y-%m-%d_%H:%M:%S.npz"
STATIC_FILENAME_NPZ = "static.npz"
LABEL_FILENAME_FORMAT = "wrfout_d03_%Y-%m-%d_%H:%M:%S.npy"
LABEL_FILENAME_FORMAT_NPZ = "wrfout_d03_%Y-%m-%d_%H:%M:%S.npz"


# Key for each example in the dataset.
# (sim_name, date)
ExampleKey = tuple[str, str]


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


def get_cached_sim_dates(path: pathlib.Path) -> list[tuple[str, str]]:
    """Return all cached simulation dates."""
    all_dates = set()
    for file in path.glob("**/labels/wrfout_d03_????-??-??_??:??:??.npz"):
        relative_path = file.relative_to(path)
        sim_name = str(relative_path.parent.parent)
        ts = datetime.strptime(relative_path.name, LABEL_FILENAME_FORMAT_NPZ)
        all_dates.add((sim_name, ts.date().strftime(DATE_FORMAT)))

    return sorted(all_dates)


def make_dataset(generator, output_channels: int) -> tf.data.Dataset:
    return tf.data.Dataset.from_generator(
        lambda: generator(),
        output_signature=(
            constants.INPUT_SPEC,
            tf.TensorSpec(
                shape=(
                    constants.OUTPUT_TIME_STEPS,
                    constants.MAP_HEIGHT,
                    constants.MAP_WIDTH,
                    output_channels,
                ),
                dtype=tf.float32,
            ),
        ),
    )


def load_dataset_cached(
    filecache_dir: pathlib.Path,
    hash_range=(0.0, 1.0),
    example_keys: list[ExampleKey] | None = None,
    output_vars: list[vars.SpatiotemporalOutput] | None = None,
    shuffle: bool = True,
):
    """Loads a dataset from a filecache."""
    example_keys = example_keys or get_cached_sim_dates(filecache_dir)
    output_vars = output_vars or list(vars.SpatiotemporalOutput)
    output_mask = tf.constant([var in output_vars for var in vars.SpatiotemporalOutput])

    if shuffle:
        random.shuffle(example_keys)

    def generator() -> Iterable[tuple[dict[str, tf.Tensor], tf.Tensor]]:
        missing_days: int = 0
        generated_count: int = 0
        for sim_name, day in example_keys:
            # Skip days not in this dataset
            if not (hash_range[0] <= hash_day(sim_name, day) < hash_range[1]):
                continue

            load_result = load_day_cached(
                filecache_dir / sim_name,
                datetime.strptime(day, DATE_FORMAT),
            )
            if load_result is None:
                missing_days += 1
                continue

            generated_count += 1
            inputs, labels = load_result
            yield inputs, tf.boolean_mask(labels, output_mask, axis=3)

        logging.info("Total generated samples: %d", generated_count)
        if missing_days > 0:
            logging.warning("Total days with missing data: %d", missing_days)

    return make_dataset(generator, len(output_vars))


def load_dataset(
    data_bucket_name: str,
    label_bucket_name: str,
    sim_names: list[str],
    storage_client: storage.Client = None,
    shuffle: bool = True,
    hash_range=(0.0, 1.0),
    dates: list[str] | None = None,
    output_vars: list[vars.SpatiotemporalOutput] | None = None,
) -> tf.data.Dataset:
    storage_client = storage_client or storage.Client()
    output_vars = output_vars or list(vars.SpatiotemporalOutput)
    output_mask = [var in output_vars for var in vars.SpatiotemporalOutput]

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

    def generator() -> Iterable[tuple[dict[str, tf.Tensor], tf.Tensor]]:
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
            inputs, labels = load_result
            yield inputs, tf.boolean_mask(labels, output_mask, axis=3)

        logging.info("Total generated samples: %d", generated_count)
        if missing_days > 0:
            logging.warning("Total days with missing data: %d", missing_days)

    return make_dataset(generator, output_channels=len(output_vars))


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

    spatiotemporal_data = load_day_spatiotemporal(sim_name, date, feature_bucket)
    if spatiotemporal_data is None:
        return None

    label_data = load_day_label(sim_name, date, label_bucket)
    if label_data is None:
        return None

    return {
        "spatiotemporal": spatiotemporal_data,
        "spatial": spatial_data,
        "lu_index": tf.reshape(
            lu_index_data, shape=(constants.MAP_HEIGHT, constants.MAP_WIDTH)
        ),
    }, label_data


def load_day_spatiotemporal(
    sim_name: str, date: datetime, bucket: storage.Bucket
) -> tf.Tensor | None:
    """Load spatiotemporal tensors for a day."""
    spatiotemporal_path = f"{sim_name}/spatiotemporal/"
    timestep_interval = timedelta(hours=6)
    timestamps = [date + timestep_interval * i for i in range(-1, 5)]
    spatiotemporal_tensors = [
        downloader.try_download_tensor(
            bucket, ts.strftime(spatiotemporal_path + FEATURE_FILENAME_FORMAT)
        )
        for ts in timestamps
    ]
    if None in spatiotemporal_tensors:
        logging.warning(
            "Missing feature timestamp(s) for date %s",
            date.strftime(spatiotemporal_path + DATE_FORMAT),
        )
        return None
    return tf.concat([spatiotemporal_tensors], axis=0)


def load_day_label(
    sim_name: str,
    date: datetime,
    bucket: storage.Bucket,
) -> tf.Tensor | None:
    """Load label tensor for a day."""
    label_path = f"{sim_name}/"
    label_timestep_interval = timedelta(hours=3)
    label_timestamps = [date + label_timestep_interval * i for i in range(8)]
    label_tensors = [
        downloader.try_download_tensor(
            bucket, ts.strftime(label_path + LABEL_FILENAME_FORMAT)
        )
        for ts in label_timestamps
    ]
    if None in label_tensors:
        logging.warning(
            "Missing label timestamp(s) for date %s",
            date.strftime(label_path + DATE_FORMAT),
        )
        return None

    return tf.stack(label_tensors)


def load_day_cached(
    path: pathlib.Path, date: datetime
) -> tuple[dict[str, tf.Tensor], tf.Tensor] | None:
    spatiotemporal = load_day_spatiotemporal_cached(path / "spatiotemporal", date)
    if spatiotemporal is None:
        return None
    labels = load_day_label_cached(path / "labels", date)
    if labels is None:
        return None
    static_data = np.load(path / STATIC_FILENAME_NPZ)
    if static_data is None:
        return None
    return (
        dict(
            spatiotemporal=spatiotemporal,
            spatial=tf.convert_to_tensor(static_data["spatial"]),
            lu_index=tf.convert_to_tensor(
                static_data["lu_index"].reshape(
                    (constants.MAP_HEIGHT, constants.MAP_WIDTH)
                )
            ),
        ),
        labels,
    )


def try_load_npz(path: pathlib.Path) -> dict[str, np.ndarray] | None:
    """Tries to load an npz file."""
    if not path.exists():
        logging.warning("Missing path: %s", path)
        return None

    npz = np.load(path)
    if "arr_0" not in npz:
        logging.warning(
            "Missing array 'arr_0' for path %s",
            path,
        )
        return None

    return npz


def load_day_spatiotemporal_cached(
    path: pathlib.Path, date: datetime
) -> tf.Tensor | None:
    """Load spatiotemporal tensors for a day."""
    timestep_interval = timedelta(hours=6)
    timestamps = [date + timestep_interval * i for i in range(-1, 5)]

    arrays = []
    for ts in timestamps:
        npz = try_load_npz(path / ts.strftime(FEATURE_FILENAME_FORMAT_NPZ))
        if npz is None:
            return None

        arrays.append(npz["arr_0"])

    return tf.stack(arrays)


def load_day_label_cached(path: pathlib.Path, date: datetime) -> tf.Tensor | None:
    """Load label tensor for a day."""
    timestep_interval = timedelta(hours=3)
    timestamps = [date + timestep_interval * i for i in range(8)]

    arrays = []
    for ts in timestamps:
        npz = try_load_npz(path / ts.strftime(LABEL_FILENAME_FORMAT_NPZ))
        if npz is None:
            return None

        arrays.append(npz["arr_0"])

    return tf.stack(arrays)


def download_simulation(
    feature_bucket: storage.Bucket,
    label_bucket: storage.Bucket,
    sim_path: pathlib.Path,
    output_path: pathlib.Path,
    worker_type=transfer_manager.PROCESS,
):
    """Downloads a simulation to the output_path.

    For the format looks like:
    ```
    NYC_summer_2017_25p/
        ├── spatiotemporal/
        │   ├── met_em.d03.2017-05-24_00:00:00.npz
        │   ├── met_em.d03.2017-05-24_00:06:00.npz
        │   └── ...
        ├── labels/
        │   ├── met_em.d03.2017-05-24_00:00:00.npz
        │   ├── met_em.d03.2017-05-24_00:03:00.npz
        │   ├── met_em.d03.2017-05-24_00:06:00.npz
        │   └── ...
        ├── met_em.d03.2017-05-24_06:00:00.npz
        └── static.npz
    ```
    """
    output_path.mkdir(parents=True, exist_ok=True)

    # Download static files (time invariant)
    [(_, spatial)] = list(
        downloader.bulk_download_numpy(
            feature_bucket, sim_path / "spatial", worker_type=worker_type, max_files=1
        )
    )
    [(_, lu_index)] = list(
        downloader.bulk_download_numpy(
            feature_bucket, sim_path / "lu_index", worker_type=worker_type, max_files=1
        )
    )
    np.savez_compressed(output_path / "static", spatial=spatial, lu_index=lu_index)

    spatiotemporal_path = output_path / "spatiotemporal"
    spatiotemporal_path.mkdir(parents=True, exist_ok=True)
    for filename, array in downloader.bulk_download_numpy(
        feature_bucket, sim_path / "spatiotemporal", worker_type=worker_type
    ):
        np.savez_compressed(spatiotemporal_path / pathlib.Path(filename).stem, array)

    labels_path = output_path / "labels"
    labels_path.mkdir(parents=True, exist_ok=True)
    for filename, array in downloader.bulk_download_numpy(
        label_bucket, sim_path, worker_type=worker_type
    ):
        np.savez_compressed(labels_path / pathlib.Path(filename).stem, array)


def download_dataset(
    sim_names: list[str],
    output_path: pathlib.Path,
    client: storage.Client | None = None,
    feature_bucket_name: str = FEATURE_BUCKET_NAME,
    label_bucket_name: str = LABEL_BUCKET_NAME,
):
    """Download a dataset from GCS to the given local path."""
    client = client or storage.Client()
    output_path.mkdir(parents=True, exist_ok=True)
    feature_bucket = client.bucket(feature_bucket_name)
    label_bucket = client.bucket(label_bucket_name)
    for sim_name in sim_names:
        logging.info('Download simulation "%s"', sim_name)
        sim_path = pathlib.Path(sim_name)
        download_simulation(
            sim_path=sim_path,
            output_path=output_path / sim_path,
            feature_bucket=feature_bucket,
            label_bucket=label_bucket,
        )

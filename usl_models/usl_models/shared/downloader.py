import urllib.parse
import numpy as np
import numpy.typing as npt
from google.cloud import storage  # type:ignore[attr-defined]
import tensorflow as tf


def download_as_array(client: storage.Client, gcs_url: str) -> npt.NDArray:
    """Retrieves the contents at `gcs_url` from GCS as a numpy array."""
    parsed = urllib.parse.urlparse(gcs_url)
    bucket = client.bucket(parsed.netloc)
    blob = bucket.blob(parsed.path.lstrip("/"))

    with blob.open("rb") as fd:
        return np.load(fd)


def download_as_tensor(client: storage.Client, gcs_url: str) -> tf.Tensor:
    """Retrieves the contents at `gcs_url` from GCS as a tf tensor."""
    return tf.convert_to_tensor(
        download_as_array(client, gcs_url),
        dtype=tf.float32,
    )


def blob_to_array(blob: storage.Blob) -> npt.NDArray:
    with blob.open("rb") as fd:
        return np.load(fd)


def blob_to_tensor(blob: storage.Blob):
    return tf.convert_to_tensor(
        blob_to_array(blob),
        dtype=tf.float32,
    )

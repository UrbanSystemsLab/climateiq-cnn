import numpy as np

from google.cloud import storage


def upload_npy(array: np.ndarray, bucket: storage.Bucket, path: str):
    """Upload a NPY array to GCS."""
    blob = bucket.blob(path)
    with blob.open("wb") as fd:
        np.save(fd, array)

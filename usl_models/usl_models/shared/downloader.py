import io
import urllib.parse
import pathlib
from typing import Iterable

import numpy as np
import logging
import numpy.typing as npt
import tensorflow as tf

from google.cloud import storage  # type:ignore[attr-defined]
from google.cloud.storage import transfer_manager


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


def _rmdir_recursive(directory: pathlib.Path):
    """Recursively remove a directory. Assumes all inner paths are empty directories."""
    directory = pathlib.Path(directory)
    for item in directory.iterdir():
        _rmdir_recursive(item)
        if item.is_dir():
            item.rmdir()
        else:
            item.unlink()


def bulk_download(
    bucket: storage.Bucket,
    path: pathlib.Path,
    tmp_path: pathlib.Path = pathlib.Path("/tmp/filecache"),
    workers: int = 8,
    worker_type=transfer_manager.PROCESS,
    max_files: int = 100000,
) -> Iterable[tuple[str, io.BufferedIOBase]]:
    """Download all blobs in the bucket under the given path."""
    blobs: list[storage.Blob] = list(bucket.list_blobs(prefix=path))
    if not blobs:
        raise ValueError(
            f"Blob path '{path}' contained no files in bucket '{bucket.name}'"
        )
    if worker_type == transfer_manager.THREAD:
        # If using threads, can download files directly into memory.
        blob_bytesio = [(blob, io.BytesIO()) for blob in blobs[:max_files]]
        transfer_manager.download_many(
            blob_bytesio,
            max_workers=workers,
            worker_type=worker_type,
            raise_exception=True,
        )
        for blob, bytesio in blob_bytesio:
            yield blob.name, bytesio
    else:
        # If using processes, must download files to temporary directory
        # and clean up afterwards.
        tmp_path.mkdir(parents=True, exist_ok=True)
        blob_filenames: list[tuple[storage.Blob, str]] = [
            (blob, str(tmp_path / blob.name)) for blob in blobs[:max_files]
        ]
        for _, filename in blob_filenames:
            pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)
        transfer_manager.download_many(
            blob_filenames,
            max_workers=workers,
            worker_type=worker_type,
            raise_exception=True,
        )
        for blob, filename in blob_filenames:
            with open(filename, "rb") as fd:
                yield blob.name, fd
            pathlib.Path(filename).unlink(missing_ok=False)  # type: ignore
        _rmdir_recursive(tmp_path)


def bulk_download_numpy(
    bucket: storage.Bucket,
    path: pathlib.Path,
    tmp_path: pathlib.Path = pathlib.Path("/tmp/filecache"),
    workers: int = 8,
    worker_type=transfer_manager.PROCESS,
    max_files: int = 100000,
) -> Iterable[tuple[str, np.ndarray]]:
    """Bulk download a path in a bucket."""
    for filename, file in bulk_download(
        bucket=bucket,
        path=path,
        tmp_path=tmp_path,
        workers=workers,
        worker_type=worker_type,
        max_files=max_files,
    ):
        yield filename, np.load(file)


def try_download_tensor(bucket: storage.Bucket, path: str) -> tf.Tensor | None:
    blob = bucket.blob(path)
    if not blob.exists():
        logging.warning("blob does not exist: %s", path)
        return None
    return blob_to_tensor(blob)

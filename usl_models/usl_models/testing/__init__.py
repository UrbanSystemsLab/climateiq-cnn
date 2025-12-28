import builtins
import io
import unittest
from unittest import mock
import urllib.parse
from typing import Any, Self
import pathlib

import numpy as np
import tensorflow as tf

from google.cloud import storage  # type: ignore


class TestCase(unittest.TestCase):
    """Testing utilss."""

    def assertShape(
        self, obj: np.ndarray | tf.Tensor, expected: np.ndarray | tf.Tensor
    ):
        """Checks the shape of two tensor like objects."""
        self.assertEqual(obj.shape, expected, msg="Shape mismatch.")

    def assertShapesRecursive(self, obj: Any, expected: Any, path: str = ""):
        """Recursively checks the shapes of numpy arrays in a data structure."""
        if isinstance(obj, np.ndarray) or tf.is_tensor(obj):
            self.assertEqual(obj.shape, expected, msg=f"Shape mismatch in path {path}")
            return

        self.assertEqual(type(obj), type(expected), msg=f"Type mismatch in path {path}")

        match type(obj):
            case builtins.list:
                self.assertEqual(len(obj), len(expected))
                for i, (o, e) in enumerate(zip(obj, expected)):
                    self.assertShapesRecursive(o, e, path + f"[{i}]")
            case builtins.tuple:
                for i, (o, e) in enumerate(zip(obj, expected)):
                    self.assertShapesRecursive(o, e, path + f"[{i}]")
            case builtins.dict:
                assert set(obj.keys()) == set(expected.keys())
                for k in obj:
                    self.assertShapesRecursive(obj[k], expected[k], path + f"['{k}']")


def wrap_mock(spec):
    """Wraps a class in `mock.MagicMock`."""

    def wrapper(cls):
        """Wraps a spec class as a mock."""

        def __init__(self, *args, **kwargs):
            mock.MagicMock.__init__(self, spec=spec, *args, **kwargs)

        cls.__init__ = __init__
        return cls

    return wrapper


@wrap_mock(storage.Blob)
class MockBlob(mock.MagicMock):
    """Mock `storage.Blob`."""

    def _buf_open(self, _mode: str) -> io.BytesIO:
        """Returns a copy of the blob's buffer."""
        buf = io.BytesIO(self._buf.read())
        self._buf.seek(0)
        return buf

    def with_buf(self, buf: io.BytesIO) -> Self:
        """Sets the buffer of this blob."""
        self._buf = buf
        self.open.side_effect = self._buf_open
        return self

    def with_npy(self, arr: np.ndarray, allow_pickle=True) -> Self:
        """Sets the buffer to a serialized np array."""
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=allow_pickle)
        buf.seek(0)
        self.with_buf(buf)
        return self

    def with_path(self, path: str) -> Self:
        """Sets the path and name of a blob."""
        parsed_path: pathlib.Path = pathlib.Path(path)
        self.bucket.name = parsed_path.parents[-2]
        self.path = urllib.parse.quote_plus(path)
        self.name = parsed_path.name
        return self

    def with_exists(self, exists: bool) -> Self:
        """Sets existance bool."""
        self.exists.return_value = exists
        return self


@wrap_mock(storage.Bucket)
class MockBucket(mock.MagicMock):
    """Mock `storage.Bucket`."""

    def with_blobs(self, blobs: dict[str, MockBlob]) -> Self:
        """Construct mock bucket with dictionary of blobs."""

        def _list_blobs(prefix: str, max_results: int | None = None):
            results = [b for p, b in blobs.items() if p.startswith(prefix)][
                :max_results
            ]
            return results

        def _blob(path: str):
            if path in blobs:
                return blobs[path]
            else:
                return MockBlob().with_exists(False)

        self.list_blobs.side_effect = _list_blobs
        self.blob.side_effect = _blob
        self.exists.return_value = True
        return self


@wrap_mock(storage.Client)
class MockStorageClient(mock.MagicMock):
    """Mock `storage.Client`."""

    def with_buckets(self, buckets: dict[str, MockBucket]) -> Self:
        """Adds mock buckets to the storage client."""
        self.bucket.side_effect = lambda bucket_name: buckets[bucket_name]
        return self

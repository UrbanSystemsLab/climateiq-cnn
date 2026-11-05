"""Chunk paths, chunk errors and writing arrays to GCS."""

import io
import pathlib
import re
from typing import Tuple
from google.cloud import firestore
from google.cloud import storage
import numpy
from numpy.typing import NDArray
from usl_lib.storage import metastore


def _write_chunk_metastore_error(chunk_path: str, error_message: str) -> None:
    """Updates the metastore with an error message for this chunk.

    Args:
      chunk_path: The GCS path to the chunk being processed.
      error_message: The error message to write into Firestore.
    """
    db = firestore.Client()
    file_path = pathlib.PurePosixPath(chunk_path)
    study_area_name = file_path.parts[0]
    doc_id = file_path.stem.replace(".", "_").replace(":", "_")
    metastore.StudyAreaChunk(id_=doc_id, error=error_message).merge(db, study_area_name)


def _parse_chunk_path(chunk_path: str) -> Tuple[str, str]:
    """Parses a GCS chunk path into (study_area, chunk) components.

    Args:
      chunk_path: The GCS path of a chunk to parse.

    Returns:
      Given a GCS path in the form "study_area_name/chunk_name" returns a tuple
      (study_area_name, chunk_name)
    """
    as_path = pathlib.PurePosixPath(chunk_path)
    return str(as_path.parent), as_path.stem


def _parse_spatial_chunk_indices_from_name(chunk_name: str) -> Tuple[int, int]:
    """Extract the x, y indices from a name like "chunk_x_y.tar"."""
    indices = re.search(r"(\d+)_(\d+)", chunk_name)
    if indices is None:
        raise ValueError(
            f"Unexpected chunk name {chunk_name} does not contain chunk indices"
        )
    return int(indices.group(1)), int(indices.group(2))


def _write_as_npy(blob: storage.Blob, array: NDArray) -> None:
    """Writes `array` into the GCS `blob` in the numpy binary array format.

    Args:
      blob: The blob to write the array into.
      array: The array to upload.
    """
    npy_file = io.BytesIO()
    numpy.save(npy_file, array)
    # Seek to the beginning so the file can be read.
    npy_file.flush()
    npy_file.seek(0)
    blob.upload_from_file(npy_file)

from typing import Iterable, Tuple

import numpy
from numpy.typing import NDArray


def split_raster_into_chunks(
    chunk_size: int, raster: NDArray
) -> Iterable[Tuple[int, int, NDArray]]:
    """Chunks the given array into chunk_size squares along the 0 and 1 axis.

    All chunks will be of shape (chunk_size, chunk_size)
    For chunks which lie at the bounds of the array, fill the areas which would exceed
    the array bounds with zeros.

    Args:
        chunk_size: The size squared of each chunk.
        raster: The numpy array to break into chunks.

    Yields:
        Tuples of (x_chunk_index, y_chunk_index, chunk) where `x_chunk_index` &
        `y_chunk_index` describe the index of the chunk relative to other chunks and
        `chunk` is the array for the chunk itself.
    """
    row_count = raster.shape[0]
    col_count = raster.shape[1]

    for y_chunk_index, y_start in enumerate(range(0, row_count, chunk_size)):
        for x_chunk_index, x_start in enumerate(range(0, col_count, chunk_size)):
            x_end = min(x_start + chunk_size, col_count)
            y_end = min(y_start + chunk_size, row_count)
            chunk = raster[y_start:y_end, x_start:x_end]

            y_pad = chunk_size - chunk.shape[0]
            x_pad = chunk_size - chunk.shape[1]
            if y_pad > 0 or x_pad > 0:
                chunk = numpy.pad(chunk, numpy.array(((0, y_pad), (0, x_pad))))

            yield x_chunk_index, y_chunk_index, chunk

from osgeo import gdal
import typing

from usl_lib.readers import elevation_readers
from usl_lib.shared import suppliers


def split_geotiff_into_chunks(
    elevation_file_path: str,
    chunk_size: int,
    chunk_file_path_generator: suppliers.ChunkFilePathGenerator,
) -> typing.Tuple[int, int]:
    """Produces a grid of chunk GeoTIFF files based on input GeoTIFF file.

    Args:
        elevation_file_path: GeoTIFF file with elevation data to be split into chunks.
        chunk_size: Size of a chunk in cells (along both X and Y axes).
        chunk_file_path_generator: Caller provided source to generate chunk file names.

    Returns:
        Tuple (number of chunks along the Y-axis, number of chunks along the X-axis).
    """
    with open(elevation_file_path, "rb") as input_file:
        elevation = elevation_readers.read_from_geotiff(input_file, header_only=True)

    global_col_count = elevation.header.col_count
    global_row_count = elevation.header.row_count
    x_chunk_count = int((global_col_count + chunk_size - 1) / chunk_size)
    y_chunk_count = int((global_row_count + chunk_size - 1) / chunk_size)

    ds = gdal.Open(elevation_file_path)
    for y_chunk_index in range(y_chunk_count):
        for x_chunk_index in range(x_chunk_count):
            chunk_file_path = chunk_file_path_generator.generate(
                y_chunk_index, x_chunk_index
            )
            # Let's extract one chunk for a given positional indices.
            row_start = y_chunk_index * chunk_size
            row_count = min(global_row_count - row_start, chunk_size)
            col_start = x_chunk_index * chunk_size
            col_count = min(global_col_count - col_start, chunk_size)
            gdal.Translate(
                chunk_file_path, ds, srcWin=[col_start, row_start, col_count, row_count]
            )

    return y_chunk_count, x_chunk_count

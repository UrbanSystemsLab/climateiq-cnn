import dataclasses
import pathlib

from osgeo import gdal

from usl_lib.readers import elevation_readers


@dataclasses.dataclass
class ChunkDescriptor:
    """Information about chunk file."""

    y_chunk_index: int
    x_chunk_index: int
    path: pathlib.Path


def split_geotiff_into_chunks(
    elevation_file_path: str | pathlib.Path,
    chunk_size: int,
    output_dir_path: str | pathlib.Path,
    chunk_file_name_pattern: str = "chunk_{y}_{x}",
) -> list[ChunkDescriptor]:
    """Produces a grid of chunk GeoTIFF files based on input GeoTIFF file.

    Args:
        elevation_file_path: GeoTIFF file with elevation data to be split into chunks.
        chunk_size: Size of a chunk in cells (along both X and Y axes).
        output_dir_path: Path to the directory where chunk files will be stored.
        chunk_file_name_pattern: Format pattern used to generate chink file names.

    Returns:
        List of descriptors for produced chunk files.
    """
    with pathlib.Path(elevation_file_path).open("rb") as input_file:
        elevation = elevation_readers.read_from_geotiff(input_file, header_only=True)

    global_col_count = elevation.header.col_count
    global_row_count = elevation.header.row_count
    x_chunk_count = int((global_col_count + chunk_size - 1) / chunk_size)
    y_chunk_count = int((global_row_count + chunk_size - 1) / chunk_size)

    ds = gdal.Open(str(elevation_file_path))
    chunk_descriptors: list[ChunkDescriptor] = []
    for y_chunk_index in range(y_chunk_count):
        for x_chunk_index in range(x_chunk_count):
            chunk_file_path = pathlib.Path(
                output_dir_path
            ) / chunk_file_name_pattern.format(y=y_chunk_index, x=x_chunk_index)
            # Let's extract one chunk for a given positional indices.
            row_start = y_chunk_index * chunk_size
            row_count = min(global_row_count - row_start, chunk_size)
            col_start = x_chunk_index * chunk_size
            col_count = min(global_col_count - col_start, chunk_size)
            gdal.Translate(
                str(chunk_file_path),
                ds,
                srcWin=[col_start, row_start, col_count, row_count],
            )
            chunk_descriptors.append(
                ChunkDescriptor(
                    y_chunk_index=y_chunk_index,
                    x_chunk_index=x_chunk_index,
                    path=chunk_file_path,
                )
            )

    return chunk_descriptors

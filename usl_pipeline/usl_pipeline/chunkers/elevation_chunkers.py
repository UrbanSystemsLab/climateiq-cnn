from osgeo import gdal

from usl_pipeline.readers import elevation_readers
from usl_pipeline.shared import suppliers


class GeoTiffChunker:
    """Chunker for GeoTIFF elevation data."""

    def __init__(self, elevation_file_path: str, chunk_size: int):
        """Constructor of GeoTIFF chunker.

        Args:
            elevation_file_path: GeoTIFF file with elevation data to be split into
                chunks.
            chunk_size: Size of a chunk in cells (along both X and Y axes).
        """
        self.chunk_size = chunk_size
        self.elevation_file_path = elevation_file_path

        with open(elevation_file_path, "rb") as input_file:
            elevation = elevation_readers.read_from_geotiff(
                input_file, header_only=True
            )

        self.global_col_count = elevation.header.col_count
        self.global_row_count = elevation.header.row_count
        self.x_chunk_count = int((self.global_col_count + chunk_size - 1) / chunk_size)
        self.y_chunk_count = int((self.global_row_count + chunk_size - 1) / chunk_size)

    def extract_chunk(
        self, y_chunk_index: int, x_chunk_index: int, chunk_file_path: str
    ):
        """Extracts one chunk for a given positional indices.

        Args:
            y_chunk_index: Y-axis index of the chunk.
            x_chunk_index: X-axis index of the chunk.
            chunk_file_path: Output file path to write the chunk to.
        """
        row_start = y_chunk_index * self.chunk_size
        row_count = min(self.global_row_count - row_start, self.chunk_size)
        col_start = x_chunk_index * self.chunk_size
        col_count = min(self.global_col_count - col_start, self.chunk_size)
        ds = gdal.Open(self.elevation_file_path)
        gdal.Translate(
            chunk_file_path, ds, srcWin=[col_start, row_start, col_count, row_count]
        )

    def split_into_chunks(
        self,
        chunk_file_path_generator: suppliers.ChunkFilePathGenerator,
    ) -> (int, int):
        """Produces a grid of chunk GeoTIFF files based on input GeoTIFF file.

        Args:
            chunk_file_path_generator: User provided source to generate chunk file
                names.
        """
        for y_chunk_index in range(self.y_chunk_count):
            for x_chunk_index in range(self.x_chunk_count):
                chunk_file_path = chunk_file_path_generator.generate(
                    y_chunk_index, x_chunk_index
                )
                self.extract_chunk(y_chunk_index, x_chunk_index, chunk_file_path)

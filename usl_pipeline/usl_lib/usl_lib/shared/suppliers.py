import abc
import os


class ChunkFilePathGenerator(abc.ABC):
    """Abstract layer that can be used to generate paths for chunk files."""

    @abc.abstractmethod
    def generate(self, y_chunk_index: int, x_chunk_index: int) -> str:
        """Generates path for a chunk file for given positional indices.

        Args:
            y_chunk_index: Index along the Y-axis for a chunk position.
            x_chunk_index: Index along the X-axis for a chunk position.

        Returns:
            A path to a chunk file.
        """
        pass


class PatternBasedChunkFilePathGenerator(ChunkFilePathGenerator):
    """An implementation of a chunk file path generator based on file name pattern."""

    def __init__(self, dir_path: str, file_name_pattern: str):
        """Constructor.

        Args:
            dir_path: A directory where every chunk file path will point.
            file_name_pattern: A pattern for chunk file name that will be used in
                formatter together with y-axis index of a chunk (parameter {0}) and
                x-axis index of a chunk (parameter {1}).
        """
        self.dir_path = dir_path
        self.file_name_pattern = file_name_pattern

    def generate(self, y_chunk_index: int, x_chunk_index: int) -> str:
        """Generates path for a chunk file for given positional indices.

        Args:
            y_chunk_index: Index along the Y-axis for a chunk position.
            x_chunk_index: Index along the X-axis for a chunk position.

        Returns:
            A path to a chunk file.
        """
        return os.path.join(
            self.dir_path, self.file_name_pattern.format(y_chunk_index, x_chunk_index)
        )

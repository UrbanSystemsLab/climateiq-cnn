import abc


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

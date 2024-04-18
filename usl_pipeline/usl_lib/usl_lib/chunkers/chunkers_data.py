import dataclasses
import pathlib


@dataclasses.dataclass
class ChunkDescriptor:
    """Information about chunk file."""

    y_chunk_index: int
    x_chunk_index: int
    path: pathlib.Path

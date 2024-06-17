import typing
from typing import Dict

import h5py
import numpy
import numpy.typing as npt


def read_from_hdf5(
    input_stream: typing.BinaryIO,
    group_name: str = "features",
) -> Dict[str, npt.NDArray[numpy.float32]]:
    """Reads feature matrix layers from HDF5 file.

    Args:
        input_stream: Input byte stream to read from.
        group_name: Name of HDF5 group where layers are stored.

    Returns:
        Dictionary with keys corresponding to feature layer names and values containing
        Numpy 2d arrays of each feature layer.
    """
    hf = h5py.File(input_stream, "r")
    group = hf.get(group_name)
    layers: Dict[str, npt.NDArray[numpy.float32]] = {}
    for key, layer in group.items():
        layers[key] = numpy.array(layer, dtype=numpy.float32)
    hf.close()
    return layers

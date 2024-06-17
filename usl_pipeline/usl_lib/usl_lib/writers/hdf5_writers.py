import typing
from typing import Dict

import h5py
import numpy
import numpy.typing as npt


def write_to_hdf5(
    layers: Dict[str, npt.NDArray[numpy.float32]],
    output_stream: typing.BinaryIO,
    group_name: str = "features",
) -> None:
    """Reads feature matrix layers from HDF5 file.

    Args:
        layers: Dictionary with keys corresponding to feature layer names and values
            containing Numpy 2d arrays of each feature layer.
        output_stream: Output byte stream to write to.
        group_name: Name of HDF5 group to attach layers to.
    """
    hf = h5py.File(output_stream, "w")
    group = hf.create_group(group_name)
    for key, layer in layers.items():
        group.create_dataset(key, data=layer)
    hf.close()

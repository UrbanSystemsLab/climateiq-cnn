import typing
from typing import Dict

import h5py
import numpy.typing as npt


# Name of HDF5 group where feature layers are stored.
DEFAULT_HDF5_FEATURE_GROUP = "features"


def write_to_hdf5(
    layers: Dict[str, npt.NDArray],
    output_stream: typing.BinaryIO,
) -> None:
    """Reads feature matrix layers from HDF5 file.

    Args:
        layers: Dictionary with keys corresponding to feature layer names and values
            containing Numpy 2d arrays of each feature layer.
        output_stream: Output byte stream to write to.
    """
    with h5py.File(output_stream, "w") as hf:
        group = hf.create_group(DEFAULT_HDF5_FEATURE_GROUP)
        for key, layer in layers.items():
            group.create_dataset(key, data=layer)

import typing
from typing import Dict

import h5py
import numpy.typing as npt

# Name of HDF5 group where feature layers are stored.
DEFAULT_HDF5_FEATURE_GROUP = "features"


def read_from_hdf5(
    input_stream: typing.BinaryIO,
) -> Dict[str, npt.NDArray]:
    """Reads feature matrix layers from HDF5 file.

    Args:
        input_stream: Input byte stream to read from.

    Returns:
        Dictionary with keys corresponding to feature layer names and values containing
        Numpy 2d arrays of each feature layer.
    """
    with h5py.File(input_stream, "r") as hf:
        group = hf.get(DEFAULT_HDF5_FEATURE_GROUP)
        return {key: layer[:] for key, layer in group.items()}

"""Data functions for Flood CNN model."""

import numpy as np
from numpy.lib import stride_tricks


def temporal_window_view(data: np.ndarray, window: int) -> np.ndarray:
    """Creates a sliding window view over 1D temporal data.

    Args:
        data: A 2D array, where the first axis is the batch dimension and the
            second axis is the temporal data. Must be coercible into a
            np.ndarray.
        window: The window size.

    Returns:
        A 3D array of shape [B, T, window]. In order to maintain the same T,
        we use zero padding at the beginning of the temporal data.
    """
    B = data.shape[0]
    padding = np.zeros([B, window - 1], dtype=np.float32)
    data = np.concatenate([padding, data], axis=-1)
    return stride_tricks.sliding_window_view(data, window, axis=-1)

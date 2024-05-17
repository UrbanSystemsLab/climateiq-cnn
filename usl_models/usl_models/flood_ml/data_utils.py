"""Data functions for Flood CNN model."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.lib import stride_tricks
import tensorflow as tf


@dataclass(kw_only=True, slots=True)
class FloodModelData:
    """Dataclass for flood model data.

    Args:
        storm_duration: Storm duration, as a unit of time steps (at 5-min
            resolution). Synonymous with the number of flood predictions.
        geospatial: Geospatial feature tensor. [H, W, f]
        temporal: Temporal rainfall data. This can be 1D, in which case the
            window view needs to be applied, or 2D, which is assumed to follow
            the expected window view. [T_max(, m)]
        spatiotemporal: Optional. A *single* flood map to specify the initial
            flood conditions. [H, W, 1]
        labels: Flood map labels, required during training. The first axis must
            be the same as storm_duration.
    """

    storm_duration: int
    geospatial: tf.Tensor
    temporal: tf.Tensor
    spatiotemporal: Optional[tf.Tensor] = None
    # Labels must be included during training.
    labels: Optional[tf.Tensor] = None


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

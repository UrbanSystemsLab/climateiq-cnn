"""Data functions for Flood CNN model."""

from constants import GEO_FEATURES, MAP_HEIGHT, MAP_WIDTH, N_FLOOD_MAPS

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import tensorflow as tf


def temporal_window_view(data, window):
    """Creates a sliding window view over 1D temporal data.

    Args:
        data: A 2D tensor, where the first axis is the batch dimension and the
            second axis is the temporal data. Shape: [B, T].
        window: The window size.

    Returns:
        A 3D tensor of shape [B, T, window]. In order to maintain the same T,
        we use zero padding at the beginning of the temporal data.
    """
    B = data.shape[0]
    padding = np.zeros([B, window - 1])
    data = np.concatenate([padding, data], axis=-1)
    return sliding_window_view(data, window, axis=-1)


def fake_input_batch(
    batch_size,
    height=None,
    width=None,
    rainfall_duration=24,
    flood_maps=None,
    include_flood_maps=False,
):
    """Creates a fake training batch for testing.

    All dimensions can be specified via optional args, otherwise will use the values
    defined within constants.py.

    Args:
        batch_size: Batch size, axis 0.
        height: Optional height. Defaults to MAP_HEIGHT.
        width: Optional width. Defaults to MAP_WIDTH.
        rainfall_duration: Optional rainfall duration.
        flood_maps: Optional number of flood maps. Defaults to N_FLOOD_MAPS.
            Only used if include_flood_maps=True.
        include_flood_maps: Whether or not to pass in flood maps as
            spatiotemporal inputs.
    """
    height = height or MAP_HEIGHT
    width = width or MAP_WIDTH

    geospatial = tf.random.normal((batch_size, height, width, GEO_FEATURES))
    temporal = tf.random.normal((batch_size, rainfall_duration))

    input = {"geospatial": geospatial, "temporal": temporal}

    if include_flood_maps:
        flood_maps = flood_maps or N_FLOOD_MAPS
        spatiotemporal = tf.random.normal((batch_size, flood_maps, height, width, 1))
        input["spatiotemporal"] = spatiotemporal

    return input

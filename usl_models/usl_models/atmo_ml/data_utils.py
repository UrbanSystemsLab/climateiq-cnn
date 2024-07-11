"""Data functions for AtmoML model."""

from numpy.lib import stride_tricks
import tensorflow as tf


def boundary_pairs(data: tf.Tensor) -> tf.Tensor:
    """Retrieves boundary condition pairs from a time sequence.

    Takes a time sequence of T values and returns (T-1) pairs of adjacent
    values (i.e., boundary conditions).

        Input:  [T0, T1, T2, T3]
        Output: [[T0, T1], [T1, T2], [T2, T3]]

    The pairs are stored along a new axis. In order to maintain consistent
    dimensionality, the tensor is reshaped. Notably, this interleaves the
    channel features between the two time steps.

        Example: Consider features {x0, y0, z0} at time 0, and {x1, y1, z1} at
        time 1. The channels axis will be ordered as [x0, x1, y0, y1, z0, z1].

    Args:
        data: A time sequence of tensors of shape [B, T, H, W, C].

    Returns:
        A time sequence of "boundary condition" paired tensors of shape
        [B, T-1, H, W, 2C].
    """
    pairs = stride_tricks.sliding_window_view(data, 2, axis=1)
    pairs = tf.reshape(pairs, pairs.shape[:-2] + (-1,))
    return pairs

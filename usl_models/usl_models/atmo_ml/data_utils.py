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


def split_time_step_pairs(data: tf.Tensor) -> tf.Tensor:
    """Splits a tensor into two time steps.

    Takes a time sequence of T paired tensors and returns 2T single tensors.
    This function follows the work done by boundary_pairs, which groups tensors
    tensors up, and splits them up again prior to retrieving the final output.

        Input: [[T0, T1], [T2, T3], [T4, T5]]
        Output: [T0, T1, T2, T3, T4, T5]

    For more details about this function and boundary_pairs, see
    https://www.notion.so/climate-iq/AtmoML-Architecture-Proposals-and-Design-c00d0e54265c4bb8a72ce01fd475f116?pvs=4#455d785c174a482c90a78efde380adf3.

    Args:
        data: A time sequence of paired tensors of shape [B, T, H, W, 2C].

    Returns:
        A time sequence of single tensors of shape [B, 2T, H, W, C].
    """
    # We need to permute the tensor to place the time and channel axes together
    # during the reshape, in order to avoid disturbing the spatial dimensions.
    permuted = tf.transpose(data, (0, 2, 3, 1, 4))
    shape = permuted.shape[:3] + (permuted.shape[-2] * 2, permuted.shape[-1] // 2)
    reshaped = tf.reshape(permuted, shape)
    output = tf.transpose(reshaped, (0, 3, 1, 2, 4))
    return output

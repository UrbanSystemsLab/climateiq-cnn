"""Data functions for AtmoML model."""

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
    # Shift the time dimension by 1 and pair each step with the next one
    t1 = data[:, :-1, :, :, :]  # [B, T-1, H, W, C]
    t2 = data[:, 1:, :, :, :]  # [B, T-1, H, W, C]

    # Concatenate along the channel axis to form pairs
    pairs = tf.concat([t1, t2], axis=-1)  # [B, T-1, H, W, 2C]
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
    # Transpose to [B, H, W, T, 2C] so that we can split the channel dimension
    permuted = tf.transpose(data, (0, 2, 3, 1, 4))

    # Get the dynamic shape
    batch_size = tf.shape(permuted)[0]
    height = tf.shape(permuted)[1]
    width = tf.shape(permuted)[2]
    time_steps = (tf.shape(permuted)[3] - 1) * 2  # Original time steps are doubled
    channels = tf.shape(permuted)[4] // 2  # Original channels are halved

    # Reshape to [B, H, W, 2T, C]
    reshaped = tf.reshape(permuted, (batch_size, height, width, time_steps, channels))

    # Transpose back to [B, 2T, H, W, C]
    output = tf.transpose(reshaped, (0, 3, 1, 2, 4))
    return output

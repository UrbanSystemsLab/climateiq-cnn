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

    Args:
        data: A time sequence of paired tensors of shape [B, T, H, W, 2C].

    Returns:
        A time sequence of single tensors of shape [B, 2T - 2, H, W, C].
    """
    # Transpose to [B, H, W, T, 2C] for easier manipulation
    permuted = tf.transpose(data, (0, 2, 3, 1, 4))
    # Get the dynamic shape
    batch_size = tf.shape(permuted)[0]
    height = tf.shape(permuted)[1]
    width = tf.shape(permuted)[2]
    original_time_steps = tf.shape(permuted)[3]  # This will be T
    channels = tf.shape(permuted)[4] // 2  # Halve the channels (2C -> C)
    # Split the time sequence into two halves for each pair
    # Example: [ (X_{d-1}^{18}, X^0), (X^0, X^6),..] -> separate into individual tensors
    split = tf.reshape(
        permuted, (batch_size, height, width, original_time_steps, 2, channels)
    )
    print("Split shape:", split.shape)
    # Reshape into individual time steps
    separated = tf.reshape(
        split, (batch_size, height, width, original_time_steps * 2, channels)
    )
    print("Separated shape (pre-slice):", separated.shape)
    # Remove the outermost single time steps (X_{d-1}^{18} and X_{d+1}^0)
    reduced = separated[:, :, :, 1:-1, :]
    print("Reduced shape after removing outermost time steps:", reduced.shape)
    # Transpose back to [B, T_adjusted, H, W, C]
    output = tf.transpose(reduced, (0, 3, 1, 2, 4))
    tf.print("Output shape:", output.shape)
    return output

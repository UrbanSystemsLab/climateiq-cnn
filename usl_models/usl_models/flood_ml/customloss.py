"""loss used for FloodML modeling."""

import tensorflow as tf


def weighted_mse_small_targets(y_true, y_pred):
    """Custom MSE loss that gives higher importance to smaller target values."""
    # Ignore NaNs (if any) in labels
    mask = tf.math.logical_not(tf.math.is_nan(y_true))
    y_true = tf.where(mask, y_true, tf.zeros_like(y_true))
    y_pred = tf.where(mask, y_pred, tf.zeros_like(y_pred))

    # Define inverse-weighting: smaller values get higher weight
    weights = 1.0 / (1.0 + 100.0 * y_true)  # e.g., 0.001 -> ~1, 0.1 -> ~0.09
    squared_error = tf.square(y_true - y_pred)
    weighted_squared_error = weights * squared_error

    # Normalize by number of valid entries
    return tf.reduce_sum(weighted_squared_error) / tf.reduce_sum(
        tf.cast(mask, tf.float32)
    )


def hybrid_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    small_weighted = weighted_mse_small_targets(y_true, y_pred)
    return 0.5 * mse + 0.5 * small_weighted

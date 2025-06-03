"""loss used for FloodML modeling."""

import tensorflow as tf


def weighted_mse_small_targets(y_true, y_pred, scale=100.0):
    """Custom MSE loss that gives higher importance to smaller target values.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        scale: It adjusts the emphasis on smaller targets. Default is 100.

    Returns:
        A scalar loss value.
    """
    mask = tf.math.logical_not(tf.math.is_nan(y_true))
    y_true = tf.where(mask, y_true, tf.zeros_like(y_true))
    y_pred = tf.where(mask, y_pred, tf.zeros_like(y_pred))
    weights = 1.0 / (1.0 + scale * y_true)
    squared_error = tf.square(y_true - y_pred)
    weighted_squared_error = weights * squared_error
    return tf.reduce_sum(weighted_squared_error) / tf.reduce_sum(
        tf.cast(mask, tf.float32)
    )


def make_hybrid_loss(scale=100.0):
    def loss_fn(y_true, y_pred):
        mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
        small_weighted = weighted_mse_small_targets(y_true, y_pred, scale=scale)
        return 0.5 * mse + 0.5 * small_weighted

    return loss_fn

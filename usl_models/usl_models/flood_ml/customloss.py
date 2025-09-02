"""loss used for FloodML modeling."""

import tensorflow as tf
from keras.saving import register_keras_serializable


def weighted_mse_small_targets(y_true, y_pred):
    """Custom MSE loss that gives higher importance to smaller target values.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        A scalar loss value.
    """
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


def log_cosh_loss(y_true, y_pred):
    diff = y_pred - y_true
    return tf.reduce_mean(tf.math.log(tf.cosh(diff + 1e-12)))


# @register_keras_serializable(package="Custom", name="loss_fn")
# def make_hybrid_loss(scale=100.0):
#     def loss_fn(y_true, y_pred):
#         # mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
#         logcosh = log_cosh_loss(y_true, y_pred)
#         # small_weighted = weighted_mse_small_targets(y_true, y_pred, scale=scale)
#         # return 0.3 * mse + 0.4 * small_weighted + 0.3 * logcosh
#         return logcosh


# return loss_fn
@register_keras_serializable(package="Custom", name="make_hybrid_loss")
def make_hybrid_loss(y_true, y_pred):
    logcosh = log_cosh_loss(y_true, y_pred)
    # mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    # small_weighted = weighted_mse_small_targets(y_true, y_pred)
    return logcosh

"""Evaluation metrics for FloodML."""

import tensorflow as tf


def spatial_mae(predictions: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    """Calculates spatial MAE over a batch of predictions.

    Args:
        predictions: [N, H, W] tensor of predictions.
        labels: [N, H, W] tensor of ground-truth labels.

    Returns:
        [H, W] tensor of the absolute error, averaged over the N examples.
    """
    return tf.reduce_mean(tf.abs(predictions - labels), axis=0)


def temporal_mae(predictions: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    """Calculates MAE over a time series.

    Calculates the MAE separately at each time step. This is useful to evaluate
    whether prediction errors are propagated over the course of the autoregression
    (i.e., if errors are increasing over time).

    We expect T to be constant.

    Args:
        predictions: [N, T, H, W] tensor of predictions.
        labels: [N, T, H, W] tensor of ground-truth labels.

    Returns:
        T-length tensor of the MAE.
    """
    # We want to aggregate over all axes except time.
    axes = (0, 2, 3)
    return tf.reduce_mean(tf.abs(predictions - labels), axis=axes)


def temporal_rmse(predictions: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    """Calculates RMSE over a time series.

    Calculates the RMSE separately at each time step. This is useful to evaluate
    whether prediction errors are propagated over the course of the autoregression
    (i.e., if errors are increasing over time).

    We expect T to be constant.

    Args:
        predictions: [N, T, H, W] tensor of predictions.
        labels: [N, T, H, W] tensor of ground-truth labels.

    Returns:
        T-length tensor of the RMSE.
    """
    # We want to aggregate over all axes except time.
    axes = (0, 2, 3)
    return tf.sqrt(tf.reduce_mean((predictions - labels) ** 2, axis=axes))

"""Evaluation metrics for FloodML.

By default, all the calculations are assumed to be over a single prediction/label
due to the computational cost of model prediction. If you want to calculate
metrics *per prediction* over a batch, they should be unstacked into individual
examples:

```
for predictions, labels in batch_predict():
    for prediction, label in zip(tf.unstack(predictions), tf.unstack(labels)):
        # calculate metrics
```

Otherwise, specifying `batch = True` in any of the metric functions will average
over the examples.
"""

import tensorflow as tf


def spatial_mae(
    predictions: tf.Tensor, labels: tf.Tensor, batch: bool = False
) -> tf.Tensor:
    """Calculates spatial MAE.

    Args:
        predictions: [H, W] tensor of predictions.
        labels: [H, W] tensor of ground-truth labels.
        batch: Whether the inputs include a batch dimension, in which case they
            have shape [N, H, W].

    Returns:
        [H, W] tensor of the MAE. Note that in the absence of a batch dimension,
        this is equivalent to the absolute error.
    """
    abs_error = tf.abs(predictions - labels)
    if batch:
        return tf.reduce_mean(abs_error, axis=0)
    return abs_error


def spatial_nse(
    predictions: tf.Tensor, labels: tf.Tensor, batch: bool = False
) -> tf.Tensor:
    """Calculates spatial Nash-Sutcliffe efficiency (NSE).

    Args:
            predictions: [H, W] tensor of predictions.
            labels: [H, W] tensor of ground-truth labels.
            batch: Whether the inputs include a batch dimension, in which case they
                have shape [N, H, W].

    Returns:
        [H, W] tensor of NSE. If a batch dimension is included, the output is the
        average NSE over all examples.
    """
    temp_axis = 1 if batch else 0
    obs_mean = tf.reduce_mean(labels, axis=temp_axis)
    # Add temporal dimension back into obs_mean so shapes are compatible
    obs_mean = tf.expand_dims(obs_mean, temp_axis)
    mse = tf.reduce_sum((labels - predictions) ** 2, axis=temp_axis)
    nse = 1 - mse / tf.reduce_sum((labels - obs_mean) ** 2, axis=temp_axis)
    if batch:
        return tf.reduce_mean(nse, axis=0)
    return nse


def temporal_mae(
    predictions: tf.Tensor, labels: tf.Tensor, batch: bool = False
) -> tf.Tensor:
    """Calculates MAE over a time series.

    Calculates the MAE separately at each time step. This is useful to evaluate
    whether prediction errors are propagated over the course of the autoregression
    (i.e., if errors are increasing over time).

    If passing in a batch of predictions, we compute the error over the batch
    dimension too.

    Args:
        predictions: [T, H, W] tensor of predictions.
        labels: [T, H, W] tensor of ground-truth labels.
        batch: Whether the inputs include a batch dimension, in which case they
            have shape [N, T, H, W].

    Returns:
        T-length tensor of the MAE.
    """
    # We want to aggregate over all axes except time.
    axes = (0, 2, 3) if batch else (1, 2)
    return tf.reduce_mean(tf.abs(predictions - labels), axis=axes)


def temporal_rmse(
    predictions: tf.Tensor, labels: tf.Tensor, batch: bool = False
) -> tf.Tensor:
    """Calculates RMSE over a time series.

    Calculates the RMSE separately at each time step. This is useful to evaluate
    whether prediction errors are propagated over the course of the autoregression
    (i.e., if errors are increasing over time).

    If passing in a batch of predictions, we compute the error over the batch
    dimension too.

    Args:
        predictions: [T, H, W] tensor of predictions.
        labels: [T, H, W] tensor of ground-truth labels.
        batch: Whether the inputs include a batch dimension, in which case they
            have shape [N, T, H, W].

    Returns:
        T-length tensor of the RMSE.
    """
    # We want to aggregate over all axes except time.
    axes = (0, 2, 3) if batch else (1, 2)
    return tf.sqrt(tf.reduce_mean((predictions - labels) ** 2, axis=axes))

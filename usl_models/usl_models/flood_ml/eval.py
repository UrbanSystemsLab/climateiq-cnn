"""Evaluation metrics for FloodML.

There are two versions for each metric, one computing over a single prediction
and another computing over a batch of predictions.

The batch version of each metric function will include the batch dimension in
the computation for more aggregate metrics. These may be better for reporting
general performance, but will lose some of the granularity needed for digging
into individual predictions, e.g., for explainability.

If you want to calculate metrics *per prediction* over a batch, the batch
should be unstacked into individual examples:

```
for predictions, labels in batch_predict():
    for prediction, label in zip(tf.unstack(predictions), tf.unstack(labels)):
        # use the single prediction version of the metric
        mae = spatial_mae(prediction, label)
```
"""

import tensorflow as tf


def spatial_mae(prediction: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
    """Calculates spatial MAE over a single prediction.

    Args:
        prediction: [H, W] prediction tensor.
        label: [H, W] label tensor.

    Returns:
        [H, W] tensor of the MAE. Note that in the absence of multiple examples,
        this is equivalent to the absolute error.
    """
    assert prediction.shape == label.shape
    assert tf.rank(prediction) == 2

    return tf.abs(prediction - label)


def batch_spatial_mae(predictions: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    """Calculates spatial MAE over a batch of predictions.

    Args:
        predictions: [B, H, W] tensor of predictions.
        labels: [B, H, W] tensor of ground-truth labels.

    Returns:
        [H, W] tensor of the MAE.
    """
    assert predictions.shape == labels.shape
    assert tf.rank(predictions) == 3

    return tf.reduce_mean(tf.abs(predictions - labels), axis=0)


def spatial_nse(prediction: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
    """Calculates spatial NSE over a single prediction sequence.

    Args:
        prediction: [T, H, W] prediction sequence tensor.
        label: [T, H, W] label sequence tensor.

    Returns:
        [H, W] tensor of Nash-Sutcliffe efficiency (NSE).
    """
    assert prediction.shape == label.shape
    assert tf.rank(prediction) == 3

    temp_axis = 0
    obs_mean = tf.reduce_mean(label, axis=temp_axis)
    mse = tf.reduce_sum((label - prediction) ** 2, axis=temp_axis)
    return 1 - mse / tf.reduce_sum((label - obs_mean) ** 2, axis=temp_axis)


def batch_spatial_nse(predictions: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    """Calculates spatial NSE over a batch of prediction sequences.

    Args:
        predictions: [B, T, H, W] tensor of time series predictions.
        labels: [B, T, H, W] tensor of time series ground-truth labels.

    Returns:
        [H, W] tensor of Nash-Sutcliffe efficiency (NSE) averaged over the
        entire batch.
    """
    assert predictions.shape == labels.shape
    assert tf.rank(predictions) == 4

    temp_axis = 1
    obs_mean = tf.reduce_mean(labels, axis=temp_axis)
    # Add temporal dimension back so shapes are compatible
    obs_mean = tf.expand_dims(obs_mean, temp_axis)
    mse = tf.reduce_sum((labels - predictions) ** 2, axis=temp_axis)
    nse = 1 - mse / tf.reduce_sum((labels - obs_mean) ** 2, axis=temp_axis)
    return tf.reduce_mean(nse, axis=0)


def temporal_mae(prediction: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
    """Calculates MAE per time step over a single prediction sequence.

    Calculates the MAE separately at each time step. This is useful to evaluate
    whether prediction errors are propagated over the course of the autoregression
    (i.e., if errors are increasing over time).

    Args:
        prediction: [T, H, W] tensor of predictions.
        label: [T, H, W] tensor of ground-truth labels.

    Returns:
        T-length tensor of the MAE. Note that in the absence of multiple examples,
        this is equivalent to the absolute error at each time step.
    """
    assert prediction.shape == label.shape
    assert tf.rank(prediction) == 3

    # We want to aggregate over all axes except time.
    return tf.reduce_mean(tf.abs(prediction - label), axis=(1, 2))


def batch_temporal_mae(predictions: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    """Calculates MAE per time step over a batch of prediction sequences.

    Calculates the MAE separately at each time step. This is useful to evaluate
    whether prediction errors are propagated over the course of the autoregression
    (i.e., if errors are increasing over time).

    Args:
        predictions: [B, T, H, W] tensor of prediction sequences.
        labels: [B, T, H, W] tensor of ground-truth label sequences.

    Returns:
        T-length tensor of the MAE.
    """
    assert predictions.shape == labels.shape
    assert tf.rank(predictions) == 4

    # We want to aggregate over all axes except time.
    return tf.reduce_mean(tf.abs(predictions - labels), axis=(0, 2, 3))


def temporal_rmse(prediction: tf.Tensor, label: tf.Tensor) -> tf.Tensor:
    """Calculates RMSE per time step over a single prediction sequence.

    Calculates the RMSE separately at each time step. This is useful to evaluate
    whether prediction errors are propagated over the course of the autoregression
    (i.e., if errors are increasing over time).

    Args:
        prediction: [T, H, W] tensor of predictions.
        label: [T, H, W] tensor of ground-truth labels.

    Returns:
        T-length tensor of the RMSE.
    """
    assert prediction.shape == label.shape
    assert tf.rank(prediction) == 3

    # We want to aggregate over all axes except time.
    return tf.sqrt(tf.reduce_mean((prediction - label) ** 2, axis=(1, 2)))


def batch_temporal_rmse(
    predictions: tf.Tensor, labels: tf.Tensor, batch: bool = False
) -> tf.Tensor:
    """Calculates RMSE per time step over a batch of prediction sequences.

    Calculates the RMSE separately at each time step. This is useful to evaluate
    whether prediction errors are propagated over the course of the autoregression
    (i.e., if errors are increasing over time). The error is computed over the
    batch dimension as well.

    Args:
        predictions: [B, T, H, W] tensor of prediction sequences.
        labels: [B, T, H, W] tensor of ground-truth label sequences.

    Returns:
        T-length tensor of the RMSE.
    """
    assert predictions.shape == labels.shape
    assert tf.rank(predictions) == 4

    # We want to aggregate over all axes except time.
    return tf.sqrt(tf.reduce_mean((predictions - labels) ** 2, axis=(0, 2, 3)))

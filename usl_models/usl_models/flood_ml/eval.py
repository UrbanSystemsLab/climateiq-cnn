"""Evaluation metrics for FloodML."""

import numpy as np


def spatial_mae(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Calculates spatial MAE over a batch of predictions.

    Args:
        predictions: [N, H, W] array of predictions.
        labels: [N, H, W] array of ground-truth labels.

    Returns:
        [H, W] array of the absolute error, averaged over the N examples.
    """
    return np.mean(np.absolute(predictions - labels), axis=0)


def time_series_errors(
    predictions: np.ndarray, labels: np.ndarray
) -> dict[str, np.ndarray]:
    """Calculates various errors over a time series.

    Calculates error at each time step. This is useful to evaluate whether
    prediction errors are propagated over the course of the autoregression
    (i.e., if errors are increasing over time).

    Numpy doesn't like ragged arrays, so we expect T to be constant.

    Args:
        predictions: [N, T, H, W] array of predictions.
        labels: [N, T, H, W] array of ground-truth labels.

    Returns:
        Dictionary of error arrays of length T.
    """
    # We want to aggregate over all axes except time.
    axes = (0, 2, 3)
    rmse = np.sqrt(np.mean((predictions - labels) ** 2, axis=axes))
    mae = np.mean(np.absolute(predictions - labels), axis=axes)

    errors = {
        "mae": mae,
        "rmse": rmse,
    }
    return errors

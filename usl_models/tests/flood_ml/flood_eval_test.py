"""Tests for FloodML evaluation metrics."""

import numpy as np

from usl_models.flood_ml import eval


def test_spatial_mae():
    """Tests spatial MAE calculation."""
    label = np.arange(18, dtype=np.float64).reshape((2, 3, 3))
    pred = label * 1.01

    actual = eval.spatial_mae(pred, label)
    expected = [
        [0.045, 0.055, 0.065],
        [0.075, 0.085, 0.095],
        [0.105, 0.115, 0.125],
    ]

    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_time_series_errors():
    """Tests time series error calculations."""
    label = np.arange(72, dtype=np.float32).reshape((2, 4, 3, 3))
    error = np.arange(8).reshape((2, 4)) / 10
    pred = label + np.tile(np.expand_dims(error, (2, 3)), (1, 1, 3, 3))

    actual = eval.time_series_errors(pred, label)
    expected = {
        "mae": [0.2, 0.3, 0.4, 0.5],
        "rmse": np.sqrt([0.08, 0.13, 0.2, 0.29]),
    }

    assert actual.keys() == expected.keys()
    for k, v in actual.items():
        np.testing.assert_allclose(v, expected[k], rtol=1e-6)

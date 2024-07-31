"""Tests for FloodML evaluation metrics."""

import numpy as np
import tensorflow as tf

from usl_models.flood_ml import eval


def test_spatial_mae():
    """Tests spatial MAE calculation."""
    label = tf.reshape(tf.range(9, dtype=tf.float64), (3, 3))
    pred = label * 1.01

    actual = eval.spatial_mae(pred, label)
    expected = tf.convert_to_tensor(
        [
            [0.0, 0.01, 0.02],
            [0.03, 0.04, 0.05],
            [0.06, 0.07, 0.08],
        ],
        dtype=tf.float64,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_batch_spatial_mae():
    """Tests spatial MAE calculation over a batch."""
    label = tf.reshape(tf.range(18, dtype=tf.float64), (2, 3, 3))
    pred = label * 1.01

    actual = eval.batch_spatial_mae(pred, label)
    expected = tf.convert_to_tensor(
        [
            [0.045, 0.055, 0.065],
            [0.075, 0.085, 0.095],
            [0.105, 0.115, 0.125],
        ],
        dtype=tf.float64,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_spatial_nse():
    """Tests spatial NSE calculation."""
    label = tf.transpose(
        tf.reshape(tf.range(36, dtype=tf.float64), (3, 3, 4)), (2, 0, 1)
    )
    error = tf.range(4, dtype=tf.float64) / 10
    pred = label + tf.tile(error[:, tf.newaxis, tf.newaxis], (1, 3, 3))

    actual = eval.spatial_nse(pred, label)
    expected = tf.broadcast_to(0.972, (3, 3))
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_batch_spatial_nse():
    """Tests spatial NSE calculation over a batch."""
    label = tf.transpose(
        tf.reshape(tf.range(72, dtype=tf.float64), (2, 3, 3, 4)), (0, 3, 1, 2)
    )
    error = tf.reshape(tf.range(8, dtype=tf.float64), (2, 4)) / 10
    pred = label + tf.tile(error[:, :, tf.newaxis, tf.newaxis], (1, 1, 3, 3))

    actual = eval.batch_spatial_nse(pred, label)
    # NSEs per batch: [0.972, 0.748] -> avg = 0.86
    expected = tf.broadcast_to(0.86, (3, 3))
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_spatial_nse_zero_deviation():
    """Tests spatial NSE calculation when there's no temporal deviation.

    When the label doesn't change over the course of the entire time series,
    the SSD in the denominator evaluates to zero, which can lead to NaN and
    inf values.

    This test checks that the epsilon added to the denominator prevents such
    non-numerical outputs.
    """
    shape = (4, 3, 3)
    label = tf.ones(shape)
    pred = label + tf.random.normal(shape)

    nse = eval.spatial_nse(pred, label)
    assert tf.reduce_all(tf.math.is_finite(nse))


def test_temporal_mae():
    """Tests temporal MAE calculation."""
    label = tf.reshape(tf.range(36, dtype=tf.float64), (4, 3, 3))
    error = tf.range(4, dtype=tf.float64) / 10
    pred = label + tf.tile(error[:, tf.newaxis, tf.newaxis], (1, 3, 3))

    actual = eval.temporal_mae(pred, label)
    expected = tf.convert_to_tensor([0.0, 0.1, 0.2, 0.3], dtype=tf.float64)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_batch_temporal_mae():
    """Tests temporal MAE calculation over a batch."""
    label = tf.reshape(tf.range(72, dtype=tf.float64), (2, 4, 3, 3))
    error = tf.reshape(tf.range(8, dtype=tf.float64), (2, 4)) / 10
    pred = label + tf.tile(error[:, :, tf.newaxis, tf.newaxis], (1, 1, 3, 3))

    actual = eval.batch_temporal_mae(pred, label)
    expected = tf.convert_to_tensor([0.2, 0.3, 0.4, 0.5], dtype=tf.float64)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_temporal_rmse():
    """Tests temporal RMSE calculation."""
    label = tf.reshape(tf.range(36, dtype=tf.float64), (4, 3, 3))
    error = tf.range(4, dtype=tf.float64) / 10
    pred = label + tf.tile(error[:, tf.newaxis, tf.newaxis], (1, 3, 3))

    actual = eval.temporal_rmse(pred, label)
    expected = tf.convert_to_tensor([0.0, 0.1, 0.2, 0.3], dtype=tf.float64)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_batch_temporal_rmse():
    """Tests temporal RMSE calculation over a batch."""
    label = tf.reshape(tf.range(72, dtype=tf.float64), (2, 4, 3, 3))
    error = tf.reshape(tf.range(8, dtype=tf.float64), (2, 4)) / 10
    pred = label + tf.tile(error[:, :, tf.newaxis, tf.newaxis], (1, 1, 3, 3))

    actual = eval.batch_temporal_rmse(pred, label)
    expected = expected = tf.sqrt([0.08, 0.13, 0.2, 0.29])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)

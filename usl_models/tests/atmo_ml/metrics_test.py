import numpy as np
import tensorflow as tf
from usl_models.atmo_ml import metrics


# Test for SSIMMetric: Identical images should have SSIM close to 1.
def test_ssim_metric_identical():
    # Create two identical tensors (batch, height, width, channels)
    x = tf.ones((2, 64, 64, 3))
    metric = metrics.SSIMMetric(max_val=1.0)
    metric.update_state(x, x)
    result = metric.result().numpy()
    np.testing.assert_allclose(result, 1.0, atol=1e-5)


# Test for SSIMMetric: Slightly different images should yield SSIM < 1.
def test_ssim_metric_different():
    x = tf.ones((2, 64, 64, 3))
    y = x + tf.random.uniform((2, 64, 64, 3), minval=-0.1, maxval=0.1)
    metric = metrics.SSIMMetric(max_val=1.0)
    metric.update_state(x, y)
    result = metric.result().numpy()
    assert result < 1.0


# Test for PSNRMetric: Identical images should yield a high PSNR.
def test_psnr_metric_identical():
    x = tf.ones((2, 64, 64, 3))
    metric = metrics.PSNRMetric(max_val=1.0)
    metric.update_state(x, x)
    result = metric.result().numpy()
    # For identical images, tf.image.psnr returns inf.
    assert np.isinf(
        result
    ), f"Expected PSNR to be inf for identical images, got {result}"


# Different images should yield a PSNR lower than for identical images.
def test_psnr_metric_different():
    x = tf.ones((2, 64, 64, 3))
    y = x + tf.random.uniform((2, 64, 64, 3), minval=-0.1, maxval=0.1)
    metric = metrics.PSNRMetric(max_val=1.0)
    metric.update_state(x, y)
    result = metric.result().numpy()
    assert result < 100.0


# Zero error when predictions equal ground truth.
def test_normalized_rmse_zero_error():
    x = tf.random.uniform((2, 10, 10, 1))
    metric = metrics.NormalizedRootMeanSquaredError()
    metric.update_state(x, x)
    result = metric.result().numpy()
    np.testing.assert_allclose(result, 0.0, atol=1e-5)


# Compute expected normalized RMSE for a simple case.
def test_normalized_rmse_nonzero():
    # Define a simple 2x2 ground truth and prediction.
    y_true = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    y_pred = y_true + 1.0  # constant error of 1.
    # Reshape to (batch, height, width, channels)
    y_true = tf.reshape(y_true, (1, 2, 2, 1))
    y_pred = tf.reshape(y_pred, (1, 2, 2, 1))
    metric = metrics.NormalizedRootMeanSquaredError()
    metric.update_state(y_true, y_pred)
    result = metric.result().numpy()
    # RMSE is 1; range is 4 - 1 = 3; so normalized RMSE should be about 1/3.
    np.testing.assert_allclose(result, 1.0 / 3, atol=1e-3)

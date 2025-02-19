"""Metrics used for AtmoML modeling.

Metrics used for AtmoML modeling.
"""

import tensorflow as tf

import keras

from usl_models.atmo_ml import vars


@keras.saving.register_keras_serializable()
class OutputVarMeanSquaredError(keras.metrics.MeanMetricWrapper):
    """Output variable mean squared error."""

    def __init__(self, sto_var: vars.SpatiotemporalOutput, name=None, dtype=tf.float32):
        """Create a mean squared error metric function for an sto_var."""

        def mse(y_true: tf.Tensor, y_pred: tf.Tensor):
            """Computes MSE on RH output tensor only."""
            return keras.metrics.mean_squared_error(
                y_true[:, :, :, :, sto_var.value],
                y_pred[:, :, :, :, sto_var.value],
            )

        self.sto_var = sto_var
        name = name or "mse_" + sto_var.name
        keras.metrics.MeanMetricWrapper.__init__(self, fn=mse, name=name, dtype=dtype)

    def get_config(self):
        """Returns the serializable config of the metric."""
        return {"name": self.name, "dtype": self.dtype, "sto_var": self.sto_var}


def ssim_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Compute the mean Structural Similarity Index (SSIM) between y_true and y_pred.

    SSIM is a perceptual metric that measures the structural similarity between images.
    Assumes inputs are scaled to [0, 1]. Higher values indicate better similarity.

    Args:
        y_true (tf.Tensor): Ground truth tensor.
        y_pred (tf.Tensor): Predicted tensor.

    Returns:
        tf.Tensor: Mean SSIM value.
    """
    ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
    return tf.reduce_mean(ssim)


def psnr_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Compute the mean Peak Signal-to-Noise Ratio (PSNR) between y_true and y_pred.

    PSNR measures the ratio between the maximum possible power of a signal
    and the power of the noise that affects the accuracy of its representation.

    Assumes inputs are scaled to [0, 1]. Higher values indicate less distortion.

    Args:
        y_true (tf.Tensor): Ground truth tensor.
        y_pred (tf.Tensor): Predicted tensor.

    Returns:
        tf.Tensor: Mean PSNR value.
    """
    psnr = tf.image.psnr(y_true, y_pred, max_val=1.0)
    return tf.reduce_mean(psnr)


class NormalizedRootMeanSquaredError(keras.metrics.Metric):
    """Compute the Normalized Root Mean Squared Error (NRMSE).

    This metric calculates the RMSE and normalizes it by the range (max-min)
    of the ground truth values over all batches.
    """

    def __init__(self, name="nrmse", **kwargs):
        """Initialize the Normalized RMSE metric.

        Args:
            name (str, optional): Name of the metric. Defaults to "nrmse".
            **kwargs: Additional arguments for the base class.
        """
        super().__init__(name=name, **kwargs)
        self.rmse = keras.metrics.RootMeanSquaredError()
        self.min_val = self.add_weight(name="min_val", initializer="zeros")
        self.max_val = self.add_weight(name="max_val", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        """Update the state variables for computing the metric.

        Args:
            y_true (tf.Tensor): Ground truth values.
            y_pred (tf.Tensor): Predicted values.
            sample_weight (tf.Tensor, optional): Optional weight for samples.
        """
        self.rmse.update_state(y_true, y_pred, sample_weight)
        current_min = tf.reduce_min(y_true)
        current_max = tf.reduce_max(y_true)

        if tf.equal(self.count, 0):
            self.min_val.assign(current_min)
            self.max_val.assign(current_max)
        else:
            self.min_val.assign(tf.minimum(self.min_val, current_min))
            self.max_val.assign(tf.maximum(self.max_val, current_max))

        self.count.assign_add(1)

    def result(self) -> tf.Tensor:
        """Compute the final NRMSE value.

        Returns:
            tf.Tensor: The normalized RMSE value.
        """
        range_val = self.max_val - self.min_val
        return tf.cond(
            tf.equal(range_val, 0.0),
            lambda: tf.constant(0.0),
            lambda: self.rmse.result() / range_val,
        )

    def reset_states(self):
        """Reset the metric state."""
        self.rmse.reset_states()
        self.min_val.assign(0.0)
        self.max_val.assign(0.0)
        self.count.assign(0.0)

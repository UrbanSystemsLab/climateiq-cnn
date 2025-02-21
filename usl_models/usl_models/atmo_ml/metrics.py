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


@keras.saving.register_keras_serializable(package="CustomMetrics")
class SSIMMetric(keras.metrics.Metric):
    """Computes the mean Structural Similarity Index (SSIM) between images."""

    def __init__(self, name="ssim_metric", max_val=1.0, **kwargs):
        """Initialize the SSIM metric.

        Args:
            name (str): Name of the metric.
            max_val (float): Maximum possible value in the images.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, **kwargs)
        self.max_val = max_val
        self.ssim_total = self.add_weight(name="ssim_total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        """Update state by computing SSIM for the current batch."""
        # Compute SSIM for each image in the batch and take the mean.
        ssim = tf.image.ssim(y_true, y_pred, max_val=self.max_val)
        self.ssim_total.assign_add(tf.reduce_mean(ssim))
        self.count.assign_add(1.0)

    def result(self) -> tf.Tensor:
        """Return the mean SSIM over all batches."""
        return self.ssim_total / self.count

    def reset_state(self):
        """Reset the metric state."""
        self.ssim_total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        """Return the configuration of the metric for serialization."""
        base_config = super().get_config()
        return {**base_config, "max_val": self.max_val}

    @classmethod
    def from_config(cls, config):
        """Create a new instance from the given configuration."""
        return cls(**config)


@keras.saving.register_keras_serializable(package="CustomMetrics")
class PSNRMetric(keras.metrics.Metric):
    """Computes the mean Peak Signal-to-Noise Ratio (PSNR) between images."""

    def __init__(self, name="psnr_metric", max_val=1.0, **kwargs):
        """Initialize the PSNR metric.

        Args:
            name (str): Name of the metric.
            max_val (float): Maximum possible pixel value.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(name=name, **kwargs)
        self.max_val = max_val
        self.psnr_total = self.add_weight(name="psnr_total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        """Update state by computing PSNR for the current batch."""
        psnr = tf.image.psnr(y_true, y_pred, max_val=self.max_val)
        self.psnr_total.assign_add(tf.reduce_mean(psnr))
        self.count.assign_add(1.0)

    def result(self) -> tf.Tensor:
        """Return the mean PSNR over all batches."""
        return self.psnr_total / self.count

    def reset_state(self):
        """Reset the metric state."""
        self.psnr_total.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        """Return the configuration of the metric for serialization."""
        base_config = super().get_config()
        return {**base_config, "max_val": self.max_val}

    @classmethod
    def from_config(cls, config):
        """Create a new instance from the given configuration."""
        return cls(**config)


@keras.utils.register_keras_serializable(package="CustomMetrics")
class NormalizedRootMeanSquaredError(keras.metrics.Metric):
    """Normalized Root Mean Squared Error."""

    def __init__(self, name="nrmse", **kwargs):
        """Initialize the NormalizedRootMeanSquaredError metric."""
        super().__init__(name=name, **kwargs)
        self.rmse = keras.metrics.RootMeanSquaredError()
        self.min_val = self.add_weight(name="min_val", initializer="zeros")
        self.max_val = self.add_weight(name="max_val", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        """Update the metric state using the current batch."""
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
        """Compute and return the normalized RMSE."""
        range_val = self.max_val - self.min_val
        return tf.cond(
            tf.equal(range_val, 0.0),
            lambda: tf.constant(0.0),
            lambda: self.rmse.result() / range_val,
        )

    def reset_state(self):
        """Reset the metric state."""
        self.rmse.reset_state()
        self.min_val.assign(0.0)
        self.max_val.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        """Return the configuration of the metric."""
        base_config = super().get_config()
        return {**base_config}

    @classmethod
    def from_config(cls, config):
        """Create a new instance from the given configuration."""
        return cls(**config)

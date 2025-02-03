"""Metrics used for AtmoML modeling.

Metrics used for AtmoML modeling.
"""

import tensorflow as tf

import keras

from usl_models.atmo_ml import vars


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

"""Flood model definition."""

from typing import Any

import tensorflow as tf
from tensorflow.keras import layers


class FloodConvLSTM(tf.keras.Model):
    """Flood ConvLSTM model.

    The architecture is an autoregressive ConvLSTM. Spatiotemporal and
    geospatial features are passed through initial CNN blocks for feature
    extraction, then concatenated with temporal inputs. The combined inputs are
    then passed into ConvLSTM and TransposeConv layers to output a map of flood
    predictions.

    The spatiotemporal inputs are "initial condition" flood maps, with previous
    flood predictions being fed back into the model for future predictions.
    This creates the autoregressive loop. We define a maximum number
    N_FLOOD_MAPS of flood maps to use as inputs.

    Architecture diagram: https://miro.com/app/board/uXjVKd7C19U=/.
    """

    def __init__(self, params: dict[str, Any]):
        """Creates the ConvLSTM model.

        Args:
            params: A dict of tunable model parameters.
        """
        super().__init__()

        self.params = params

        # Spatiotemporal CNN
        st_cnn_params = {"strides": 2, "padding": "same", "activation": "relu"}
        st_cnn_layers = [
            layers.Conv2D(16, 5, **st_cnn_params),
            layers.MaxPool2D(pool_size=2, strides=1, padding="same"),
            layers.Conv2D(64, 5, **st_cnn_params),
            layers.MaxPool2D(pool_size=2, strides=1, padding="same"),
        ]
        self.st_cnn = tf.keras.Sequential(
            [layers.TimeDistributed(layer) for layer in st_cnn_layers]
        )

        # Geospatial CNN
        geo_cnn_params = {"strides": 2, "padding": "same", "activation": "relu"}
        self.geo_cnn = tf.keras.Sequential(
            [
                layers.Conv2D(16, 5, **geo_cnn_params),
                layers.MaxPool2D(pool_size=2, strides=1, padding="same"),
                layers.Conv2D(64, 5, **geo_cnn_params),
                layers.MaxPool2D(pool_size=2, strides=1, padding="same"),
            ]
        )

        # ConvLSTM
        self.concat = layers.Concatenate(axis=-1)
        self.conv_lstm = layers.ConvLSTM2D(
            self.params["lstm_units"],
            self.params["lstm_kernels"],
            strides=1,
            padding="same",
            activation="tanh",
            dropout=self.params["lstm_dropout"],
            recurrent_dropout=self.params["lstm_recurrent_dropout"],
        )

        # Output (upsampling) CNN
        output_cnn_params = {"padding": "same", "activation": "relu"}
        self.output_cnn = tf.keras.Sequential(
            [
                layers.Conv2DTranspose(8, 4, strides=4, **output_cnn_params),
                layers.Conv2DTranspose(1, 1, strides=1, **output_cnn_params),
            ]
        )

    def forward(
        self,
        st_input: tf.Tensor,
        geo_input: tf.Tensor,
        temp_input: tf.Tensor,
    ) -> tf.Tensor:
        """Makes a single forward pass on a batch of data.

        The forward pass represents a single prediction on an input batch
        (i.e., a single flood map). This functions implements the logic of the
        internal ConvLSTM and ignores autoregressive steps.

        Args:
            st_input: Flood maps tensor of shape [B, n, H, W, 1].
            geo_input: Geospatial tensor of shape [B, H, W, f].
            temp_input: Rainfall windows tensor of shape [B, n, m].

        Returns:
            The flood map prediction. A tensor of shape [B, H, W, 1].
        """
        # Spatiotemporal CNN
        # [B, n, H, W, 1] -> [B, n, H', W', k1]
        st_cnn_output = self.st_cnn(st_input)

        # Geospatial CNN
        # [B, H, W, f ]-> [B, H', W', k2]
        # Add a new time axis and repeat n times -> [B, n, H', W', k2].
        n = st_input.shape[1]
        geo_cnn_output = self.geo_cnn(geo_input)
        geo_cnn_output = geo_cnn_output[:, tf.newaxis, :, :, :]
        geo_cnn_output = tf.repeat(geo_cnn_output, [n], axis=1)

        # Expand temporal inputs into maps
        # [B, n, m] -> [B, n, H', W', m]
        H_out = st_cnn_output.shape[2]
        W_out = st_cnn_output.shape[3]
        temp_input = temp_input[:, :, tf.newaxis, tf.newaxis, :]
        temp_input = tf.tile(temp_input, [1, 1, H_out, W_out, 1])

        # Concatenate and feed into remaining ConvLSTM and TransposeConv layers
        # [B, n, H', W', k'] -> [B, H, W, 1]
        lstm_input = self.concat([st_cnn_output, geo_cnn_output, temp_input])
        lstm_output = self.conv_lstm(lstm_input)
        output = self.output_cnn(lstm_output)

        return output

    def call(self):
        """Runs the entire autoregressive model."""
        raise NotImplementedError

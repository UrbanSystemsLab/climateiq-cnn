"""Flood model definition."""

from typing import Any, TypeAlias

import tensorflow as tf
from tensorflow.keras import layers

from usl_models.flood_ml import constants
from usl_models.flood_ml import data_utils

st_tensor: TypeAlias = tf.Tensor
geo_tensor: TypeAlias = tf.Tensor
temp_tensor: TypeAlias = tf.Tensor


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

    def __init__(
        self,
        params: dict[str, Any],
        n_predictions: int,
        spatial_dims: tuple[int, int] = (constants.MAP_HEIGHT, constants.MAP_WIDTH),
    ):
        """Creates the ConvLSTM model.

        Args:
            params: A dict of tunable model parameters.
            n_predictions: The number of predictions to make (i.e., simulation
                duration).
            spatial_dims: Tuple of spatial height and width input dimensions.
                Needed for defining input shapes. This is an optional arg that
                can be changed (primarily for testing/debugging).
        """
        super().__init__()

        self.params = params
        self.n_predictions = n_predictions
        self.spatial_height, self.spatial_width = spatial_dims

        # Spatiotemporal CNN
        st_cnn_params = {"strides": 2, "padding": "same", "activation": "relu"}
        self.st_cnn = tf.keras.Sequential(
            [
                # Input shape: (time_steps, height, width, channels)
                layers.InputLayer((None, self.spatial_height, self.spatial_width, 1)),
                # Remaining layers are TimeDistributed and are applied to each
                # temporal slice
                layers.TimeDistributed(layers.Conv2D(8, 5, **st_cnn_params)),
                layers.TimeDistributed(
                    layers.MaxPool2D(pool_size=2, strides=1, padding="same")
                ),
                layers.TimeDistributed(layers.Conv2D(16, 5, **st_cnn_params)),
                layers.TimeDistributed(
                    layers.MaxPool2D(pool_size=2, strides=1, padding="same")
                ),
            ],
            name="spatiotemporal_cnn",
        )

        # Geospatial CNN
        geo_cnn_params = {"strides": 2, "padding": "same", "activation": "relu"}
        self.geo_cnn = tf.keras.Sequential(
            [
                # Input shape: (height, width, channels)
                layers.InputLayer(
                    (self.spatial_height, self.spatial_width, constants.GEO_FEATURES)
                ),
                layers.Conv2D(16, 5, **geo_cnn_params),
                layers.MaxPool2D(pool_size=2, strides=1, padding="same"),
                layers.Conv2D(64, 5, **geo_cnn_params),
                layers.MaxPool2D(pool_size=2, strides=1, padding="same"),
            ],
            name="geospatial_cnn",
        )
        # This is a constant (time-invariant), so we cache the output.
        self.geo_cnn_output = None

        # ConvLSTM
        # The spatial dimensions have been reduced 4x by the CNNs.
        # The "channel" dimension is the sum of the channels from the CNNs
        # and the rainfall window size.
        conv_lstm_height = self.spatial_height // 4
        conv_lstm_width = self.spatial_width // 4
        conv_lstm_channels = 16 + 64 + self.params["m_rainfall"]
        self.conv_lstm = tf.keras.Sequential(
            [
                # Input shape: (time_steps, height, width, channels)
                layers.InputLayer(
                    (None, conv_lstm_height, conv_lstm_width, conv_lstm_channels)
                ),
                layers.ConvLSTM2D(
                    self.params["lstm_units"],
                    self.params["lstm_kernels"],
                    strides=1,
                    padding="same",
                    activation="tanh",
                    dropout=self.params["lstm_dropout"],
                    recurrent_dropout=self.params["lstm_recurrent_dropout"],
                ),
            ],
            name="conv_lstm",
        )

        # Output CNN (upsampling via TransposeConv)
        output_cnn_params = {"padding": "same", "activation": "relu"}
        self.output_cnn = tf.keras.Sequential(
            [
                # Input shape: (height, width, channels)
                layers.InputLayer(
                    (conv_lstm_height, conv_lstm_width, self.params["lstm_units"])
                ),
                layers.Conv2DTranspose(8, 4, strides=4, **output_cnn_params),
                layers.Conv2DTranspose(1, 1, strides=1, **output_cnn_params),
            ],
            name="output_cnn",
        )

    def forward(
        self,
        st_input: st_tensor,
        geo_input: geo_tensor,
        temp_input: temp_tensor,
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
        # This output is cached.
        if self.geo_cnn_output is None:
            output = self.geo_cnn(geo_input)
            self.geo_cnn_output = output[:, tf.newaxis, :, :, :]
        n = st_input.shape[1]
        geo_cnn_output = tf.repeat(self.geo_cnn_output, [n], axis=1)

        # Expand temporal inputs into maps
        # [B, n, m] -> [B, n, H', W', m]
        H_out = st_cnn_output.shape[2]
        W_out = st_cnn_output.shape[3]
        temp_input = temp_input[:, :, tf.newaxis, tf.newaxis, :]
        temp_input = tf.tile(temp_input, [1, 1, H_out, W_out, 1])

        # Concatenate and feed into remaining ConvLSTM and TransposeConv layers
        # [B, n, H', W', k'] -> [B, H, W, 1]
        lstm_input = tf.concat([st_cnn_output, geo_cnn_output, temp_input], axis=-1)
        lstm_output = self.conv_lstm(lstm_input)
        output = self.output_cnn(lstm_output)

        return output

    def call(self, inputs: dict[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
        """Runs the entire autoregressive model.

        Args:
            inputs: A dictionary of input tensors.

        Returns:
            A tuple (all_flood_predictions, max_flood_map) of the flood predictions and
            the max predicted flood depth at each cell over the entire storm duration.
            The maps are the same shape as the input: (H, W).
        """
        st_input, geo_input, full_temp_input = self._validate_and_preprocess_inputs(
            inputs
        )

        # This array stores the initial flood map and the n_predictions.
        # The initial flood map is added to align indexing between flood maps
        # and rainfall, i.e., the current flooding conditions and rainfall at
        # time t are stored at index t along the temporal axis.
        flood_maps = tf.TensorArray(
            tf.float32, size=self.n_predictions + 1, clear_after_read=False
        )
        flood_maps = flood_maps.write(0, st_input)

        # We use 1-indexing for simplicity. Time step t represents the t-th flood
        # prediction.
        for t in tf.range(1, self.n_predictions + 1):
            st_input, temp_input = self._update_temporal_inputs(
                flood_maps, full_temp_input, t
            )
            prediction = self.forward(st_input, geo_input, temp_input)
            flood_maps = flood_maps.write(t, prediction)

        # Concatenate the predictions.
        # This gathers the predictions along axis 0, so we need to permute the
        # time (0) and batch (1) axes.
        predictions = flood_maps.gather(tf.range(1, self.n_predictions + 1))
        predictions = tf.transpose(predictions, perm=[1, 0, 2, 3, 4])
        predictions = tf.squeeze(predictions, axis=-1)
        max_flood = tf.math.reduce_max(predictions, axis=1)

        # Close the TensorArray and clean up the cached geo_cnn_output.
        flood_maps.close()
        self.geo_cnn_output = None

        return (predictions, max_flood)

    def _validate_and_preprocess_inputs(
        self,
        inputs: dict[str, tf.Tensor],
    ) -> tuple[st_tensor, geo_tensor, temp_tensor]:
        """Validates inputs and does all necessary preprocessing.

        Args:
            inputs: A dictionary of input tensors.
                - "geospatial" (required): [H, W, f]
                - "temporal" (required): [T]
                - "spatiotemporal" (optional): a single flood map [H, W]

        Returns:
            A tuple of (spatiotemporal, geospatial, temporal) tensors.
        """
        if "geospatial" not in inputs:
            raise ValueError("Missing required tensor 'geospatial'.")
        if "temporal" not in inputs:
            raise ValueError("Missing required tensor 'temporal'.")

        geo_input = inputs["geospatial"]
        full_temp_input = data_utils.temporal_window_view(
            inputs["temporal"], self.params["m_rainfall"]
        )

        # We assume that, if provided, this input is a *single* flood map.
        st_input = inputs.get("spatiotemporal")
        if st_input is None:
            st_shape = geo_input.shape[:3] + [1]
            st_input = tf.zeros(st_shape)

        return (st_input, geo_input, full_temp_input)

    def _update_temporal_inputs(
        self, flood_maps: tf.TensorArray, full_temp_input: tf.Tensor, t: int
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Updates temporal inputs for a new time step (prediction).

        Returns the appropriate rainfall and flood map inputs for the t-th
        prediction. The number n of flood maps and rainfall windows returned is
        the minimum of self.params["n_flood_maps"] and t.

        Args:
            flood_maps: A TensorArray of all flood maps.
            full_time_input: The [T, m] tensor of all rainfall windows. This
                function retrieves the appropriate windows given the time step.
            t: Time step in the autoregression.

        Returns:
            A tuple of tensors for the flood maps and rainfall, having shapes
            [n, H, W, 1] and [n, m], respectively.
        """
        n = min(self.params["n_flood_maps"], t)
        step_range = tf.range(t - n, t)
        # Gather the relevant flood maps. This will stack them along axis 0, so
        # we need to permute the time (0) and batch (1) axes.
        st_input = flood_maps.gather(step_range)
        st_input = tf.transpose(st_input, perm=[1, 0, 2, 3, 4])
        # Gather the relevant rainfall windows.
        temp_input = tf.gather(full_temp_input, step_range, axis=1)
        return (st_input, temp_input)

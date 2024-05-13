"""Flood model definition."""

from typing import TypeAlias

import tensorflow as tf
from tensorflow.keras import layers

from usl_models.flood_ml import constants
from usl_models.flood_ml import data_utils
from usl_models.flood_ml import model_params

st_tensor: TypeAlias = tf.Tensor
geo_tensor: TypeAlias = tf.Tensor
temp_tensor: TypeAlias = tf.Tensor
FloodModelParams: TypeAlias = model_params.FloodModelParams


class FloodModel:
    """Flood model class."""

    def __init__(
        self,
        model_params: FloodModelParams,
        spatial_dims: tuple[int, int] = (constants.MAP_HEIGHT, constants.MAP_WIDTH),
    ):
        """Creates the flood model.

        Args:
            model_params: A FloodModelParams object of configurable model parameters.
            spatial_dims: Tuple of spatial height and width input dimensions.
                Needed for defining input shapes. This is an optional arg that
                can be changed (primarily for testing/debugging).
        """
        self._model_params = model_params
        self._spatial_dims = spatial_dims
        self._model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        """Creates the correct internal (Keras) model architecture."""
        model = FloodConvLSTM(self._model_params, spatial_dims=self._spatial_dims)
        model.compile(
            optimizer=self._model_params.optimizer,
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(),
                tf.keras.metrics.RootMeanSquaredError(),
            ],
        )
        return model

    def _validate_and_preprocess_inputs(
        self,
        inputs: dict[str, tf.Tensor],
    ) -> dict[str, tf.Tensor]:
        """Validates inputs and does all necessary preprocessing.

        Args:
            inputs: A dictionary of input tensors.
                - "geospatial" (required): [H, W, f]
                - "temporal" (required): [T]
                - "spatiotemporal" (optional): a single flood map [H, W]

        Returns:
            A dictionary of processed input tensors.
        """
        if "geospatial" not in inputs:
            raise ValueError("Missing required tensor 'geospatial'.")
        if "temporal" not in inputs:
            raise ValueError("Missing required tensor 'temporal'.")

        geo_input = inputs["geospatial"]
        full_temp_input = data_utils.temporal_window_view(
            inputs["temporal"], self._model_params.m_rainfall
        )

        # We assume that, if provided, this input is a *single* flood map.
        st_input = inputs.get("spatiotemporal")
        if st_input is None:
            st_shape = geo_input.shape[:3] + [1]
            st_input = tf.zeros(st_shape)

        return {
            "geospatial": geo_input,
            "temporal": full_temp_input,
            "spatiotemporal": st_input,
        }

    def train(
        self,
        train_data: dict[str, tf.Tensor],
        train_labels: tf.Tensor,
        n_predictions=1,
    ) -> tf.keras.callbacks.History:
        """Trains the model.

        Args:
            train_data: A dictionary of training inputs.
            train_labels: A tensor of labels on which to fit the model.
            n_predictions: The number of predictions to make, synonymous with
                the storm duration.

        Returns:
            A History object containing the record of training and, if applicable,
            validation loss and metrics.
        """
        train_inputs = self._validate_and_preprocess_inputs(train_data)
        self._model.set_n_predictions(n_predictions)
        history = self._model.fit(
            train_inputs,
            train_labels,
            batch_size=self._model_params.batch_size,
            epochs=self._model_params.epochs,
            validation_split=0.2,
        )
        return history


###############################################################################
#                       Custom Keras Model definitions
#
# The following are keras.Model class implementations for flood model
# architecture(s). They can be used within the generic FloodModel class above
# for training and evaluation, and only define basic components of the model,
# such as layers and the forward pass. While these models are callable, all
# data pre- and post-processing are expected to be handled externally.
###############################################################################


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
        params: FloodModelParams,
        n_predictions: int = 1,
        spatial_dims: tuple[int, int] = (constants.MAP_HEIGHT, constants.MAP_WIDTH),
    ):
        """Creates the ConvLSTM model.

        Args:
            params: A FloodModelParams object of configurable model parameters.
            n_predictions: The number of predictions to make; storm duration.
            spatial_dims: Tuple of spatial height and width input dimensions.
                Needed for defining input shapes. This is an optional arg that
                can be changed (primarily for testing/debugging).
        """
        super().__init__()

        self._params = params
        self._n_predictions = n_predictions
        self._spatial_height, self._spatial_width = spatial_dims

        # Spatiotemporal CNN
        st_cnn_params = {"strides": 2, "padding": "same", "activation": "relu"}
        self.st_cnn = tf.keras.Sequential(
            [
                # Input shape: (time_steps, height, width, channels)
                layers.InputLayer((None, self._spatial_height, self._spatial_width, 1)),
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
                    (self._spatial_height, self._spatial_width, constants.GEO_FEATURES)
                ),
                layers.Conv2D(16, 5, **geo_cnn_params),
                layers.MaxPool2D(pool_size=2, strides=1, padding="same"),
                layers.Conv2D(64, 5, **geo_cnn_params),
                layers.MaxPool2D(pool_size=2, strides=1, padding="same"),
            ],
            name="geospatial_cnn",
        )

        # ConvLSTM
        # The spatial dimensions have been reduced 4x by the CNNs.
        # The "channel" dimension is the sum of the channels from the CNNs
        # and the rainfall window size.
        conv_lstm_height = self._spatial_height // 4
        conv_lstm_width = self._spatial_width // 4
        conv_lstm_channels = 16 + 64 + self._params.m_rainfall
        self.conv_lstm = tf.keras.Sequential(
            [
                # Input shape: (time_steps, height, width, channels)
                layers.InputLayer(
                    (None, conv_lstm_height, conv_lstm_width, conv_lstm_channels)
                ),
                layers.ConvLSTM2D(
                    self._params.lstm_units,
                    self._params.lstm_kernel_size,
                    strides=1,
                    padding="same",
                    activation="tanh",
                    dropout=self._params.lstm_dropout,
                    recurrent_dropout=self._params.lstm_recurrent_dropout,
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
                    (conv_lstm_height, conv_lstm_width, self._params.lstm_units)
                ),
                layers.Conv2DTranspose(8, 4, strides=4, **output_cnn_params),
                layers.Conv2DTranspose(1, 1, strides=1, **output_cnn_params),
            ],
            name="output_cnn",
        )

    def set_n_predictions(self, n_predictions: int) -> None:
        """Updates n_predictions."""
        self._n_predictions = n_predictions

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
        geo_cnn_output = self.geo_cnn(geo_input)
        geo_cnn_output = geo_cnn_output[:, tf.newaxis, :, :, :]
        n = st_input.shape[1]
        geo_cnn_output = tf.repeat(geo_cnn_output, [n], axis=1)

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

    def call(self, inputs: dict[str, tf.Tensor]) -> tf.Tensor:
        """Runs the entire autoregressive model.

        Args:
            inputs: A dictionary of input tensors.

        Returns:
            A tensor of all the flood predictions: [B, T, H, W].
        """
        st_input = inputs["spatiotemporal"]
        geo_input = inputs["geospatial"]
        full_temp_input = inputs["temporal"]

        # This array stores the initial flood map and the n_predictions.
        # The initial flood map is added to align indexing between flood maps
        # and rainfall, i.e., the current flooding conditions and rainfall at
        # time t are stored at index t along the temporal axis.
        flood_maps = tf.TensorArray(
            tf.float32, size=self._n_predictions + 1, clear_after_read=False
        )
        flood_maps = flood_maps.write(0, st_input)

        # We use 1-indexing for simplicity. Time step t represents the t-th flood
        # prediction.
        for t in range(1, self._n_predictions + 1):
            st_input, temp_input = self._update_temporal_inputs(
                flood_maps, full_temp_input, t
            )
            prediction = self.forward(st_input, geo_input, temp_input)
            flood_maps = flood_maps.write(t, prediction)

        # Concatenate the predictions.
        # This gathers the predictions along axis 0, so we need to permute the
        # time (0) and batch (1) axes.
        predictions = flood_maps.gather(tf.range(1, self._n_predictions + 1))
        predictions = tf.transpose(predictions, perm=[1, 0, 2, 3, 4])
        predictions = tf.squeeze(predictions, axis=-1)

        # Close the TensorArray and clean up the cached geo_cnn_output.
        flood_maps.close()
        self.geo_cnn_output = None

        return predictions

    def _update_temporal_inputs(
        self, flood_maps: tf.TensorArray, full_temp_input: tf.Tensor, t: int
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Updates temporal inputs for a new time step (prediction).

        Returns the appropriate rainfall and flood map inputs for the t-th
        prediction. The number n of flood maps and rainfall windows returned is
        the minimum of self._params.n_flood_maps and t.

        Args:
            flood_maps: A TensorArray of all flood maps.
            full_time_input: The [T, m] tensor of all rainfall windows. This
                function retrieves the appropriate windows given the time step.
            t: Time step in the autoregression.

        Returns:
            A tuple of tensors for the flood maps and rainfall, having shapes
            [n, H, W, 1] and [n, m], respectively.
        """
        n = min(self._params.n_flood_maps, t)
        step_range = tf.range(t - n, t)
        # Gather the relevant flood maps. This will stack them along axis 0, so
        # we need to permute the time (0) and batch (1) axes.
        st_input = flood_maps.gather(step_range)
        st_input = tf.transpose(st_input, perm=[1, 0, 2, 3, 4])
        # Gather the relevant rainfall windows.
        temp_input = tf.gather(full_temp_input, step_range, axis=1)
        return (st_input, temp_input)

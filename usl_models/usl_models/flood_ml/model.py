"""Flood model definition."""

import dataclasses
import logging
from typing import Optional, TypeAlias

import tensorflow as tf
from tensorflow.keras import layers

from usl_models.flood_ml import constants
from usl_models.flood_ml import data_utils
from usl_models.flood_ml import model_params

st_tensor: TypeAlias = tf.Tensor
geo_tensor: TypeAlias = tf.Tensor
temp_tensor: TypeAlias = tf.Tensor
FloodModelData: TypeAlias = data_utils.FloodModelData
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

    def _validate_and_preprocess_data(
        self,
        data: FloodModelData,
        training=True,
    ) -> FloodModelData:
        """Validates model data and does all necessary preprocessing.

        Args:
            data: A FloodModelData object. Labels are required for training.
            training: Whether data is used for model training. If True, labels
                will be validated.

        Returns:
            A processed FloodModelData object.
        """
        if training:
            # Labels are required for training.
            if data.labels is None:
                raise ValueError("Labels must be provided during model training.")

            # Labels must match the storm duration.
            expected_label_shape = list(data.labels.shape)
            expected_label_shape[1] = data.storm_duration
            assert data.labels.shape[1] == data.storm_duration, (
                "Provided labels are inconsistent with storm duration. "
                f"Labels are expected to have shape {expected_label_shape}. "
                f"Actual shape: {data.labels.shape}."
            )

        # Check whether the temporal data is already windowed. If it is, checks
        # the expected shape. Otherwise, create the window view.
        if tf.rank(data.temporal) == 3:  # windowed: (B, T_max, m)
            assert data.temporal.shape[-1] == self._model_params.m_rainfall, (
                "Mismatch between the temporal data window size "
                f"({data.temporal.shape[-1]}) and the expected window size "
                f"(m = {self._model_params.m_rainfall})."
            )
        else:
            full_temp_input = data_utils.temporal_window_view(
                data.temporal, self._model_params.m_rainfall
            )
            data = dataclasses.replace(data, temporal=full_temp_input)

        # We assume that, if provided, this input is a *single* flood map.
        st_input = data.spatiotemporal
        if st_input is None:
            st_shape = data.geospatial.shape[:3] + [1]
            st_input = tf.zeros(st_shape)

        data = dataclasses.replace(data, spatiotemporal=st_input)
        return data

    def _model_fit(
        self, data: FloodModelData, early_stopping: Optional[int]
    ) -> tf.keras.callbacks.History:
        """Fits the model on a single batch of FloodModelData.

        Args:
            data: A FloodModelData object.
            early_stopping: Optional integer representing the number of epochs to assign
                as the patience when enabling early stopping. See the train method for
                more details.

        Returns: A History object containing the training and validation loss
            and metrics.
        """
        inputs = {
            "spatiotemporal": data.spatiotemporal,
            "geospatial": data.geospatial,
            "temporal": data.temporal,
        }
        self._model.set_n_predictions(data.storm_duration)

        callbacks = None
        if early_stopping:
            es_callback = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", mode="min", patience=early_stopping
            )
            callbacks = [es_callback]

        history = self._model.fit(
            inputs,
            data.labels,
            batch_size=self._model_params.batch_size,
            epochs=self._model_params.epochs,
            validation_split=0.2,
            callbacks=callbacks,
        )
        return history

    def train(
        self,
        data: list[FloodModelData],
        early_stopping: Optional[int] = None,
    ) -> list[tf.keras.callbacks.History]:
        """Trains the model.

        The internal flood model architecture is restricted to a single storm
        duration for each training run. In order to train on varying storm durations,
        the model must be trained incrementally via several `fit` calls.
        This function provides a wrapper around the incremental training logic,
        allowing the user to pass in data for multiple storm durations at once
        via a list of FloodModelData objects.

        Each FloodModelData instance is associated with a single storm duration.
        In other words, data for different storm durations must be passed in as
        separate FloodModelData objects.

        Args:
            data: A list of FloodModelData objects. Labels are required for training.
            early_stopping: Optional integer representing the number of epochs to assign
                as the patience when enabling early stopping. The model will stop
                training if the validation loss doesn't improve after that many
                epochs, up until the total number of epochs set for training.

        Returns: A list of History objects containing the record of training and,
            if applicable, validation loss and metrics.
        """
        model_history = []
        processed = [self._validate_and_preprocess_data(x, training=True) for x in data]

        for x in processed:
            history = self._model_fit(x, early_stopping=early_stopping)
            model_history.append(history)

        return model_history

    def load_model(self, filepath: str) -> None:
        """Loads weights from an existing file.

        Args:
            filepath: Path to the weights file to load into the current model.
        """
        self._model.load_weights(filepath)
        logging.info("Loaded model weights from %s", filepath)

    def save_model(self, filepath: str, overwrite: bool = False) -> None:
        """Saves a .keras model to the specified path.

        Args:
            filepath: Path to which to save the model. Must end in ".keras".
            overwrite: If True and the specified filepath already exists, the model
                will overwrite the existing file. Otherwise, the user will be prompted
                on whether to overwrite any existing model.
        """
        self._model.save(filepath, overwrite=overwrite)
        logging.info("Saved model to %s", filepath)


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

"""Flood model definition."""

import dataclasses
import logging
from typing import Iterator, TypeAlias, TypedDict, List, Callable

import tensorflow as tf
import keras
from keras import layers

from usl_models.flood_ml import constants
from usl_models.flood_ml import data_utils
from usl_models.flood_ml import model_params

st_tensor: TypeAlias = tf.Tensor
geo_tensor: TypeAlias = tf.Tensor
temp_tensor: TypeAlias = tf.Tensor
FloodModelData: TypeAlias = data_utils.FloodModelData
FloodModelParams: TypeAlias = model_params.FloodModelParams


class Input(TypedDict):
    geospatial: tf.Tensor
    temporal: tf.Tensor
    spatiotemporal: tf.Tensor


class FloodModel:
    """Flood model class."""

    class Input(TypedDict):
        """Input tensors dictionary."""

        geospatial: tf.Tensor
        temporal: tf.Tensor
        spatiotemporal: tf.Tensor

    class Result(TypedDict):
        """Prediction result dictionary."""

        prediction: tf.Tensor
        chunk_id: str

    def __init__(
        self,
        model_params: FloodModelParams,
        spatial_dims: tuple[int, int] = (constants.MAP_HEIGHT, constants.MAP_WIDTH),
    ):
        """Creates the flood model.

        Args:
            model_params: A dictionary of configurable model parameters.
            spatial_dims: Tuple of spatial height and width input dimensions.
                Needed for defining input shapes. This is an optional arg that
                can be changed (primarily for testing/debugging).
        """
        self._model_params = model_params
        self._spatial_dims = spatial_dims
        self._model = self._build_model()

    def _build_model(self) -> keras.Model:
        """Creates the correct internal (Keras) model architecture."""
        model = FloodConvLSTM(
            self._model_params,
            spatial_dims=self._spatial_dims,
        )
        model.compile(
            optimizer=tf.keras.optimizers.get(self._model_params["optimizer_config"]),
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
            assert data.temporal.shape[-1] == self._model_params["m_rainfall"], (
                "Mismatch between the temporal data window size "
                f"({data.temporal.shape[-1]}) and the expected window size "
                f"(m = {self._model_params['m_rainfall']})."
            )
        else:
            full_temp_input = data_utils.temporal_window_view(
                data.temporal, self._model_params["m_rainfall"]
            )
            data = dataclasses.replace(data, temporal=full_temp_input)

        # We assume that, if provided, this input is a *single* flood map.
        st_input = data.spatiotemporal
        if st_input is None:
            st_shape = data.geospatial.shape[:3] + [1]
            st_input = tf.zeros(st_shape)

        data = dataclasses.replace(data, spatiotemporal=st_input)
        return data

    def call(self, input: Input) -> tf.Tensor:
        """Predict the next timestep. See `FloodConvLSTM.call`."""
        return self._model.call(input)

    def call_n(self, full_input: Input, n: int = 1) -> tf.Tensor:
        """Predict the next n timesteps. See `FloodConvLSTM.call_n`."""
        return self._model.call_n(full_input, n=n)

    def batch_predict_n(
        self, dataset: tf.data.Dataset, n: int = 1
    ) -> Iterator[list[Result]]:
        """Runs batch prediction (call_n).

        The strategy should be the same as the one used to initialize the model.

        Example usage:
        ```py
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
          model = FloodModel(artifact_uri="gs://path/to/model")
          for results in model.batch_predict_n(strategy, dataset, n=4):
            for result in results:
              print(result)
        ```

        Args:
            strategy: multi-GPU distribute strategy.
            dataset: Dataset generating (inputs, metadata) tuples.
            n: number of timesteps to predict.

        Returns: an iterator containing batches of results.
        """
        dataset = strategy.experimental_distribute_dataset(dataset)
        strategy = tf.distribute.get_strategy()

        @tf.function(reduce_retracing=True)
        def predict(inputs: FloodModel.Input, n: int):
            prediction = self.call_n(inputs, n=n)
            return tf.reduce_max(prediction, axis=1)

        for inputs, metadata in dataset:
            batch_predictions = strategy.run(predict, [inputs, n])
            results = []
            if strategy.num_replicas_in_sync > 1:
                flat_batches = []
                for batch_replicas in batch_predictions.values:
                    for batch in batch_replicas:
                        flat_batches.append(batch)
                for prediction, metadata in zip(flat_batches, metadata):
                    results.append(
                        self.Result(
                            prediction=prediction, chunk_id=metadata["chunk_id"]
                        )
                    )
            else:
                for prediction, metadata in zip(batch_predictions, metadata):
                    results.append(
                        self.Result(
                            prediction=prediction, chunk_id=metadata["chunk_id"]
                        )
                    )
            yield results

    def fit(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset | None = None,
        epochs: int = 1,
        steps_per_epoch: int | None = None,
        early_stopping: int | None = None,
        callbacks: List[Callable] | None = None,
    ):
        """Fit the model to the given dataset."""
        if callbacks is None:
            callbacks = []
        if early_stopping is not None:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="loss", mode="min", patience=early_stopping
                )
            )

        # Fit the model for this sample
        return self._model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
        )

    def load_model(self, filepath: str) -> None:
        """Loads weights from an existing file.

        Args:
            filepath: Path to the weights file to load into the current model.
        """
        self._model.load_weights(filepath)
        logging.info("Loaded model weights from %s", filepath)

    def save_model(self, filepath: str, **kwargs) -> None:
        """Saves a .keras model to the specified path.

        Args:
            filepath: Path to which to save the model. Must end in ".keras".
            kwargs: Additional arguments to pass to keras' model.save method.
        """
        self._model.save(filepath, **kwargs)
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
        spatial_dims: tuple[int, int] = (constants.MAP_HEIGHT, constants.MAP_WIDTH),
    ):
        """Creates the ConvLSTM model.

        Args:
            params: A dictionary of configurable model parameters.
            spatial_dims: Tuple of spatial height and width input dimensions.
                Needed for defining input shapes. This is an optional arg that
                can be changed (primarily for testing/debugging).
        """
        super().__init__()

        self._params = params
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
        conv_lstm_channels = 16 + 64 + self._params["m_rainfall"]
        self.conv_lstm = tf.keras.Sequential(
            [
                # Input shape: (time_steps, height, width, channels)
                layers.InputLayer(
                    (None, conv_lstm_height, conv_lstm_width, conv_lstm_channels)
                ),
                layers.ConvLSTM2D(
                    self._params["lstm_units"],
                    self._params["lstm_kernel_size"],
                    strides=1,
                    padding="same",
                    activation="tanh",
                    dropout=self._params["lstm_dropout"],
                    recurrent_dropout=self._params["lstm_recurrent_dropout"],
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
                    (conv_lstm_height, conv_lstm_width, self._params["lstm_units"])
                ),
                layers.Conv2DTranspose(8, 4, strides=4, **output_cnn_params),
                layers.Conv2DTranspose(1, 1, strides=1, **output_cnn_params),
            ],
            name="output_cnn",
        )

    def call(self, input: FloodModel.Input) -> tf.Tensor:
        """Makes a single forward pass on a batch of data.

        The forward pass represents a single prediction on an input batch
        (i.e., a single flood map). This functions implements the logic of the
        internal ConvLSTM and ignores autoregressive steps.

        Args:
            input: Dictionary containing:
              - spatiotemporal: Flood maps tensor of shape [B, n, H, W, 1].
              - geospatial: Geospatial tensor of shape [B, H, W, f].
              - temporal: Rainfall windows tensor of shape [B, n, m].

        Returns:
            The flood map prediction. A tensor of shape [B, H, W, 1].
        """
        spatiotemporal: tf.Tensor = input["spatiotemporal"]
        geospatial: tf.Tensor = input["geospatial"]
        temporal: tf.Tensor = input["temporal"]

        B = spatiotemporal.shape[0]
        C = 1  # Channel dimension for spatiotemporal tensor
        F = constants.GEO_FEATURES
        N = self._params["n_flood_maps"]
        M = self._params["m_rainfall"]
        H, W = self._spatial_height, self._spatial_width

        tf.ensure_shape(spatiotemporal, (B, N, H, W, C))
        tf.ensure_shape(geospatial, (B, H, W, F))
        tf.ensure_shape(temporal, (B, N, M))

        # Spatiotemporal CNN
        # [B, n, H, W, 1] -> [B, n, H', W', k1]
        st_cnn_output = self.st_cnn(spatiotemporal)

        # Geospatial CNN
        # [B, H, W, f ]-> [B, H', W', k2]
        # Add a new time axis and repeat n times -> [B, n, H', W', k2].
        geo_cnn_output = self.geo_cnn(geospatial)
        geo_cnn_output = geo_cnn_output[:, tf.newaxis, :, :, :]
        n = spatiotemporal.shape[1]
        geo_cnn_output = tf.repeat(geo_cnn_output, [n], axis=1)

        # Expand temporal inputs into maps
        # [B, n, m] -> [B, n, H', W', m]
        H_out = st_cnn_output.shape[2]
        W_out = st_cnn_output.shape[3]
        temp_input = temporal[:, :, tf.newaxis, tf.newaxis, :]
        temp_input = tf.tile(temp_input, [1, 1, H_out, W_out, 1])

        # Concatenate and feed into remaining ConvLSTM and TransposeConv layers
        # [B, n, H', W', k'] -> [B, H, W, 1]
        lstm_input = tf.concat([st_cnn_output, geo_cnn_output, temp_input], axis=-1)
        lstm_output = self.conv_lstm(lstm_input)
        output = self.output_cnn(lstm_output)

        return output

    def call_n(self, full_input: FloodModel.Input, n: int = 1) -> tf.Tensor:
        """Runs the entire autoregressive model.

        Args:
            full_input: A dictionary of input tensors.
                While `call` expects only input data for a single context window,
                `call_n` requires the full temporal tensor.
            n: Number of autoregressive iterations to run.

        Returns:
            A tensor of all the flood predictions: [B, n, H, W].
        """
        spatiotemporal = full_input["spatiotemporal"]
        geospatial = full_input["geospatial"]
        temporal = full_input["temporal"]

        B = spatiotemporal.shape[0]
        C = 1  # Channel dimension for spatiotemporal tensor
        F = constants.GEO_FEATURES
        N, M = self._params["n_flood_maps"], self._params["m_rainfall"]
        T_MAX = constants.MAX_RAINFALL_DURATION
        H, W = self._spatial_height, self._spatial_width

        tf.ensure_shape(spatiotemporal, (B, N, H, W, C))
        tf.ensure_shape(geospatial, (B, H, W, F))
        tf.ensure_shape(temporal, (B, T_MAX, M))

        # This array stores the n predictions.
        predictions = tf.TensorArray(tf.float32, size=n)

        # We use 1-indexing for simplicity. Time step t represents the t-th flood
        # prediction.
        for t in range(1, n + 1):
            input = FloodModel.Input(
                geospatial=geospatial,
                temporal=self._get_temporal_window(temporal, t, N),
                spatiotemporal=spatiotemporal,
            )
            prediction = self.call(input)
            predictions = predictions.write(t - 1, prediction)

            # Append new predictions along time axis, drop the first.
            spatiotemporal = tf.concat(
                [spatiotemporal, tf.expand_dims(prediction, axis=1)], axis=1
            )[:, 1:]

        # Gather dense tensor out of TensorArray along the time axis.
        predictions = tf.stack(tf.unstack(predictions.stack()), axis=1)
        # Drop channels dimension.
        return tf.squeeze(predictions, axis=-1)

    @staticmethod
    def _get_temporal_window(temporal: tf.Tensor, t: int, n: int) -> tf.Tensor:
        """Returns a zero-padded n-sized window at timestep t.

        Args:
            temporal: Temporal tensor of shape (B, T_MAX, M)
            t: timestep to fetch the windows for. At time t, we use at temporal[t-n:t].
            n: window size

        Returns:
            Returns a zero-padded n-sized window at timestep t of shape (B, n, M)
        """
        B, _, M = temporal.shape
        return tf.concat(
            [
                tf.zeros(shape=(B, tf.maximum(n - t, 0), M)),
                temporal[:, tf.maximum(t - n, 0) : t],
            ],
            axis=1,
        )

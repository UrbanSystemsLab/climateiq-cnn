import logging
from typing import Iterator, TypedDict, List, Callable, Literal, TypeAlias, Any
import dataclasses
import numpy as np
from usl_models.shared import pad_layers
import keras
from keras import layers
import keras_tuner
import tensorflow as tf
from typing import Self
from usl_models.flood_ml import constants
from usl_models.shared import keras_dataclasses
@keras.saving.register_keras_serializable()

def weighted_mse_small_targets(y_true, y_pred):
    """Custom MSE loss that gives higher importance to smaller target values."""
    # Ignore NaNs (if any) in labels
    mask = tf.math.logical_not(tf.math.is_nan(y_true))
    y_true = tf.where(mask, y_true, tf.zeros_like(y_true))
    y_pred = tf.where(mask, y_pred, tf.zeros_like(y_pred))

    # Define inverse-weighting: smaller values get higher weight
    weights = 1.0 / (1.0 + 100.0 * y_true)  # e.g., 0.001 -> ~1, 0.1 -> ~0.09
    squared_error = tf.square(y_true - y_pred)
    weighted_squared_error = weights * squared_error

    # Normalize by number of valid entries
    return tf.reduce_sum(weighted_squared_error) / tf.reduce_sum(
        tf.cast(mask, tf.float32)
    )


@keras.saving.register_keras_serializable()
def hybrid_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    small_weighted = weighted_mse_small_targets(y_true, y_pred)
    return 0.5 * mse + 0.5 * small_weighted


Activation: TypeAlias = Literal["relu", "sigmoid", "tanh", "softmax", "linear"]
PadMode: TypeAlias = Literal["REFLECT", "CONSTANT"]


class SpatialAttention(layers.Layer):
    def __init__(self, **kwargs):
        """Initialize the spatial attention instance."""
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(
            1, kernel_size=7, padding="same", activation="sigmoid"
        )

    def call(self, inputs):
        """Compute the attention weights."""
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)
        return inputs * attention


class FloodModel:
    """Flood model class."""

    @keras_dataclasses.dataclass(kw_only=True)
    class Params(keras_dataclasses.Base):
        """Flood model hyperparameters."""

        lstm_units: int = 64
        lstm_kernel_size: int = 3
        lstm_dropout: float = 0.2
        lstm_recurrent_dropout: float = 0.2
        m_rainfall: int = 3
        n_flood_maps: int = 6
        num_features: int = 22
        pad_mode: PadMode = "REFLECT"
        optimizer: keras.optimizers.Optimizer = dataclasses.field(
            default_factory=lambda: keras.optimizers.Adam(learning_rate=1e-3)
        )

        def to_dict(self) -> dict[str, Any]:
            """Convert Params instance to dictionary."""
            return {
                "lstm_units": self.lstm_units,
                "lstm_kernel_size": self.lstm_kernel_size,
                "lstm_dropout": self.lstm_dropout,
                "lstm_recurrent_dropout": self.lstm_recurrent_dropout,
                "m_rainfall": self.m_rainfall,
                "n_flood_maps": self.n_flood_maps,
                "num_features": self.num_features,
                "optimizer": {
                    "class_name": type(self.optimizer).__name__,
                    "config": {
                        k: (float(v) if isinstance(v, (float, np.floating)) else v)
                        for k, v in self.optimizer.get_config().items()
                    },
                },
            }

        @classmethod
        def from_dict(cls, d: dict[str, Any]) -> "FloodModel.Params":
            """Create Params instance from dictionary."""
            d = d.copy()
            optimizer_info = d.pop("optimizer")
            optimizer = keras.optimizers.get(
                {
                    "class_name": optimizer_info["class_name"],
                    "config": optimizer_info["config"],
                }
            )
            return cls(optimizer=optimizer, **d)

    class Input(TypedDict):
        """Input tensors dictionary."""

        geospatial: tf.Tensor
        temporal: tf.Tensor
        spatiotemporal: tf.Tensor

    class Result(TypedDict):
        """Prediction result dictionary."""

        prediction: tf.Tensor
        chunk_id: str | tf.Tensor

    def __init__(
        self, params: Params | None = None, spatial_dims: tuple[int, int] | None = None
    ):
        """Initialize the FloodModel instance."""
        self._params = params or self.Params()
        self._spatial_dims = spatial_dims or (constants.MAP_HEIGHT, constants.MAP_WIDTH)
        self._model = self._build_model()

    @classmethod
    def from_checkpoint(cls, artifact_uri: str, **kwargs) -> "FloodModel":
        """Loads the model from a checkpoint URI.

        We load weights only to keep custom methods (e.g. `call_n`) intact.
        This only works if the model architecture is identical to the architecure
        used during export.
        Ideally, we would load the entire Keras model and use that directly to allow
        loading different architectures within the same wrapper class.
        Unfortunately, `call_n` is not trivially serializeable in its current state.

        Args:
            artifact_uri: The path to the SavedModel directory.
                This should end in `/model` if using a GCloud artifact.

        Returns:
            The loaded FloodModel.
        """
        loaded_model = keras.models.load_model(artifact_uri)
        params = FloodModel.Params.from_config(loaded_model.get_config())
        model = cls(params=params, **kwargs)
        model._model.set_weights(loaded_model.get_weights())
        return model

    @classmethod
    def get_hypermodel(cls, **kwargs) -> keras_tuner.HyperModel:
        """Returns a hypermodel function for use with Keras Tuner.

        The kwargs (hp_options) should be a dict where keys are parameter names
        and values are lists of possible choices.
        """

        def hypermodel(hp: keras_tuner.HyperParameters):
            hp_kwargs = {k: hp.Choice(k, v) for k, v in kwargs.items()}
            params = cls.Params(**hp_kwargs)
            return cls(params=params)._model

        return hypermodel

    def _build_model(self) -> keras.Model:
        model = FloodConvLSTM(self._params, spatial_dims=self._spatial_dims)
        model.compile(
            optimizer=self._params.optimizer,
            # loss=keras.losses.MeanSquaredError(),
            loss=hybrid_loss,
            metrics=[
                keras.metrics.MeanAbsoluteError(),
                keras.metrics.RootMeanSquaredError(),
            ],
        )
        return model

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
          model = FloodModel.from_checkpoint(artifact_uri="gs://path/to/model")
          for results in model.batch_predict_n(dataset, n=4):
            for result in results:
              print(result)
        ```

        Args:
            strategy: multi-GPU distribute strategy.
            dataset: Dataset generating (inputs, metadata) tuples.
            n: number of timesteps to predict.

        Returns: an iterator containing batches of results.
        """
        strategy = tf.distribute.get_strategy()
        dataset = strategy.experimental_distribute_dataset(dataset)

        @tf.function(reduce_retracing=True)
        def predict(inputs: FloodModel.Input, n: int):
            prediction = self.call_n(inputs, n=n)
            return tf.reduce_max(prediction, axis=1)

        for inputs, metadata in dataset:
            batch_predictions = strategy.run(predict, [inputs, n])

            # For multi-gpu, flatten per-replica batches
            if strategy.num_replicas_in_sync > 1:
                replica_batches = batch_predictions.values
                batch_predictions = []
                for batches in replica_batches:
                    for batch in batches:
                        batch_predictions.append(batch)

            results = []
            # Predictions are returned in the same order as the inputs,
            # which is a parallel array w.r.t. metadata.
            # https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy
            for prediction, chunk_id in zip(
                batch_predictions, metadata["feature_chunk"]
            ):
                results.append(self.Result(prediction=prediction, chunk_id=chunk_id))
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

    def load_weights(self, filepath: str) -> None:
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


class FloodConvLSTM(keras.Model):
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
        params: FloodModel.Params,
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

        K_PAD = 2
        cnn_pad = (K_PAD, K_PAD)
        activation = "relu"

        self.st_cnn = keras.Sequential(
            [
                layers.InputLayer((None, self._spatial_height, self._spatial_width, 1)),
                layers.TimeDistributed(pad_layers.Pad2D(cnn_pad, mode="REFLECT")),
                layers.TimeDistributed(
                    layers.Conv2D(
                        8, 5, strides=2, padding="valid", activation=activation
                    )
                ),
                layers.TimeDistributed(
                    layers.MaxPool2D(pool_size=2, strides=1, padding="same")
                ),
                layers.TimeDistributed(pad_layers.Pad2D(cnn_pad, mode="REFLECT")),
                layers.TimeDistributed(
                    layers.Conv2D(
                        16, 5, strides=2, padding="valid", activation=activation
                    )
                ),
                layers.TimeDistributed(
                    layers.MaxPool2D(pool_size=2, strides=1, padding="same")
                ),
            ]
        )

        self.geo_cnn = keras.Sequential(
            [
                layers.InputLayer(
                    (self._spatial_height, self._spatial_width, constants.GEO_FEATURES)
                ),
                pad_layers.Pad2D(cnn_pad, mode="REFLECT"),
                layers.Conv2D(16, 5, strides=2, padding="valid", activation=activation),
                layers.MaxPool2D(pool_size=2, strides=1, padding="same"),
                pad_layers.Pad2D(cnn_pad, mode="REFLECT"),
                layers.Conv2D(64, 5, strides=2, padding="valid", activation=activation),
                layers.MaxPool2D(pool_size=2, strides=1, padding="same"),
            ]
        )

        conv_lstm_height = self._spatial_height // 4
        conv_lstm_width = self._spatial_width // 4
        conv_lstm_channels = 16 + 64 + self._params.m_rainfall

        self.pre_attention = SpatialAttention()  # attention before ConvLSTM

        self.conv_lstm = keras.Sequential(
            [
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
            ]
        )

        self.attention = SpatialAttention()  # attention after ConvLSTM

        self.output_cnn = keras.Sequential(
            [
                layers.InputLayer(
                    (conv_lstm_height, conv_lstm_width, self._params.lstm_units)
                ),
                layers.Conv2DTranspose(
                    8, 4, strides=4, padding="same", activation="relu"
                ),
                layers.Conv2DTranspose(
                    1, 1, strides=1, padding="same", activation="relu"
                ),
            ]
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
        spatiotemporal = input["spatiotemporal"]
        geospatial = input["geospatial"]
        temporal = input["temporal"]

        B, N, H, W, _ = spatiotemporal.shape

        st_cnn_output = self.st_cnn(spatiotemporal)
        geo_cnn_output = self.geo_cnn(geospatial)
        geo_cnn_output = geo_cnn_output[:, tf.newaxis, :, :, :]
        geo_cnn_output = tf.repeat(geo_cnn_output, [N], axis=1)

        H_out, W_out = st_cnn_output.shape[2], st_cnn_output.shape[3]
        temp_input = temporal[:, :, tf.newaxis, tf.newaxis, :]
        temp_input = tf.tile(temp_input, [1, 1, H_out, W_out, 1])

        lstm_input = tf.concat([st_cnn_output, geo_cnn_output, temp_input], axis=-1)
        lstm_input = self.pre_attention(lstm_input)
        lstm_output = self.conv_lstm(lstm_input)
        lstm_output = self.attention(lstm_output)
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
        C = 1
        F = constants.GEO_FEATURES
        N, M = self._params.n_flood_maps, self._params.m_rainfall
        T_MAX = constants.MAX_RAINFALL_DURATION
        H, W = self._spatial_height, self._spatial_width

        tf.ensure_shape(spatiotemporal, (B, N, H, W, C))
        tf.ensure_shape(geospatial, (B, H, W, F))
        tf.ensure_shape(temporal, (B, T_MAX, M))

        predictions = tf.TensorArray(tf.float32, size=n)
        for t in range(1, n + 1):
            input = FloodModel.Input(
                geospatial=geospatial,
                temporal=self._get_temporal_window(temporal, t, N),
                spatiotemporal=spatiotemporal,
            )
            prediction = self.call(input)
            predictions = predictions.write(t - 1, prediction)
            spatiotemporal = tf.concat(
                [spatiotemporal, tf.expand_dims(prediction, axis=1)], axis=1
            )[:, 1:]

        predictions = tf.stack(tf.unstack(predictions.stack()), axis=1)
        return tf.squeeze(predictions, axis=-1)

    def get_config(self) -> dict:
        """Keras serialization."""
        return self._params.get_config()
    
    @classmethod
    def from_config(cls, config: dict) -> Self:
        """Keras deserialization."""
        return cls(params=FloodModel.Params.from_config(config))
    
    @staticmethod
    def _get_temporal_window(temporal: tf.Tensor, t: int, n: int) -> tf.Tensor:
        B, _, M = temporal.shape
        return tf.concat(
            [
                tf.zeros(shape=(B, tf.maximum(n - t, 0), M)),
                temporal[:, tf.maximum(t - n, 0) : t],
            ],
            axis=1,
        )

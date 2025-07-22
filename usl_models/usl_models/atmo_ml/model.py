"""AtmoML model definition."""

import logging
import dataclasses
from typing import TypedDict, TypeAlias, List, Callable, Literal, Tuple, Iterable

import keras
from keras import layers
import keras_tuner
import tensorflow as tf
import numpy as np

from usl_models.atmo_ml import data_utils
from usl_models.atmo_ml import constants
from usl_models.atmo_ml import metrics
from usl_models.atmo_ml import vars

from usl_models.shared import keras_dataclasses
from usl_models.shared import pad_layers


Activation: TypeAlias = Literal["relu", "sigmoid", "tanh", "softmax", "linear"]


PadMode: TypeAlias = Literal["REFLECT", "CONSTANT"]


class ConvParams(TypedDict):
    """Conv layer parameters."""

    activation: Activation
    padding: Literal["valid", "same"]


class AtmoModel:
    """Atmospheric model class."""

    @keras_dataclasses.dataclass(kw_only=True)
    class Params(keras_dataclasses.Base):
        """Model parameters."""

        # Input CNN params
        input_cnn_kernel_size: int = 5

        # Output CNN Params
        output_cnn_kernel_size: int = 1

        # LSTM parameters.
        lstm_units: int = 64
        lstm_kernel_size: int = 5
        lstm_dropout: float = 0.2
        lstm_recurrent_dropout: float = 0.2

        # New activation parameters for each block.
        spatial_activation: Activation = "relu"
        st_activation: Activation = "relu"
        lstm_activation: Activation = "tanh"
        output_activation: Activation = "relu"

        # The optimizer configuration.
        optimizer: keras.optimizers.Optimizer = dataclasses.field(
            default_factory=lambda: keras.optimizers.Adam(learning_rate=1e-3)
        )

        output_timesteps: int = constants.OUTPUT_TIME_STEPS
        conv1_stride: int = 1
        conv2_stride: int = 1

        lu_index_vocab_size: int = constants.LU_INDEX_VOCAB_SIZE
        lu_index_embedding_dim: int = constants.EMBEDDING_DIM
        spatial_features: int = constants.NUM_SAPTIAL_FEATURES
        spatiotemporal_features: int = constants.NUM_SPATIOTEMPORAL_FEATURES
        spatial_filters: int = 128
        spatiotemporal_filters: int = 64

        sto_vars: Tuple[vars.SpatiotemporalOutput, ...] = (
            vars.SpatiotemporalOutput.RH2,
            vars.SpatiotemporalOutput.T2,
            vars.SpatiotemporalOutput.WSPD_WDIR10,
        )

        pad_mode: PadMode = "REFLECT"

    class Input(TypedDict):
        """Input tensors."""

        spatial: tf.Tensor
        spatiotemporal: tf.Tensor
        lu_index: tf.Tensor
        sim_name: str
        date: str

    class InputSpec(TypedDict):
        """TensorSpec for Input."""

        spatial: tf.TensorSpec
        spatiotemporal: tf.TensorSpec
        lu_index: tf.TensorSpec
        sim_name: tf.TensorSpec
        date: tf.TensorSpec

    @classmethod
    def get_input_spec(cls, params: Params) -> InputSpec:
        """Input spec for given params."""
        T, H, W = None, None, None
        return cls.InputSpec(
            spatiotemporal=tf.TensorSpec(
                shape=(T, H, W, params.spatiotemporal_features),
                dtype=tf.float32,
            ),
            spatial=tf.TensorSpec(
                shape=(H, W, params.spatial_features),
                dtype=tf.float32,
            ),
            lu_index=tf.TensorSpec(
                shape=(H, W),
                dtype=tf.int32,
            ),
            sim_name=tf.TensorSpec(shape=(), dtype=tf.string),
            date=tf.TensorSpec(shape=(), dtype=tf.string),
        )

    @classmethod
    def get_input_shape_batched(cls, params: Params) -> dict[str, tf.TypeSpec]:
        """Returns the batched input shape."""
        spec = cls.get_input_spec(params)
        return {k: (None, *v.shape) for k, v in spec.items()}  # type: ignore

    @classmethod
    def get_output_spec(cls, params: Params) -> tf.TensorSpec:
        """Returns the output shape for the given params."""
        H, W = None, None

        return tf.TensorSpec(
            shape=(params.output_timesteps, H, W, len(params.sto_vars)),
            dtype=tf.float32,
        )

    def __init__(self, params: Params | None = None, model: keras.Model | None = None):
        """Creates the Atmo model.

        Args:
            params: Model parameters.
            model: If you already have a keras.Model constructed, pass it here.
        """
        if model is not None:
            self._params = model._params  # type: ignore
            self._model = model
        else:
            self._params = params or self.Params()
            self._model = self._build_model()

    @classmethod
    def from_checkpoint(cls, artifact_uri: str, **kwargs) -> "AtmoModel":
        """Loads the model from a checkpoint URI.

        We load weights only to keep custom methods intact.
        This only works if the model architecture is identical to the architecture
        used during export.

        Args:
            artifact_uri: The path to the SavedModel directory.
                This should end in `/model` if using a GCloud artifact.

        Returns:
            The loaded AtmoModel.
        """
        loaded_model = keras.models.load_model(artifact_uri)
        params = AtmoModel.Params.from_config(loaded_model.get_config())
        model = cls(params=params, **kwargs)
        assert loaded_model is not None, f"Failed to load model from: {artifact_uri}"
        model._model.set_weights(loaded_model.get_weights())
        return model

    @classmethod
    def get_hypermodel(cls, **kwargs) -> keras_tuner.HyperModel:
        """Returns a hypermodel with the given param overrides."""

        def hypermodel(hp: keras_tuner.HyperParameters):
            hp_kwargs = {k: hp.Choice(k, v) for k, v in kwargs.items()}
            return cls(cls.Params(**hp_kwargs))._model

        return hypermodel

    def _build_model(self) -> keras.Model:
        """Creates the correct internal (Keras) model architecture."""
        eval_metrics = [
            keras.metrics.MeanAbsoluteError(),
            keras.metrics.RootMeanSquaredError(),
            keras.metrics.MeanSquaredLogarithmicError(),
            metrics.NormalizedRootMeanSquaredError(),
            metrics.SSIMMetric(),
            metrics.PSNRMetric(),
        ]
        for sto_var in self._params.sto_vars:
            eval_metrics.append(metrics.OutputVarMeanSquaredError(sto_var))

        model = AtmoConvLSTM(self._params)
        model.compile(
            optimizer=self._params.optimizer,
            loss=keras.losses.MeanSquaredError(),
            metrics=eval_metrics,
        )
        model.build(self.get_input_shape_batched(self._params))
        return model

    def summary(self, expand_nested: bool = False):
        """Print the model summary."""
        self._model.summary(expand_nested=expand_nested)

    def call(self, input: Input) -> tf.Tensor:
        """Forward pass for predictions. See `AtmoConvLSTM.call`."""
        return self._model.call(input)

    def predict(self, inputs: Iterable[Input]) -> np.ndarray:
        """Predict on input data."""
        return self._model.predict(inputs)

    def fit(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset | None = None,
        epochs: int = 1,
        steps_per_epoch: int | None = None,
        early_stopping: int | None = None,
        callbacks: List[Callable] | None = None,
        validation_freq: int = 1,
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

        return self._model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            validation_freq=validation_freq,
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

    def save_weights(self, filepath: str) -> None:
        """Saves the model weights."""
        self._model.save_weights(filepath)
        logging.info("Saved model weights to %s", filepath)

    def load_model(self, filepath: str) -> None:
        """Loads the entire model (architecture + weights)."""
        self._model = keras.models.load_model(filepath)
        logging.info("Loaded full model from %s", filepath)


###############################################################################
#                       Custom Keras Model definitions
#
# The following are keras.Model class implementations for AtmoML model
# architecture(s). They can be used within the generic AtmoModel class above
# for training and evaluation, and only define basic components of the model,
# such as layers and the forward pass. While these models are callable, all
# data pre- and post-processing are expected to be handled externally.
###############################################################################


class AtmoConvLSTM(keras.Model):
    """Atmo ConvLSTM model.

    The architecture is a multi-head ConvLSTM model.

    Spatial and spatiotemporal features are passed into separate CNN blocks for
    feature extraction before being concatenated and fed into a ConvLSTM. The
    ConvLSTM outputs are then fed into separate TransposeConv blocks for
    multi-prediction. There are four different output heads for 2m temperature,
    2m relative humidity, 10m wind speed, and 10m wind direction.

    Architecture diagram: https://miro.com/app/board/uXjVK3Q31rQ=/.
    Architecture details:
    https://www.notion.so/climate-iq/AtmoML-Architecture-Proposals-and-Design-c00d0e54265c4bb8a72ce01fd475f116.
    """

    def __init__(self, params: AtmoModel.Params):
        """Creates the Atmo ConvLSTM model.

        Args:
            params: An dictionary of configurable model parameters.
            num_spatial_features: Total dimensionality of the spatial features.
                Needed for defining input shapes.
            num_spatiotemporal_features: Total dimensionality of the spatiotemporal
                features. Needed for defining input shapes.
            lu_index_vocab_size (int): The number of unique values in the lu_index
                feature. This is used to define the size of the vocabulary for the
                embedding layer.
            embedding_dim (int): Size of the embedding vectors for the lu_index
            feature. This determines the dimensionality of the embedding space.
        """
        super().__init__()

        self._params = params

        # Model definition
        T, H, W = None, None, None
        K_SIZE = self._params.input_cnn_kernel_size
        K_PAD = K_SIZE // 2
        C1_STRIDE, C2_STRIDE = (
            self._params.conv1_stride,
            self._params.conv2_stride,
        )
        F_S = self._params.spatial_features
        F_ST = self._params.spatiotemporal_features
        LUI_VOCAB = self._params.lu_index_vocab_size
        LUI_DIM = self._params.lu_index_embedding_dim
        S_FILTERS = self._params.spatial_filters
        ST_FILTERS = self._params.spatiotemporal_filters
        PAD_MODE = self._params.pad_mode

        # Define Embedding Layer for lu_index
        self.lu_index_embedding = keras.Sequential(
            [
                layers.InputLayer((H, W)),
                layers.Embedding(input_dim=LUI_VOCAB, output_dim=LUI_DIM),
            ]
        )

        # Spatial CNN
        spatial_cnn_params = ConvParams(
            padding="valid", activation=self._params.spatial_activation
        )
        self._spatial_cnn = keras.Sequential(
            [
                layers.InputLayer((H, W, F_S + LUI_DIM)),
                pad_layers.Pad2D((K_PAD, K_PAD), mode=PAD_MODE),
                layers.Conv2D(
                    S_FILTERS // 2, K_SIZE, strides=C1_STRIDE, **spatial_cnn_params
                ),
                layers.MaxPool2D(pool_size=2, strides=1, padding="same"),
                pad_layers.Pad2D((K_PAD, K_PAD), mode=PAD_MODE),
                layers.Conv2D(
                    S_FILTERS, K_SIZE, strides=C2_STRIDE, **spatial_cnn_params
                ),
                layers.MaxPool2D(pool_size=2, strides=1, padding="same"),
            ],
            name="spatial_cnn",
        )

        # Spatiotemporal CNN
        st_cnn_params = ConvParams(
            padding="valid", activation=self._params.st_activation
        )
        self._st_cnn = keras.Sequential(
            [
                layers.InputLayer((T, H, W, F_ST)),
                # Remaining layers are TimeDistributed and are applied to each
                # temporal slice
                layers.TimeDistributed(pad_layers.Pad2D((K_PAD, K_PAD), mode=PAD_MODE)),
                layers.TimeDistributed(
                    layers.Conv2D(
                        ST_FILTERS // 4, K_SIZE, strides=C1_STRIDE, **st_cnn_params
                    )
                ),
                layers.TimeDistributed(
                    layers.MaxPool2D(pool_size=2, strides=1, padding="same")
                ),
                layers.TimeDistributed(pad_layers.Pad2D((K_PAD, K_PAD), mode=PAD_MODE)),
                layers.TimeDistributed(
                    layers.Conv2D(
                        ST_FILTERS, K_SIZE, strides=C2_STRIDE, **st_cnn_params
                    )
                ),
                layers.TimeDistributed(
                    layers.MaxPool2D(pool_size=2, strides=1, padding="same")
                ),
            ],
            name="spatiotemporal_cnn",
        )

        # ConvLSTM
        # The spatial dimensions have been reduced (C1_S x C2_S) by the CNNs.
        # The "channel" dimension is double the sum of the channels from the CNNs,
        # due to stacking two boundary condition pairs.
        LSTM_C = 2 * (S_FILTERS + ST_FILTERS)  # LSTM channels
        LSTM_H, LSTM_W = None, None  # LSTM height and width
        LSTM_FILTERS = self._params.lstm_units  # LSTM Filters
        LSTM_K_SIZE = self._params.lstm_kernel_size
        LSTM_K_PAD = LSTM_K_SIZE // 2
        self.conv_lstm = keras.Sequential(
            [
                # Input shape: (time_steps, height, width, channels)
                layers.InputLayer((T, LSTM_H, LSTM_W, LSTM_C)),
                layers.TimeDistributed(
                    pad_layers.Pad2D((LSTM_K_PAD, LSTM_K_PAD), mode=PAD_MODE)
                ),
                layers.ConvLSTM2D(
                    LSTM_FILTERS,
                    LSTM_K_SIZE,
                    return_sequences=True,
                    strides=1,
                    padding="valid",
                    activation=self._params.lstm_activation,
                    dropout=self._params.lstm_dropout,
                    recurrent_dropout=self._params.lstm_recurrent_dropout,
                ),
            ],
            name="conv_lstm",
        )

        OUTPUT_K_SIZE = self._params.output_cnn_kernel_size

        # Output CNNs (upsampling via TransposeConv)
        # We return separate sub-models (i.e., branches) for each output.
        output_cnn_params = ConvParams(activation="relu", padding="valid")
        output_cnn_input_shape = (T, LSTM_H, LSTM_W, LSTM_FILTERS // 2)

        # Output: T2 (2m temperature)
        OUTPUT_K_SIZE = self._params.output_cnn_kernel_size
        self._t2_output_cnn = (
            keras.Sequential(
                [
                    layers.InputLayer(output_cnn_input_shape),
                    layers.TimeDistributed(
                        keras.Sequential(
                            [
                                layers.Conv2D(
                                    LSTM_FILTERS // 2,
                                    OUTPUT_K_SIZE,
                                    strides=C1_STRIDE,
                                    **output_cnn_params,
                                ),
                                layers.Conv2D(
                                    LSTM_FILTERS // 4,
                                    OUTPUT_K_SIZE,
                                    strides=C2_STRIDE,
                                    **output_cnn_params,
                                ),
                                layers.Conv2D(
                                    LSTM_FILTERS // 8,
                                    OUTPUT_K_SIZE,
                                    **output_cnn_params,
                                ),
                                layers.Conv2D(1, OUTPUT_K_SIZE, **output_cnn_params),
                            ]
                        )
                    ),
                ],
                name="t2_output_cnn",
            )
            if vars.SpatiotemporalOutput.T2 in self._params.sto_vars
            else None
        )

        # Output: RH2 (2m relative humidity)
        self._rh2_output_cnn = (
            keras.Sequential(
                [
                    layers.InputLayer(output_cnn_input_shape),
                    layers.TimeDistributed(
                        keras.Sequential(
                            [
                                layers.Conv2D(
                                    LSTM_FILTERS // 2,
                                    OUTPUT_K_SIZE,
                                    strides=C1_STRIDE,
                                    **output_cnn_params,
                                ),
                                layers.Conv2D(
                                    LSTM_FILTERS // 4,
                                    OUTPUT_K_SIZE,
                                    strides=C2_STRIDE,
                                    **output_cnn_params,
                                ),
                                layers.Conv2D(
                                    LSTM_FILTERS // 8,
                                    OUTPUT_K_SIZE,
                                    **output_cnn_params,
                                ),
                                layers.Conv2D(1, OUTPUT_K_SIZE, **output_cnn_params),
                            ]
                        )
                    ),
                ],
                name="rh2_output_cnn",
            )
            if vars.SpatiotemporalOutput.RH2 in self._params.sto_vars
            else None
        )

        # Output: WSPD10 (10m wind speed)
        self._wspd10_output_cnn = (
            keras.Sequential(
                [
                    layers.InputLayer(output_cnn_input_shape),
                    layers.TimeDistributed(
                        keras.Sequential(
                            [
                                layers.Conv2D(
                                    LSTM_FILTERS // 2,
                                    OUTPUT_K_SIZE,
                                    strides=C1_STRIDE,
                                    **output_cnn_params,
                                ),
                                layers.Conv2D(
                                    LSTM_FILTERS // 4,
                                    OUTPUT_K_SIZE,
                                    strides=C2_STRIDE,
                                    **output_cnn_params,
                                ),
                                layers.Conv2D(
                                    LSTM_FILTERS // 8,
                                    OUTPUT_K_SIZE,
                                    **output_cnn_params,
                                ),
                                layers.Conv2D(1, OUTPUT_K_SIZE, **output_cnn_params),
                            ]
                        )
                    ),
                ],
                name="wspd10_output_cnn",
            )
            if vars.SpatiotemporalOutput.WSPD_WDIR10 in self._params.sto_vars
            else None
        )

        # Output: WDIR10 (10m wind direction sine and cosine functions)
        has_windir10_cos = (
            vars.SpatiotemporalOutput.WSPD_WDIR10_COS in self._params.sto_vars
        )
        has_windir10_sin = (
            vars.SpatiotemporalOutput.WSPD_WDIR10_SIN in self._params.sto_vars
        )
        self._wdir10_output_cnn = (
            keras.Sequential(
                [
                    layers.InputLayer(output_cnn_input_shape),
                    layers.TimeDistributed(
                        keras.Sequential(
                            [
                                layers.Conv2D(
                                    LSTM_FILTERS // 2,
                                    OUTPUT_K_SIZE,
                                    strides=C1_STRIDE,
                                    **output_cnn_params,
                                ),
                                layers.Conv2D(
                                    LSTM_FILTERS // 4,
                                    OUTPUT_K_SIZE,
                                    strides=C2_STRIDE,
                                    **output_cnn_params,
                                ),
                                layers.Conv2D(
                                    LSTM_FILTERS // 8,
                                    OUTPUT_K_SIZE,
                                    **output_cnn_params,
                                ),
                                layers.Conv2D(2, OUTPUT_K_SIZE, **output_cnn_params),
                            ]
                        )
                    ),
                ],
                name="wdir10_output_cnn",
            )
            if (has_windir10_cos and has_windir10_sin)
            else None
        )

    def call(self, inputs: AtmoModel.Input) -> tf.Tensor:
        """Makes a forward pass of the model.

        Args:
            inputs: AtmoModel input tensors. See `AtmoModel.Input`.

        Returns:
            A rank-5 tensor of all output predictions.
        """
        spatial_input = inputs["spatial"]
        st_input = inputs["spatiotemporal"]
        lu_index_input = inputs["lu_index"]

        B, T, H, W, *_ = st_input.shape
        C = self._params.spatial_features
        STF = self._params.spatiotemporal_features
        T = constants.INPUT_TIME_STEPS
        T_O = self._params.output_timesteps

        tf.ensure_shape(spatial_input, (B, H, W, C))
        tf.ensure_shape(st_input, (B, T, H, W, STF))
        tf.ensure_shape(lu_index_input, (B, H, W))

        lu_index_embedded = self.lu_index_embedding(lu_index_input)

        # Concatenate lu_index embedding with other spatial features
        spatial_input = tf.concat([spatial_input, lu_index_embedded], axis=-1)

        spatial_cnn_output = self._spatial_cnn(spatial_input)
        st_cnn_output = self._st_cnn(st_input)

        # Concatenate spatial CNN outputs with all spatiotemporal tensors
        # and transform time sequence into boundary condition pairs.
        spatial_cnn_output = spatial_cnn_output[:, tf.newaxis, :, :, :]
        spatial_cnn_output = tf.repeat(spatial_cnn_output, [T], axis=1)
        concat_inputs = tf.concat([st_cnn_output, spatial_cnn_output], axis=-1)

        lstm_input = data_utils.boundary_pairs(concat_inputs)
        lstm_output = self.conv_lstm(lstm_input)

        # Split up paired tensors into individual time steps.
        trconv_input = data_utils.split_time_step_pairs(lstm_output)[:, -T_O:]

        outputs = []
        if self._t2_output_cnn is not None:
            outputs.append(self._t2_output_cnn(trconv_input))
        if self._rh2_output_cnn is not None:
            outputs.append(self._rh2_output_cnn(trconv_input))
        if self._wspd10_output_cnn is not None:
            outputs.append(self._wspd10_output_cnn(trconv_input))
        if self._wdir10_output_cnn is not None:
            outputs.append(self._wdir10_output_cnn(trconv_input))

        output = tf.concat(outputs, axis=-1)
        tf.ensure_shape(output, (B, T_O, H, W, None))
        return output

    def get_config(self) -> dict:
        """Keras serialization."""
        return self._params.get_config()

    @classmethod
    def from_config(cls, config: dict) -> "AtmoConvLSTM":
        """Keras deserialization."""
        return cls(params=AtmoModel.Params.from_config(config))

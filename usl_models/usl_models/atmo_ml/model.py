"""AtmoML model definition."""

import logging
from typing import TypedDict, List, Callable, Mapping, Any, Literal

import keras
from keras import layers
import tensorflow as tf

from usl_models.atmo_ml import data_utils
from usl_models.atmo_ml import constants
from usl_models.atmo_ml import metrics
from usl_models.atmo_ml import vars


class ConvParams(TypedDict):
    activation: Literal["relu", "sigmoid", "tanh", "softmax"]
    padding: Literal["valid", "same"]


class AtmoModel:
    """Atmo model class."""

    class Params(TypedDict):
        """Model parameters dictionary."""

        # General parameters.
        batch_size: int

        # Layer-specific parameters.
        lstm_units: int
        lstm_kernel_size: int
        lstm_dropout: float
        lstm_recurrent_dropout: float

        # The optimizer configuration.
        # We use the dictionary definition to ensure the model is serializable.
        # This value is passed to keras.optimizers.get to build the optimizer object.
        optimizer_config: Mapping[str, Any]

        output_timesteps: int
        conv1_stride: int
        conv2_stride: int

        lu_index_vocab_size: int
        lu_index_embedding_dim: int
        spatial_features: int
        spatiotemporal_features: int
        spatial_filters: int
        spatiotemporal_filters: int

    @classmethod
    def default_params(cls) -> Params:
        """Returns the default params for the model."""
        return cls.Params(
            optimizer_config={
                "class_name": "Adam",
                "config": {"learning_rate": 1e-4},
            },
            batch_size=4,
            lstm_units=512,
            lstm_kernel_size=5,
            lstm_dropout=0.2,
            lstm_recurrent_dropout=0.2,
            output_timesteps=1,
            conv1_stride=1,
            conv2_stride=1,
            lu_index_vocab_size=constants.LU_INDEX_VOCAB_SIZE,
            lu_index_embedding_dim=constants.EMBEDDING_DIM,
            spatial_features=constants.NUM_SAPTIAL_FEATURES,
            spatiotemporal_features=constants.NUM_SPATIOTEMPORAL_FEATURES,
            spatial_filters=128,
            spatiotemporal_filters=64,
        )

    class Input(TypedDict):
        """Input tensors dictionary."""

        spatial: tf.Tensor
        spatiotemporal: tf.Tensor
        lu_index: tf.Tensor

    class Result(TypedDict):
        """Prediction result dictionary."""

        prediction: tf.Tensor
        chunk_id: str | tf.Tensor

    def __init__(
        self,
        params: Params | None = None,
    ):
        """Creates the Atmo model.

        Args:
            params: A dictionary of configurable model parameters.
            spatial_dims: Tuple of spatial height and width input dimensions.
                Needed for defining input shapes. This is an optional arg that
                can be changed (primarily for testing/debugging).
            num_spatial_features: nb of spt features
            num_spatiotemporal_features: nb of spatiotemp feat.
            lu_index_vocab_size: Number of unique values in the lu_index
            feature.
            embedding_dim: Size of the embedding vectors for lu_index.
        """
        self._model_params = params or self.default_params()
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
        model = cls(**kwargs)
        loaded_model = keras.models.load_model(artifact_uri)
        assert loaded_model is not None, f"Failed to load model from: {artifact_uri}"
        model._model.set_weights(loaded_model.get_weights())
        return model

    def _build_model(self) -> keras.Model:
        """Creates the correct internal (Keras) model architecture."""
        model = AtmoConvLSTM(self._model_params)
        model.compile(
            optimizer=keras.optimizers.get(self._model_params["optimizer_config"]),
            loss=keras.losses.MeanSquaredError(),
            metrics=[
                keras.metrics.MeanAbsoluteError(),
                keras.metrics.RootMeanSquaredError(),
                metrics.OutputVarMeanSquaredError(vars.SpatiotemporalOutput.RH2),
                metrics.OutputVarMeanSquaredError(vars.SpatiotemporalOutput.T2),
                metrics.OutputVarMeanSquaredError(
                    vars.SpatiotemporalOutput.WSPD_WDIR10
                ),
                metrics.OutputVarMeanSquaredError(
                    vars.SpatiotemporalOutput.WSPD_WDIR10_COS
                ),
                metrics.OutputVarMeanSquaredError(
                    vars.SpatiotemporalOutput.WSPD_WDIR10_SIN
                ),
            ],
        )
        model.build(constants.get_input_shape_batched(height=None, width=None))
        return model

    def summary(self, expand_nested: bool = False):
        """Print the model summary."""
        self._model.summary(expand_nested=expand_nested)

    def call(self, input: Input) -> tf.Tensor:
        """Forward pass for predictions. See `AtmoConvLSTM.call`."""
        return self._model.call(input)

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
        K_SIZE = self._params["lstm_kernel_size"]  # Conv kernel size
        C1_STRIDE, C2_STRIDE = (
            self._params["conv1_stride"],
            self._params["conv2_stride"],
        )
        F_S = self._params["spatial_features"]
        F_ST = self._params["spatiotemporal_features"]
        LUI_VOCAB = self._params["lu_index_vocab_size"]
        LUI_DIM = self._params["lu_index_embedding_dim"]
        S_FILTERS = self._params["spatial_filters"]
        ST_FILTERS = self._params["spatiotemporal_filters"]

        # Define Embedding Layer for lu_index
        self.lu_index_embedding = keras.Sequential(
            [
                layers.InputLayer((H, W)),
                layers.Embedding(input_dim=LUI_VOCAB, output_dim=LUI_DIM),
            ]
        )

        # Spatial CNN
        spatial_cnn_params = ConvParams(padding="same", activation="relu")
        self._spatial_cnn = keras.Sequential(
            [
                layers.InputLayer((H, W, F_S + LUI_DIM)),
                layers.Conv2D(
                    S_FILTERS // 2, K_SIZE, strides=C1_STRIDE, **spatial_cnn_params
                ),
                layers.MaxPool2D(pool_size=2, strides=1, padding="same"),
                layers.Conv2D(
                    S_FILTERS, K_SIZE, strides=C2_STRIDE, **spatial_cnn_params
                ),
                layers.MaxPool2D(pool_size=2, strides=1, padding="same"),
            ],
            name="spatial_cnn",
        )

        # Spatiotemporal CNN
        st_cnn_params = ConvParams(padding="same", activation="relu")
        self._st_cnn = keras.Sequential(
            [
                layers.InputLayer((T, H, W, F_ST)),
                # Remaining layers are TimeDistributed and are applied to each
                # temporal slice
                layers.TimeDistributed(
                    layers.Conv2D(
                        ST_FILTERS // 4, K_SIZE, strides=C1_STRIDE, **st_cnn_params
                    )
                ),
                layers.TimeDistributed(
                    layers.MaxPool2D(pool_size=2, strides=1, padding="same")
                ),
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
        LSTM_FILTERS = self._params["lstm_units"]  # LSTM Filters
        self.conv_lstm = keras.Sequential(
            [
                # Input shape: (time_steps, height, width, channels)
                layers.InputLayer((T, LSTM_H, LSTM_W, LSTM_C)),
                layers.ConvLSTM2D(
                    LSTM_FILTERS,
                    K_SIZE,
                    return_sequences=True,
                    strides=1,
                    padding="same",
                    activation="tanh",
                    dropout=self._params["lstm_dropout"],
                    recurrent_dropout=self._params["lstm_recurrent_dropout"],
                ),
            ],
            name="conv_lstm",
        )

        # Output CNNs (upsampling via TransposeConv)
        # We return separate sub-models (i.e., branches) for each output.
        output_cnn_params = ConvParams(padding="same", activation="relu")
        output_cnn_input_shape = (T, LSTM_H, LSTM_W, LSTM_FILTERS // 2)

        # Output: T2 (2m temperature)
        self._t2_output_cnn = keras.Sequential(
            [
                layers.InputLayer(output_cnn_input_shape),
                layers.TimeDistributed(
                    layers.Conv2DTranspose(
                        64, K_SIZE, strides=C1_STRIDE, **output_cnn_params
                    )
                ),
                layers.TimeDistributed(
                    layers.Conv2DTranspose(
                        16, K_SIZE, strides=C2_STRIDE, **output_cnn_params
                    )
                ),
                layers.TimeDistributed(
                    layers.Conv2DTranspose(1, K_SIZE, strides=1, **output_cnn_params)
                ),
            ],
            name="t2_output_cnn",
        )

        # Output: RH2 (2m relative humidity)
        self._rh2_output_cnn = keras.Sequential(
            [
                layers.InputLayer(output_cnn_input_shape),
                layers.TimeDistributed(
                    layers.Conv2DTranspose(
                        64, K_SIZE, strides=C1_STRIDE, **output_cnn_params
                    )
                ),
                layers.TimeDistributed(
                    layers.Conv2DTranspose(
                        16, K_SIZE, strides=C2_STRIDE, **output_cnn_params
                    )
                ),
                layers.TimeDistributed(
                    layers.Conv2DTranspose(1, K_SIZE, strides=1, **output_cnn_params)
                ),
            ],
            name="rh2_output_cnn",
        )

        # Output: WSPD10 (10m wind speed)
        self._wspd10_output_cnn = keras.Sequential(
            [
                layers.InputLayer(output_cnn_input_shape),
                layers.TimeDistributed(
                    layers.Conv2DTranspose(
                        64, K_SIZE, strides=C1_STRIDE, **output_cnn_params
                    )
                ),
                layers.TimeDistributed(
                    layers.Conv2DTranspose(
                        16, K_SIZE, strides=C2_STRIDE, **output_cnn_params
                    )
                ),
                layers.TimeDistributed(
                    layers.Conv2DTranspose(1, K_SIZE, strides=1, **output_cnn_params)
                ),
            ],
            name="wspd10_output_cnn",
        )

        # Output: WDIR10 (10m wind direction sine and cosine functions)
        self._wdir10_output_cnn = keras.Sequential(
            [
                layers.InputLayer(output_cnn_input_shape),
                layers.TimeDistributed(
                    layers.Conv2DTranspose(
                        64, K_SIZE, strides=C1_STRIDE, **output_cnn_params
                    )
                ),
                layers.TimeDistributed(
                    layers.Conv2DTranspose(
                        16, K_SIZE, strides=C1_STRIDE, **output_cnn_params
                    )
                ),
                layers.TimeDistributed(
                    layers.Conv2DTranspose(2, K_SIZE, strides=1, **output_cnn_params)
                ),
            ],
            name="wdir10_output_cnn",
        )

    def call(self, inputs: AtmoModel.Input) -> tf.Tensor:
        """Makes a forward pass of the model.

        Args:
            inputs: A dictionary of input tensors, with the following key/value pairs:
                "spatial": rank-4 tensor
                "spatiotemporal": rank-5 tensor

        Returns:
            A rank-5 tensor of all output predictions.
        """
        spatial_input = inputs["spatial"]  # (B, H, W, C)
        st_input = inputs["spatiotemporal"]
        lu_index_input = inputs["lu_index"]  # lu_index is passed separately

        B, T, H, W, *_ = st_input.shape
        C = constants.NUM_SAPTIAL_FEATURES
        STF = constants.NUM_SPATIOTEMPORAL_FEATURES
        T = constants.INPUT_TIME_STEPS
        T_O = self._params["output_timesteps"]

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

        return tf.concat(outputs, axis=-1)

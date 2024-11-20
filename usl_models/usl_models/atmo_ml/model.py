"""AtmoML model definition."""

import logging
from typing import TypeAlias, TypedDict, List, Callable
import keras
import tensorflow as tf
from keras import layers
from keras.layers import Embedding
from usl_models.atmo_ml import constants
from usl_models.atmo_ml import data_utils
from usl_models.atmo_ml import model_params

AtmoModelParams: TypeAlias = model_params.AtmoModelParams


class AtmoModel:
    """Atmo model class."""

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
        params: AtmoModelParams | None = None,
        spatial_dims: tuple[int, int] = (constants.MAP_HEIGHT, constants.MAP_WIDTH),
        num_spatial_features: int = constants.NUM_SAPTIAL_FEATURES,
        num_spatiotemporal_features: int = constants.NUM_SPATIOTEMPORAL_FEATURES,
        lu_index_vocab_size: int = constants.LU_INDEX_VOCAB_SIZE,
        embedding_dim: int = constants.EMBEDDING_DIM,
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
        self._model_params = params or model_params.default_params()
        self._spatial_dims = spatial_dims
        self._spatial_features = num_spatial_features
        self._spatiotemporal_features = num_spatiotemporal_features
        self._lu_index_vocab_size = lu_index_vocab_size
        self._embedding_dim = embedding_dim
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
        model = AtmoConvLSTM(
            self._model_params,
            spatial_dims=self._spatial_dims,
            num_spatial_features=self._spatial_features,
            num_spatiotemporal_features=self._spatiotemporal_features,
            lu_index_vocab_size=self._lu_index_vocab_size,
            embedding_dim=self._embedding_dim,
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


class AtmoConvLSTM(tf.keras.Model):
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

    def __init__(
        self,
        params: AtmoModelParams,
        spatial_dims: tuple[int, int],
        num_spatial_features: int,
        num_spatiotemporal_features: int,
        lu_index_vocab_size: int = 61,  # Nb of unique classes in lu_index
        embedding_dim: int = 8,  # Size of the embedding vectors for lu_index
    ):
        """Creates the Atmo ConvLSTM model.

        Args:
            params: An dictionary of configurable model parameters.
            spatial_dims: Tuple of spatial height and width input dimensions.
                Needed for defining input shapes.
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
        self._spatial_height, self._spatial_width = spatial_dims
        self._spatial_features = num_spatial_features
        self._spatiotemporal_features = num_spatiotemporal_features
        self._embedding_dim = embedding_dim  # Save embedding_dim as a class attribute

        # Define Embedding Layer for lu_index
        self.lu_index_embedding = Embedding(
            input_dim=lu_index_vocab_size,
            output_dim=embedding_dim,
            input_length=self._spatial_height * self._spatial_width,
        )

        # Model definition

        # Spatial CNN
        spatial_cnn_params = {"strides": 2, "padding": "same", "activation": "relu"}
        self._spatial_cnn = tf.keras.Sequential(
            [
                # Input shape: (height, width, channels)
                layers.InputLayer(
                    (
                        self._spatial_height,
                        self._spatial_width,
                        self._spatial_features + embedding_dim,
                    )
                ),
                layers.Conv2D(64, 5, **spatial_cnn_params),
                layers.MaxPool2D(pool_size=2, strides=1, padding="same"),
                layers.Conv2D(128, 5, **spatial_cnn_params),
                layers.MaxPool2D(pool_size=2, strides=1, padding="same"),
            ],
            name="spatial_cnn",
        )

        # Spatiotemporal CNN
        st_cnn_params = {"strides": 2, "padding": "same", "activation": "relu"}
        self._st_cnn = tf.keras.Sequential(
            [
                # Input shape: (time_steps, height, width, channels)
                layers.InputLayer(
                    (
                        None,
                        self._spatial_height,
                        self._spatial_width,
                        self._spatiotemporal_features,
                    )
                ),
                # Remaining layers are TimeDistributed and are applied to each
                # temporal slice
                layers.TimeDistributed(layers.Conv2D(16, 5, **st_cnn_params)),
                layers.TimeDistributed(
                    layers.MaxPool2D(pool_size=2, strides=1, padding="same")
                ),
                layers.TimeDistributed(layers.Conv2D(64, 5, **st_cnn_params)),
                layers.TimeDistributed(
                    layers.MaxPool2D(pool_size=2, strides=1, padding="same")
                ),
            ],
            name="spatiotemporal_cnn",
        )

        # ConvLSTM
        # The spatial dimensions have been reduced 4x by the CNNs.
        # The "channel" dimension is double the sum of the channels from the CNNs,
        # due to stacking two boundary condition pairs.
        conv_lstm_height = self._spatial_height // 4
        conv_lstm_width = self._spatial_width // 4
        conv_lstm_channels = 2 * (128 + 64)
        self.conv_lstm = tf.keras.Sequential(
            [
                # Input shape: (time_steps, height, width, channels)
                layers.InputLayer(
                    (None, conv_lstm_height, conv_lstm_width, conv_lstm_channels)
                ),
                layers.ConvLSTM2D(
                    self._params["lstm_units"],
                    self._params["lstm_kernel_size"],
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
        output_cnn_params = {"padding": "same", "activation": "relu"}
        # Input shape: (time, height, width, channels)
        output_cnn_input_shape = (
            None,
            conv_lstm_height,
            conv_lstm_width,
            self._params["lstm_units"] // 2,
        )

        # Output: T2 (2m temperature)
        self._t2_output_cnn = tf.keras.Sequential(
            [
                layers.InputLayer(output_cnn_input_shape),
                layers.TimeDistributed(
                    layers.Conv2DTranspose(64, 2, strides=2, **output_cnn_params)
                ),
                layers.TimeDistributed(
                    layers.Conv2DTranspose(16, 2, strides=2, **output_cnn_params)
                ),
                layers.TimeDistributed(
                    layers.Conv2DTranspose(1, 1, strides=1, **output_cnn_params)
                ),
            ],
            name="t2_output_cnn",
        )

        # Output: RH2 (2m relative humidity)
        self._rh2_output_cnn = tf.keras.Sequential(
            [
                layers.InputLayer(output_cnn_input_shape),
                layers.TimeDistributed(
                    layers.Conv2DTranspose(64, 2, strides=2, **output_cnn_params)
                ),
                layers.TimeDistributed(
                    layers.Conv2DTranspose(16, 2, strides=2, **output_cnn_params)
                ),
                layers.TimeDistributed(
                    layers.Conv2DTranspose(1, 1, strides=1, **output_cnn_params)
                ),
            ],
            name="rh2_output_cnn",
        )

        # Output: WSPD10 (10m wind speed)
        self._wspd10_output_cnn = tf.keras.Sequential(
            [
                layers.InputLayer(output_cnn_input_shape),
                layers.TimeDistributed(
                    layers.Conv2DTranspose(64, 2, strides=2, **output_cnn_params)
                ),
                layers.TimeDistributed(
                    layers.Conv2DTranspose(16, 2, strides=2, **output_cnn_params)
                ),
                layers.TimeDistributed(
                    layers.Conv2DTranspose(1, 1, strides=1, **output_cnn_params)
                ),
            ],
            name="wspd10_output_cnn",
        )

        # Output: WDIR10 (10m wind direction sine and cosine functions)
        self._wdir10_output_cnn = tf.keras.Sequential(
            [
                layers.InputLayer(output_cnn_input_shape),
                layers.TimeDistributed(
                    layers.Conv2DTranspose(64, 2, strides=2, **output_cnn_params)
                ),
                layers.TimeDistributed(
                    layers.Conv2DTranspose(16, 2, strides=2, **output_cnn_params)
                ),
                layers.TimeDistributed(
                    layers.Conv2DTranspose(2, 1, strides=1, **output_cnn_params)
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

        B = st_input.shape[0]
        C = constants.NUM_SAPTIAL_FEATURES
        H, W = constants.MAP_HEIGHT, constants.MAP_WIDTH
        STF = constants.NUM_SPATIOTEMPORAL_FEATURES
        T = constants.INPUT_TIME_STEPS

        tf.ensure_shape(spatial_input, (B, H, W, C))
        tf.ensure_shape(st_input, (B, T, H, W, STF))
        tf.ensure_shape(lu_index_input, (B, H, W))

        # Ensure all spatial features are present; if not, replace with zeros
        if spatial_input.shape[-1] < self._spatial_features:
            missing_channels = self._spatial_features - spatial_input.shape[-1]
            spatial_input = tf.pad(
                spatial_input,
                paddings=[[0, 0], [0, 0], [0, 0], [0, missing_channels]],
                constant_values=0,
            )

        # Ensure all spatiotemporal features are present; if not, replace with zeros
        if st_input.shape[-1] < self._spatiotemporal_features:
            missing_channels = self._spatiotemporal_features - st_input.shape[-1]
            st_input = tf.pad(
                st_input,
                paddings=[[0, 0], [0, 0], [0, 0], [0, 0], [0, missing_channels]],
                constant_values=0,
            )
        print("lu_index_input", lu_index_input.shape)

        # Reshape lu_index matrix for embedding
        lu_index_input_flat = tf.reshape(
            lu_index_input, (-1, self._spatial_height * self._spatial_width)
        )
        lu_index_embedded_flat = self.lu_index_embedding(lu_index_input_flat)
        print("lu_index_embedded_flat", lu_index_embedded_flat)

        # Reshape back to matrix form (200, 200, ?, ?)
        lu_index_embedded = tf.reshape(
            lu_index_embedded_flat,
            (-1, self._spatial_height, self._spatial_width, self._embedding_dim),
        )
        print("lu_index_embedded", lu_index_embedded.shape)

        # Concatenate lu_index embedding with other spatial features
        spatial_input = tf.concat([spatial_input, lu_index_embedded], axis=-1)

        spatial_cnn_output = self._spatial_cnn(spatial_input)
        st_cnn_output = self._st_cnn(st_input)

        # Concatenate spatial CNN outputs with all spatiotemporal tensors
        # and transform time sequence into boundary condition pairs.
        spatial_cnn_output = spatial_cnn_output[:, tf.newaxis, :, :, :]
        n = st_input.shape[1]
        spatial_cnn_output = tf.repeat(spatial_cnn_output, [n], axis=1)
        print("spatial_cnn_output", spatial_cnn_output.shape)
        concat_inputs = tf.concat([st_cnn_output, spatial_cnn_output], axis=-1)

        lstm_input = data_utils.boundary_pairs(concat_inputs)
        lstm_output = self.conv_lstm(lstm_input)
        print("lstm_output", lstm_output.shape)

        # Split up paired tensors into individual time steps.
        trconv_input = data_utils.split_time_step_pairs(lstm_output)
        print("trconv_input", trconv_input.shape)

        outputs = [
            self._t2_output_cnn(trconv_input),
            self._rh2_output_cnn(trconv_input),
            self._wspd10_output_cnn(trconv_input),
            self._wdir10_output_cnn(trconv_input),
        ]
        print("outputs[0]", outputs[0].shape)
        return tf.concat(outputs, axis=-1)

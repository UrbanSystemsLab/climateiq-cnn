"""AtmoML model definition."""

from typing import TypeAlias, TypedDict

import tensorflow as tf
from tensorflow.keras import layers

# from usl_models.atmo_ml import constants
from usl_models.atmo_ml import data_utils
from usl_models.atmo_ml import model_params

AtmoModelParams: TypeAlias = model_params.AtmoModelParams


class AtmoInput(TypedDict):
    """Input tensors dictionary for the Atmo model."""

    spatial: tf.Tensor
    spatiotemporal: tf.Tensor


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
    ):
        """Creates the Atmo ConvLSTM model.

        Args:
            params: An object of configurable model parameters.
            spatial_dims: Tuple of spatial height and width input dimensions.
                Needed for defining input shapes.
            num_spatial_features: Total dimensionality of the spatial features.
                Needed for defining input shapes.
            num_spatiotemporal_features: Total dimensionality of the spatiotemporal
                features. Needed for defining input shapes.
        """
        super().__init__()

        self._params = params
        self._spatial_height, self._spatial_width = spatial_dims
        self._spatial_features = num_spatial_features
        self._spatiotemporal_features = num_spatiotemporal_features

        # Model definition

        # Spatial CNN
        spatial_cnn_params = {"strides": 2, "padding": "same", "activation": "relu"}
        self._spatial_cnn = tf.keras.Sequential(
            [
                # Input shape: (height, width, channels)
                layers.InputLayer(
                    (self._spatial_height, self._spatial_width, self._spatial_features)
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
                    self._params.lstm_units,
                    self._params.lstm_kernel_size,
                    return_sequences=True,
                    strides=1,
                    padding="same",
                    activation="tanh",
                    dropout=self._params.lstm_dropout,
                    recurrent_dropout=self._params.lstm_recurrent_dropout,
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
            self._params.lstm_units // 2,
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

    def call(self, inputs: AtmoInput) -> tf.Tensor:
        """Makes a forward pass of the model.

        Args:
            inputs: A dictionary of input tensors, with the following key/value pairs:
                "spatial": rank-4 tensor
                "spatiotemporal": rank-5 tensor

        Returns:
            A rank-5 tensor of all output predictions.
        """
        spatial_input = inputs["spatial"]
        st_input = inputs["spatiotemporal"]

        spatial_cnn_output = self._spatial_cnn(spatial_input)
        st_cnn_output = self._st_cnn(st_input)

        # Concatenate spatial CNN outputs with all spatiotemporal tensors
        # and transform time sequence into boundary condition pairs.
        spatial_cnn_output = spatial_cnn_output[:, tf.newaxis, :, :, :]
        n = st_input.shape[1]
        spatial_cnn_output = tf.repeat(spatial_cnn_output, [n], axis=1)
        concat_inputs = tf.concat([st_cnn_output, spatial_cnn_output], axis=-1)

        lstm_input = data_utils.boundary_pairs(concat_inputs)
        lstm_output = self.conv_lstm(lstm_input)

        # Split up paired tensors into individual time steps.
        trconv_input = data_utils.split_time_step_pairs(lstm_output)

        outputs = [
            self._t2_output_cnn(trconv_input),
            self._rh2_output_cnn(trconv_input),
            self._wspd10_output_cnn(trconv_input),
            self._wdir10_output_cnn(trconv_input),
        ]
        return tf.concat(outputs, axis=-1)

import logging
from typing import Iterator, TypedDict, List, Callable, Literal, TypeAlias, Any
import dataclasses
import numpy as np
from usl_models.shared import pad_layers
import keras
from keras import layers
from keras.saving import register_keras_serializable
import keras_tuner
import tensorflow as tf

from usl_models.flood_physical_ml.constants import *
from usl_models.shared import keras_dataclasses
from usl_models.flood_physical_ml import customloss
from usl_models.flood_physical_ml.dataset import load_dataset_windowed_patches
from usl_models.flood_physical_ml.utils import get_gauss_legendre_IRK
import pathlib



Activation: TypeAlias = Literal["relu", "sigmoid", "tanh", "softmax", "linear"]
PadMode: TypeAlias = Literal["REFLECT", "CONSTANT"]
RESOLUTION = 2
REVOLVESTEPS = 30
EPS = 1e-7

@register_keras_serializable()
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

    def get_config(self):
        """Getcongif."""
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        """fromcongif."""
        return cls(**config)

class FloodModel:
    """Flood model class."""

    @keras_dataclasses.dataclass(kw_only=True)
    class Params(keras_dataclasses.Base):
        """Flood model hyperparameters."""

        lstm_units: int = 128
        lstm_kernel_size: int = 5
        lstm_dropout: float = 0.2
        lstm_recurrent_dropout: float = 0.2
        m_rainfall: int = 6
        n_flood_maps: int = 5
        num_features: int = 22
        pad_mode: PadMode = "REFLECT"
        # V3 architecture flags (all default False = original behaviour preserved)
        # v3.1: tile rain rate/cumul to spatiotemporal input
        use_rain_broadcast: bool = False
        # v3.2: dilated conv refinement before geo_cnn
        use_dilated_geo: bool = False
        # v3.3: global storm-intensity FC -> spatial bias
        use_storm_embed: bool = False
        # v3.4: 2x convs per upsample stage + skip from geo
        use_deep_decoder: bool = False
        # v6: replace decoder BN with GroupNorm (no train/eval gap)
        use_group_norm: bool = False
        # "v1" = log-depth hybrid (default) | "v3" = depth-weighted linear MSE
        loss_version: str = "v1"
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
                "use_rain_broadcast": self.use_rain_broadcast,
                "use_dilated_geo": self.use_dilated_geo,
                "use_storm_embed": self.use_storm_embed,
                "use_deep_decoder": self.use_deep_decoder,
                "use_group_norm": self.use_group_norm,
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
        self,
        params: Params | None = None,
        spatial_dims: tuple[int, int] | None = None,
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
    def get_hypermodel(
        cls, spatial_dims: tuple[int, int] | None = None, **kwargs
    ) -> keras_tuner.HyperModel:
        """Returns a hypermodel function for use with Keras Tuner.

        Args:
            spatial_dims: Optional (H, W) for the model input size.
            **kwargs: hp_options dict where keys are parameter names
                and values are lists of possible choices.
        """

        def hypermodel(hp: keras_tuner.HyperParameters):
            hp_kwargs = {k: hp.Choice(k, v) for k, v in kwargs.items()}
            params = cls.Params(**hp_kwargs)
            return cls(params=params, spatial_dims=spatial_dims)._model

        return hypermodel

    def _build_model(self) -> keras.Model:
        model = FloodPhysicConvLSTM(self._params, spatial_dims=self._spatial_dims)

        loss_fn = customloss.physical_loss

        model.compile(
            optimizer=self._params.optimizer,
            loss=loss_fn,
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
        validation_steps: int | None = None,
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
            validation_steps=validation_steps,
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

@register_keras_serializable()
class FloodPhysicConvLSTM(keras.Model):
    """Flood Physical ConvLSTM model.

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
        spatial_dims: tuple[int, int] = (MAP_HEIGHT, MAP_WIDTH),
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
        self._sampling_prob = tf.Variable(0.0, trainable=False, name="sampling_prob")
        self.A, self.B, _ = get_gauss_legendre_IRK(MID_STEPS)

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.momentum_loss_tracker = tf.keras.metrics.Mean(name="momentum_loss")
        self.mass_loss_tracker = tf.keras.metrics.Mean(name="mass_loss")
        self.depth_loss_tracker = tf.keras.metrics.Mean(name="depth_loss")
        

        # index map:
        # x, 3, x
        # 2, x, 0
        # x, 1, x

        # used to get the adjacent outflow towards the center piexel
        core = np.zeros((3, 3, 4, 4), dtype=float)
        core[1, 2, 2, 0] = 1
        core[2, 1, 3, 1] = 1
        core[1, 0, 0, 2] = 1
        core[0, 1, 1, 3] = 1

        self.core_layer = layers.Conv2D(filters=4, kernel_size=(3, 3), padding='valid', use_bias=False, trainable=False)
        self.core_layer.build(input_shape=(None, None, None, 4))
        self.core_layer.set_weights([core])

        # the core used to get all adjacent flood velocity
        v_core = np.zeros((3, 3, 2, 2), dtype=float)
        v_core[1, 0, 0, 0] = 1
        v_core[0, 1, 1, 1] = 1

        self.v_core_layer = layers.Conv2D(filters=2, kernel_size=(3, 3), padding='valid', use_bias=False, trainable=False)
        self.v_core_layer.build(input_shape=(None, None, None, 2))
        self.v_core_layer.set_weights([v_core])

        ###################################################################################################
        ##################################### Model Structure #############################################
        ###################################################################################################

        ###################################### water + building + dem (water_level) ######################################
        # how to maintain the physical meaning? 1. normalize in hidden layers,  or 2. preprocess to get slop

        # try method 1: normalize in hidden layers
        # water size: (None, T, H, W, 1) for each time step (None, H, W, 1) 
        # building size: (None, T, H, W, 1) for each time step (None, H, W, 1)
        # dem size: (None, T, H, W, 1) for each time step (None, H, W, 1)
        # output size: (None, T, H, W, W_L_CONV2_HIDDEN_DIM)

        water_level_input_shape = (
            None, # time dimension
            self._spatial_height,
            self._spatial_width,
            3,
        )

        self.water_level_feature_encoder = keras.Sequential(
            [
                layers.InputLayer(water_level_input_shape),
                layers.TimeDistributed(pad_layers.Pad2D(W_L_CONV1_PAD_SIZE, mode="REFLECT")),
                layers.TimeDistributed(
                    layers.Conv2D(
                        filters=W_L_CONV1_FEATURE_DIM,
                        kernel_size=W_L_CONV1_FEATUER_KERNEL_SIZE,
                        strides=W_L_CONV1_FEATUER_KERNEL_STRIDE,
                        padding="valid"
                    )
                ), 
                layers.TimeDistributed(pad_layers.Pad2D(W_L_CONV2_PAD_SIZE, mode="REFLECT")),
                layers.TimeDistributed(
                    layers.Conv2D(
                        filters=W_L_CONV2_HIDDEN_DIM,
                        kernel_size=W_L_CONV2_KERNEL_SIZE,
                        strides=W_L_CONV2_KERNEL_STRIDE,
                        padding="valid"
                    )
                ), 
                layers.LayerNormalization(axis=-1),
                layers.Activation(W_L_CONV2_ACTIVATION)
            ],
            name="water_level_feature_encoder"
        )

        ###################################### water + friction #####################################
        # water size: (None, T, H, W, 1) for each time step (None, H, W, 1)
        # friction size: (None, T, H, W, 1) for each time step (None, H, W, 1)
        # output size: (None, T, H, W, FRICTION_CONV2_HIDDEN_DIM)
        self.friction_encoder_conv = keras.Sequential(
            [
                layers.InputLayer((None, self._spatial_height, self._spatial_width, 2)),
                layers.TimeDistributed(pad_layers.Pad2D(FRICTION_CONV1_PAD_SIZE)),
                layers.TimeDistributed(
                    layers.Conv2D(
                        filters=FRICTION_CONV1_HIDDEN_DIM,
                        kernel_size=FRICTION_CONV1_KERNEL_SIZE,
                        strides=FRICTION_CONV1_KERNEL_STRIDE,
                        padding="valid"
                    )
                ),
                layers.TimeDistributed(pad_layers.Pad2D(FRICTION_CONV2_PAD_SIZE)),
                layers.TimeDistributed(
                    layers.Conv2D(
                        filters=FRICTION_CONV2_HIDDEN_DIM,
                        kernel_size=FRICTION_CONV2_KERNEL_SIZE,
                        strides=FRICTION_CONV2_KERNEL_STRIDE,
                        padding="valid"
                    )
                ),
                layers.LayerNormalization(axis=-1),
                layers.Activation(FRICTION_CONV2_ACTIVATION)
            ],
            name="friction_encoder_conv"
        )

        ###################################### water + infiltration ###############################
        # water size: (None, T, H, W, 1) for each time step (None, H, W, 1)
        # infiltration size: (None, T, H, W, 4) for each time step (None, H, W, 4)
        # output size: (None, T, H, W, INFILTRATION_CONV2_HIDDEN_DIM)
        self.infiltration_encoder_conv = keras.Sequential(
            [
                layers.InputLayer((None, self._spatial_height, self._spatial_width, 5)),
                layers.TimeDistributed(pad_layers.Pad2D(INFILTRATION_CONV1_PAD_SIZE)),
                layers.TimeDistributed(
                    layers.Conv2D(
                        filters=INFILTRATION_CONV1_HIDDEN_DIM,
                        kernel_size=INFILTRATION_CONV1_KERNEL_SIZE,
                        strides=INFILTRATION_CONV1_KERNEL_STRIDE,
                        padding="valid"
                    )
                ),
                layers.TimeDistributed(pad_layers.Pad2D(INFILTRATION_CONV2_PAD_SIZE)),
                layers.TimeDistributed(
                    layers.Conv2D(
                        filters=INFILTRATION_CONV2_HIDDEN_DIM,
                        kernel_size=INFILTRATION_CONV2_KERNEL_SIZE,
                        strides=INFILTRATION_CONV2_KERNEL_STRIDE,
                        padding="valid"
                    )
                ),
                layers.LayerNormalization(axis=-1),
                layers.Activation(INFILTRATION_CONV2_ACTIVATION),
            ],
            name = "infiltration_encoder_conv"
        )

        ########################################## rainfall ################################################
        # rainfall size: (None, T, M)
        # output size: (None, T, RAINFALL_DENSE2_HIDDEN_DIM)
        self.rainfall_feature_encoder = keras.Sequential(
            [
                layers.InputLayer((None, M_RAINFALL)),
                layers.TimeDistributed(
                    layers.Dense(RAINFALL_DENSE1_HIDDEN_DIM, activation=RAINFALL_DENSE1_ACTIVATION)
                ),
                layers.TimeDistributed(
                    layers.Dense(RAINFALL_DENSE2_HIDDEN_DIM)
                ),
                layers.LayerNormalization(),
                layers.Activation(RAINFALL_DENSE2_ACTIVATION)
            ],
            name = "rainfall_feature_encoder"
        )
        
        ######################################################################
        ####################################merge############################## 

        self.merge_conv = keras.Sequential(
            [
                layers.InputLayer((
                    None, 
                    self._spatial_height, 
                    self._spatial_width, 
                    W_L_CONV2_HIDDEN_DIM+FRICTION_CONV2_HIDDEN_DIM+INFILTRATION_CONV2_HIDDEN_DIM+RAINFALL_DENSE2_HIDDEN_DIM)),
                layers.TimeDistributed(pad_layers.Pad2D(MERGE_CONV1_PAD_SIZE, mode="REFLECT")),
                layers.TimeDistributed(
                    layers.Conv2D(
                        filters=MERGE_CONV1_HIDDEN_DIM,
                        kernel_size=MERGE_CONV1_KERNEL_SIZE,
                        strides=MERGE_CONV1_KERNEL_STRIDE,
                        padding="valid"
                    )
                ),
                layers.LayerNormalization(axis=-1),
                layers.Activation(MERGE_CONV1_ACTIVATION)
            ],
            name = "features_merge_conv"
        )

        ######################################################################
        ####################################unet############################## 

        self.unet_encoder_1 = keras.Sequential(
            [
                layers.InputLayer((None, self._spatial_height, self._spatial_width, MERGE_CONV1_HIDDEN_DIM)),
                layers.TimeDistributed(pad_layers.Pad2D(UNET_ENCODER_1_PAD_SIZE, mode="REFLECT")),
                layers.TimeDistributed(
                    layers.Conv2D(
                        filters=UNET_ENCODER_1_HIDDEN_DIM,
                        kernel_size=UNET_ENCODER_1_KERNEL_SIZE,
                        strides=UNET_ENCODER_1_KERNEL_STRIDE,
                        padding="valid"
                    )
                ),
                layers.LayerNormalization(axis=-1),
                layers.Activation(UNET_ENCODER_1_ACTIVATION),
                layers.TimeDistributed(
                    layers.AveragePooling2D(
                        pool_size = UNET_ENCODER_1_AVEPOOLING_SIZE,
                        strides = UNET_ENCODER_1_AVEPOOLING_STRIDE,
                        padding="valid"
                    )
                )
            ]
        )

        self.unet_encoder_2 = keras.Sequential(
            [
                layers.InputLayer((None, self._spatial_height//2, self._spatial_width//2, UNET_ENCODER_1_HIDDEN_DIM)),
                layers.TimeDistributed(pad_layers.Pad2D(UNET_ENCODER_2_PAD_SIZE, mode="REFLECT")),
                layers.TimeDistributed(
                    layers.Conv2D(
                        filters=UNET_ENCODER_2_HIDDEN_DIM,
                        kernel_size=UNET_ENCODER_2_KERNEL_SIZE,
                        strides=UNET_ENCODER_2_KERNEL_STRIDE,
                        padding="valid"
                    )
                ),
                layers.LayerNormalization(axis=-1),
                layers.Activation(UNET_ENCODER_2_ACTIVATION),
                layers.TimeDistributed(
                    layers.AveragePooling2D(
                        pool_size=UNET_ENCODER_2_AVEPOOLING_SIZE,
                        strides=UNET_ENCODER_2_AVEPOOLING_STRIDE,
                        padding="valid"
                    )
                )
            ]
        )

        self.unet_encoder_3 = keras.Sequential(
            [
                layers.InputLayer((None, self._spatial_height//4, self._spatial_width//4, UNET_ENCODER_2_HIDDEN_DIM)),
                layers.TimeDistributed(pad_layers.Pad2D(UNET_ENCODER_3_PAD_SIZE, mode="REFLECT")),
                layers.TimeDistributed(
                    layers.Conv2D(
                        filters=UNET_ENCODER_3_HIDDEN_DIM,
                        kernel_size=UNET_ENCODER_3_KERNEL_SIZE,
                        strides=UNET_ENCODER_3_KERNEL_STRIDE,
                        padding="valid"
                    )
                ),
                layers.LayerNormalization(axis=-1),
                layers.Activation(UNET_ENCODER_3_ACTIVATION),
                layers.TimeDistributed(
                    layers.AveragePooling2D(
                        pool_size=UNET_ENCODER_3_AVEPOOLING_SIZE,
                        strides=UNET_ENCODER_3_AVEPOOLING_STRIDE,
                        padding="valid"
                    )
                )
            ]
        )
        ####################upsampling and decoder#######################
        # unet_decoder1
        self.unet_decoder1_up = layers.TimeDistributed(layers.UpSampling2D(size=UNET_DECODER_1_UP_SIZE, interpolation=UNET_DECODER_1_UP_INTERP), name = "unet_decoder1_upsample")
        self.unet_decoder1_conv = keras.Sequential(
            [
                layers.InputLayer((None, self._spatial_height//4, self._spatial_width//4, UNET_ENCODER_3_HIDDEN_DIM+UNET_ENCODER_2_HIDDEN_DIM)),
                layers.TimeDistributed(pad_layers.Pad2D(UNET_DECODER_1_CONV_PAD_SIZE)),
                layers.TimeDistributed(
                    layers.Conv2D(
                        filters=UNET_DECODER_1_CONV_HIDDEN_DIM,
                        kernel_size=UNET_DECODER_1_CONV_KERNEL_SIZE,
                        strides=UNET_DECODER_1_CONV_KERNEL_STRIDE,
                        padding="valid"
                    )
                ),
                layers.LayerNormalization(axis=-1),
                layers.Activation(UNET_DECODER_1_CONV_ACTIVATION)
            ],
            name = "unet_decoder_1_conv"
        )

        # unet_decoder2
        self.unet_decoder2_up = layers.TimeDistributed(layers.UpSampling2D(size=UNET_DECODER_2_UP_SIZE, interpolation=UNET_DECODER_2_UP_INTERP), name = "unet_decoder2_upsample")
        self.unet_decoder2_conv = keras.Sequential(
            [
                layers.InputLayer((None, self._spatial_height//2, self._spatial_width//2, UNET_DECODER_1_CONV_HIDDEN_DIM+UNET_ENCODER_1_HIDDEN_DIM)),
                layers.TimeDistributed(pad_layers.Pad2D(UNET_DECODER_2_CONV_PAD_SIZE)),
                layers.TimeDistributed(
                    layers.Conv2D(
                        filters=UNET_DECODER_2_CONV_HIDDEN_DIM,
                        kernel_size=UNET_DECODER_2_CONV_KERNEL_SIZE,
                        strides=UNET_DECODER_2_CONV_KERNEL_STRIDE,
                        padding="valid"
                    )
                ),
                layers.LayerNormalization(axis=-1),
                layers.Activation(UNET_DECODER_2_CONV_ACTIVATION)
            ],
            name = "unet_decoder_2_conv"
        )

        # unet_decoder3

        self.unet_decoder3_up = layers.TimeDistributed(layers.UpSampling2D(size=UNET_DECODER_3_UP_SIZE, interpolation=UNET_DECODER_3_UP_INTERP), name = "unet_decoder3_upsample")
        self.unet_decoder3_conv = keras.Sequential(
            [
                layers.InputLayer((None, self._spatial_height, self._spatial_width, UNET_DECODER_2_CONV_HIDDEN_DIM+MERGE_CONV1_HIDDEN_DIM)),
                layers.TimeDistributed(pad_layers.Pad2D(UNET_DECODER_3_CONV_PAD_SIZE)),
                layers.TimeDistributed(
                    layers.Conv2D(
                        filters=UNET_DECODER_3_CONV_HIDDEN_DIM,
                        kernel_size=UNET_DECODER_3_CONV_KERNEL_SIZE,
                        strides=UNET_DECODER_3_CONV_KERNEL_STRIDE,
                        padding="valid"
                    )
                ),
                layers.LayerNormalization(axis=-1),
                layers.Activation(UNET_DECODER_3_CONV_ACTIVATION)
            ],
            name = "unet_decoder_3_conv"
        )
        
        ###################################################################################################################
        ####################################################ConvLSTM#######################################################
        ###################################################################################################################
        
        self.conv_lstm = keras.Sequential(
            [
                layers.InputLayer((None, self._spatial_height, self._spatial_width, UNET_DECODER_3_CONV_HIDDEN_DIM)),
                layers.ConvLSTM2D(
                    CONVLSTM_1_LSTM_UNITS,
                    CONVLSTM_1_KERNEL_SIZE,
                    strides = CONVLSTM_1_STRIDES,
                    padding = CONVLSTM_1_PADDING,
                    dropout = CONVLSTM_1_LSTM_DROPOUT,
                    recurrent_dropout = CONVLSTM_1_RECURRENT_DROPOUT,
                    return_sequences = True,
                ),
                layers.LayerNormalization(),
                layers.Activation(CONVLSTM_1_ACTIVATION),
                
                layers.ConvLSTM2D(
                    CONVLSTM_2_LSTM_UNITS,
                    CONVLSTM_2_KERNEL_SIZE,
                    strides = CONVLSTM_2_STRIDES,
                    padding = CONVLSTM_2_PADDING,
                    dropout = CONVLSTM_2_LSTM_DROPOUT,
                    recurrent_dropout = CONVLSTM_2_RECURRENT_DROPOUT,
                    return_sequences = False
                ),
                layers.LayerNormalization(),
                layers.Activation(CONVLSTM_2_ACTIVATION)
            ],
            name = "conv_lstm"
        )

        ########################################################################################################
        ###########################multihead prediction#########################################################
        ########################################################################################################

        # infiltration prediction: OUT: [B, H, W, 1]
        self.infiltration_pred_conv = keras.Sequential(
            [
                layers.InputLayer((self._spatial_height, self._spatial_width, CONVLSTM_2_LSTM_UNITS)),
                pad_layers.Pad2D(INFILTRATION_PRED_CONV1_PAD_SIZE),
                layers.Conv2D(
                    filters = INFILTRATION_PRED_CONV1_HIDDEN_DIM,
                    kernel_size = INFILTRATION_PRED_CONV1_KERNEL_SIZE,
                    strides = INFILTRATION_PRED_CONV1_STRIDE,
                    padding = "valid",
                    activation = INFILTRATION_PRED_CONV1_ACTIVATION
                ),
                layers.Conv2D(
                    filters = INFILTRATION_PRED_CONV2_OUTPUT_DIM,
                    kernel_size = INFILTRATION_PRED_CONV2_KERNEL_SIZE,
                    strides = INFILTRATION_PRED_CONV2_STRIDE,
                    padding = "same",
                    activation = INFILTRATION_PRED_CONV2_ACTIVATION
                )
            ],
            name="infiltration_pred_conv"
        )

        # u prediction: OUT: [B, H, W, 12]
        self.u_pred_conv = keras.Sequential(
            [
                layers.InputLayer((self._spatial_height, self._spatial_width, CONVLSTM_2_LSTM_UNITS)),
                pad_layers.Pad2D(U_PRED_CONV1_PAD_SIZE),
                layers.Conv2D(
                    filters = U_PRED_CONV1_HIDDEN_DIM,
                    kernel_size = U_PRED_CONV1_KERNEL_SIZE,
                    strides = U_PRED_CONV1_STRIDE,
                    padding = "valid",
                    activation = U_PRED_CONV1_ACTIVATION
                ),
                layers.Conv2D(
                    filters = U_PRED_CONV2_OUTPUT_DIM,
                    kernel_size = U_PRED_CONV2_KERNEL_SIZE,
                    strides = U_PRED_CONV2_STRIDE,
                    padding = "same",
                    activation = U_PRED_CONV2_ACTIVATION
                )
            ],
            name="u_pred_conv"
        )

        # v prediction: OUT: [B, H, W, 12]
        self.v_pred_conv = keras.Sequential(
            [
                layers.InputLayer((self._spatial_height, self._spatial_width, CONVLSTM_2_LSTM_UNITS)),
                pad_layers.Pad2D(V_PRED_CONV1_PAD_SIZE),
                layers.Conv2D(
                    filters = V_PRED_CONV1_HIDDEN_DIM,
                    kernel_size = V_PRED_CONV1_KERNEL_SIZE,
                    strides = V_PRED_CONV1_STRIDE,
                    padding = "valid",
                    activation = V_PRED_CONV1_ACTIVATION
                ),
                layers.Conv2D(
                    filters = V_PRED_CONV2_OUTPUT_DIM,
                    kernel_size = V_PRED_CONV2_KERNEL_SIZE,
                    strides = V_PRED_CONV2_STRIDE,
                    padding = "same",
                    activation = V_PRED_CONV2_ACTIVATION
                )
            ],
            name="v_pred_conv"
        )

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.momentum_loss_tracker,
            self.mass_loss_tracker,
            self.depth_loss_tracker,
        ]
    
    def call(self, input: FloodModel.Input) -> tf.Tensor:
        """Makes a single forward pass on a batch of data.

        The forward pass represents a single prediction on an input batch
        (i.e., a single flood map). This functions implements the logic of the
        internal ConvLSTM and ignores autoregressive steps.

        Args:
            input: Dictionary containing:
              - spatiotemporal (Historical Flooding Map): Flood maps tensor of shape [B, n, H, W, 1].
              - geospatial: Geospatial tensor of shape [B, H, W, f].
              - temporal (Rainfall): Rainfall windows tensor of shape [B, n, m].

        Returns:
            The flood map prediction. A tensor of shape [B, H, W, 1].
        """
        spatiotemporal = input["spatiotemporal"]
        geospatial = input["geospatial"]
        temporal = input["temporal"]

        T = spatiotemporal.shape[1]
        H = spatiotemporal.shape[2]
        W = spatiotemporal.shape[3]

        water_depth = spatiotemporal # current depth [B, T, H, W, 1]
        building = geospatial[..., 2:3] * 4 # Building mask [B, T, H, W, 1]

        dem = geospatial[:, :, :, 0:1] - tf.reduce_min(geospatial[:, :, :, 0:1], axis=(1, 2, 3), keepdims=True)



        friction_factor = geospatial[..., 3:4] * 0.015 + 0.02
        infiltration_factor = geospatial[..., 4:8]

        dem = tf.tile(dem[:, tf.newaxis, ...], [1, T, 1, 1, 1])
        building = tf.tile(building[:, tf.newaxis, ...], [1, T, 1, 1, 1])
        friction_factor = tf.tile(friction_factor[:, tf.newaxis, ...], [1, T, 1, 1, 1])
        infiltration_factor = tf.tile(infiltration_factor[:, tf.newaxis, ...], [1, T, 1, 1, 1])

        water_level_feature = tf.concat([water_depth, building, dem], axis=-1) # [B, T, H, W, 3]
        friction_feature = tf.concat([water_depth, friction_factor], axis=-1) # [B, T, H, W, 2]
        infiltration_feature = tf.concat([water_depth, infiltration_factor], axis=-1) # [B, T, H, W, 5]
        rainfall = temporal # [B, T, 6]
        most_recent_rain_rate = rainfall[:, -1, -1] # [B, ]

        ##############################################################
        ################ initial features encoding ###################
        ##############################################################

        # water level feature encoding
        dem_embedding = self.water_level_feature_encoder(water_level_feature) # [B, T, H, W, W_L_CONV2_HIDDEN_DIM]
        
        # friction encoding
        friction_embedding = self.friction_encoder_conv(friction_feature) # [B, T, H, W, FRICTION_CONV2_HIDDEN_DIM]
        
        # infiltration encoding
        infiltration_embedding = self.infiltration_encoder_conv(infiltration_feature) # [B, T, H, W, INFILTRATION_CONV2_HIDDEN_DIM]
        
        # rainfall encoding
        rainfall_embedding = self.rainfall_feature_encoder(rainfall) # [B, T, RAINFALL_DENSE2_HIDDEN_DIM]
        # broadcast rainfall into shape [B, T, H, W, RAINFALL_DENSE2_HIDDEN_DIM]
        rainfall_embedding = rainfall_embedding[:, :, tf.newaxis, tf.newaxis, :]
        rainfall_embedding = tf.tile(rainfall_embedding, [1, 1, H, W, 1]) # [B, T, H, W, RAINFALL_DENSE2_HIDDEN_DIM]

        ##############################################################
        ############################ merge ############################
        ##############################################################
        merged_embedding = tf.concat([dem_embedding, friction_embedding, infiltration_embedding, rainfall_embedding], axis=-1)
        merged_embedding = self.merge_conv(merged_embedding) # [B, T, H, W, MERGE_CONV1_HIDDEN_DIM]

        
        ##############################################################
        ############################ unet ############################
        ##############################################################
        encoder1_embedding = self.unet_encoder_1(merged_embedding) # [B, T, H//2, W//2, UNET_ENCODER_1_HIDDEN_DIM]
        encoder2_embedding = self.unet_encoder_2(encoder1_embedding) # [B, T, H//4, W//4, UNET_ENCODER_2_HIDDEN_DIM]
        encoder3_embedding = self.unet_encoder_3(encoder2_embedding) # [B, T, H//8, W//8, UNET_ENCODER_3_HIDDEN_DIM]

        encoder3_embedding_up = self.unet_decoder1_up(encoder3_embedding) #[B, T, H//4, W//4, UNET_ENCODER_3_HIDDEN_DIM]
        decoder1_input = tf.concat([encoder2_embedding, encoder3_embedding_up], axis=-1) #[B, T, H//4, W//4, UNET_ENCODER_3_HIDDEN_DIM+UNET_ENCODER_2_HIDDEN_DIM]
        decoder1_output = self.unet_decoder1_conv(decoder1_input) # [B, T, H//4, W//4, UNET_DECODER_1_CONV_HIDDEN_DIM]

        decoder1_output_up = self.unet_decoder2_up(decoder1_output) # [B, T, H//2, W//2, UNET_DECODER_1_CONV_HIDDEN_DIM]
        decoder2_input = tf.concat([encoder1_embedding, decoder1_output_up], axis=-1) # [B, T, H//2, W//2, UNET_DECODER_1_CONV_HIDDEN_DIM+UNET_ENCODER_1_HIDDEN_DIM]
        decoder2_output = self.unet_decoder2_conv(decoder2_input) # [B, T, H//2, W//2, UNET_DECODER_2_CONV_HIDDEN_DIM]

        decoder2_output_up = self.unet_decoder3_up(decoder2_output) # [B, T, H, W, UNET_DECODER_2_CONV_HIDDEN_DIM]
        decoder3_input = tf.concat([merged_embedding, decoder2_output_up], axis=-1) # [B, T, H, W, UNET_DECODER_2_CONV_HIDDEN_DIM+MERGE_CONV1_HIDDEN_DIM]
        decoder3_output = self.unet_decoder3_conv(decoder3_input) # [B, T, H, W, UNET_DECODER_3_CONV_HIDDEN_DIM]


        ##############################################################
        ########################## ConvLSTM ##########################
        ##############################################################
        spatiotemporal_embedding = self.conv_lstm(decoder3_output) # [B, H, W, CONVLSTM_2_LSTM_UNITS]

        ##############################################################
        ######################## Prediction ##########################
        ##############################################################

        # infiltration prediction
        infil_pred = self.infiltration_pred_conv(spatiotemporal_embedding) # [B, H, W, 1] Unit: m/s

        # u prediction
        u_pred = self.u_pred_conv(spatiotemporal_embedding) # [B, H, W, U_PRED_CONV2_OUTPUT_DIM]

        # v prediction
        v_pred = self.v_pred_conv(spatiotemporal_embedding) # [B, H, W, V_PRED_CONV2_OUTPUT_DIM] Unit: m/s
        velocity_pred = tf.stack([u_pred, v_pred], axis=-1) # [B, H, W, U_PRED_CONV2_OUTPUT_DIM, 2], u to right, v to down. Unit: m/s
        velocity_pred = tf.transpose(velocity_pred, perm=[0, 3, 1, 2, 4]) # [B, U_PRED_CONV2_OUTPUT_DIM, H, W, 2]
        ##############################################################
        ###################### Calculate Depth #######################
        ##############################################################

        step_time = 300 / (MID_STEPS+1)
        # rainfall volume per middle time step
        rain_volume = most_recent_rain_rate * step_time * RESOLUTION**2 # (B, )
        rain_volume = rain_volume[:, tf.newaxis, tf.newaxis, tf.newaxis] # (B, 1, 1, 1)
        rain_volume = tf.tile(rain_volume, [1, H, W, 1]) # (B, H, W, 1)
        # infiltration volume per middle time step
        infiltration_volume = infil_pred * RESOLUTION**2 * step_time

        # calculate outflow velocity for the other two directions
        adj_velocity = tf.pad(velocity_pred, [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        adj_velocity = self.v_core_layer(adj_velocity)
        adj_velocity = tf.nn.relu(-1 * adj_velocity)

        # X 3 X 
        # 2 X 0
        # X 1 X
        out_flow_velocity = tf.concat([velocity_pred, adj_velocity], axis=-1) # [B, t, H, W, 4]
        
        current_depth = spatiotemporal[:, -1] # [B, H, W, 1]

        depth_list = []

        for i in range(1, MID_STEPS + 2):
            last_time_speed = out_flow_velocity[:, i-1] # [B, H, W, 4]
            current_speed = out_flow_velocity[:, i] # [B, H, W, 4]

            average_speed = (last_time_speed + current_speed) / 2 # [B, H, W, 4]

            outflow = average_speed * current_depth * RESOLUTION # [B, H, W, 4]

            inflow = tf.pad(outflow, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
            inflow = self.core_layer(inflow) # [B, H, W, 4]

            volume_change = tf.reduce_sum((inflow - outflow)*step_time, axis=-1, keepdims=True) + rain_volume - infiltration_volume # [B, H, W, 1]

            current_depth = tf.nn.relu(current_depth + volume_change/(RESOLUTION**2)) # [B, H, W, 1]

            depth_list.append(current_depth[:, tf.newaxis, ...])
        
        calculated_depth = tf.concat(depth_list, axis = 1) # [B, U_PRED_CONV2_OUTPUT_DIM, H, W, 1]

        return velocity_pred, calculated_depth # velocity [B, Mid-step+2, H, W, 2]; depth [B, Mid-step+1, H, W, 1]

    def call_n_modified(self, full_input: FloodModel.Input, n: int = 1) -> tf.Tensor:
        spatiotemporal = full_input["spatiotemporal"]
        geospatial = full_input["geospatial"]
        temporal = full_input["temporal"]

        batch_size = tf.shape(spatiotemporal)[0]

        cumul_F = tf.zeros(
                (batch_size, self._spatial_height, self._spatial_width, 1),
                dtype=tf.float32,
            )

        prediction = []

        has_temporal_per_step = len(temporal.shape) == 4
        

        for k in range(n):
            
            temporal_k = temporal[:, k] if has_temporal_per_step else temporal
            pred = self({
                "geospatial": geospatial,
                "temporal": temporal_k,
                "spatiotemporal": spatiotemporal
            })
            
            pred = tf.nn.relu(pred)
            # Green-Ampt infiltration correction
            # pred, cumul_F = self.green_ampt_gate(pred, geospatial, cumul_F)
            prediction.append(pred[:, tf.newaxis, :, :, :])

            spatiotemporal = tf.concat([
                spatiotemporal[:, 1:, :, :, :], 
                pred[:, tf.newaxis, :, :, :]
            ], axis=1,)
        result = tf.concat(prediction, axis=1)
        return result # velocity(B, mid-steps + 2, H, W, 2), depth(B, mid-steps + 1, H, W, 1)


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
        N, M = self._params.n_flood_maps, self._params.m_rainfall
        T_MAX = constants.MAX_RAINFALL_DURATION
        H, W = self._spatial_height, self._spatial_width

        tf.ensure_shape(spatiotemporal, (B, N, H, W, C))
        tf.ensure_shape(geospatial, (B, H, W, F))
        tf.ensure_shape(temporal, (B, T_MAX, M))

        # This array stores the n predictions.
        predictions = tf.TensorArray(tf.float32, size=n)

        # Cumulative infiltration F — resets to zero at t=0 each simulation.
        cumul_F = tf.zeros((B, H, W, 1), dtype=tf.float32)

        # We use 1-indexing for simplicity. Time step t represents the t-th flood
        # prediction.
        # TODO: consider using tf.while_loop to support serializing this function.
        for t in range(1, n + 1):
            input = FloodModel.Input(
                geospatial=geospatial,
                temporal=self._get_temporal_window(temporal, t, N),
                spatiotemporal=spatiotemporal,
            )
            prediction = self.call(input)
            prediction = tf.nn.relu(prediction)  # enforce non-negative depth

            # Green-Ampt infiltration correction — subtracts what soil absorbs
            # prediction, cumul_F = self.green_ampt_gate(prediction, geospatial, cumul_F)

            predictions = predictions.write(t - 1, prediction)

            # Append corrected prediction along time axis, drop the first.
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

    @staticmethod
    def _normalize_labels(y):
        """Normalize labels to (B, K, H, W, 1) regardless of input shape."""
        if len(y.shape) == 3:
            # (B, H, W) → single-step
            return tf.expand_dims(tf.expand_dims(y, axis=1), axis=-1)
        elif len(y.shape) == 4:
            if y.shape[-1] == 1:
                # (B, H, W, 1) → single-step
                return tf.expand_dims(y, axis=1)
            else:
                # (B, K, H, W) → multi-step
                return tf.expand_dims(y, axis=-1)
        elif len(y.shape) == 5:
            # (B, K, H, W, 1) → already correct
            return y
        else:
            raise ValueError(f"Unexpected y shape: {y.shape}")

    def train_step(self, data):
        """Autoregressive unrolling train step.

        Unrolls the model for K steps (K = number of future labels):
        - Step 0: predict from the dataset's spatiotemporal input (teacher-forced)
        - Steps 1..K-1: feed the model's own prediction back as input

        The loss includes:
        - Per-step flood_weighted_mse (via compiled loss)
        - Mass preservation penalty (mean depth must match GT)
        - Time-step weighting (later steps weighted more heavily)

        With K=1 (single-step labels), this reduces to standard single-step
        training with the mass preservation bonus.
        """
        x, y = data

        geospatial = x["geospatial"]
        temporal = x["temporal"]
        spatiotemporal = x["spatiotemporal"]

        Valid_Mask = geospatial[..., 1:2]
        DEM = geospatial[..., 0:1]
        Building = geospatial[..., 2:3]
        N = geospatial[..., 3:4] * 0.015 + 0.02
        last_final_velocity = None

        ####################################################################
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #removed: the flood depth should be infered from the beginning of the 
        #rainfall under the condition when no perfect historical depth info, 
        #instead of creating zeros
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ####################################################################
        # # Scheduled sampling probability.
        # # With temporal_feature_version=2, temporal channels already encode
        # # storm progression (cumulative/time channels), so additional random
        # # zero-context corruption is disabled by default.
        # # Curriculum AR sampling: ramp up zero-context fraction over training.
        # # Early epochs: learn flood physics from GT context.
        # # Later epochs: force geospatial-only routing for cold-start.
        # _ar_step = tf.cast(self.optimizer.iterations, tf.float32)
        # _ar_prob = tf.minimum(0.20, 0.0 + _ar_step / 50000.0 * 0.20)
        # spatiotemporal = tf.cond(
        #     tf.random.uniform(()) < _ar_prob,
        #     lambda: tf.zeros_like(spatiotemporal),
        #     lambda: spatiotemporal,
        # )
        ####################################################################
        ####################################################################

        y_steps = self._normalize_labels(y)
        # Unstack along step axis → Python list with static length K.
        # Avoids tf.while_loop dynamic-shape XLA issues.
        

        y_step_list = tf.unstack(y_steps, axis=1)
        # if len(y_step_list) > 1:
        #     varied_future_steps = np.random.randint(1, len(y_step_list))
        #     y_step_list = y_step_list[0:varied_future_steps]
        K = len(y_step_list)
        print(K)
        has_temporal_per_step = len(temporal.shape) == 4

        with tf.GradientTape() as tape:
            batch_size = tf.shape(spatiotemporal)[0]

            total_loss = tf.constant(0.0)
            momentum_loss = tf.constant(0.0)
            mass_loss = tf.constant(0.0)
            depth_loss = tf.constant(0.0)

            st = spatiotemporal
            last_pred = tf.zeros(
                (batch_size, self._spatial_height, self._spatial_width, 1),
                dtype=tf.float32,
            )
            for k, yk in enumerate(y_step_list):
                temporal_k = temporal[:, k] if has_temporal_per_step else temporal
                pred = self(
                    {
                        "geospatial": geospatial,
                        "temporal": temporal_k,
                        "spatiotemporal": st,
                    },
                    training=True,
                )

                step_loss, step_momentum, step_mass, step_depth = customloss.physical_loss(
                    velocity_sequence = pred[0],
                    depth_sequence = pred[1],
                    A = self.A,
                    B = self.B,
                    DEM = DEM,
                    Building = Building,
                    N = N,
                    label = yk,
                    last_final_velocity = last_final_velocity,
                    Valid_Mask = Valid_Mask
                )

                # ##############################################################
                # # Mass preservation: penalise mismatch in mean depth
                # # Need to think about
                # ##############################################################
                # pred_mass = tf.reduce_mean(pred, axis=[1, 2, 3])
                # gt_mass = tf.reduce_mean(yk, axis=[1, 2, 3])
                # mass_loss = tf.reduce_mean(tf.square(pred_mass - gt_mass))

                # ##############################################################
                # Later steps matter more (error should not accumulate)
                # # Need to think about
                # ##############################################################
                # time_weight = 1.0 + 0.3 * k
                # total_loss += time_weight * (step_loss + 0.2 * mass_loss)
                
                total_loss += step_loss
                momentum_loss += step_momentum
                mass_loss += step_mass
                depth_loss += step_depth

                # Feed corrected prediction back: clip at 4.0m to match GT depth cap
                # (was 2.5m which caused hard ceiling — pred never learned >2.5m)
                # fb = tf.minimum(tf.stop_gradient(pred), 4.0)
                fb = tf.stop_gradient(pred[1][:, -1])

                # Feedback noise: teaches robustness to imperfect context
                # fb = fb + tf.random.normal(tf.shape(fb), stddev=0.02)

                # fb = tf.nn.relu(fb)  # keep non-negative after noise
                st = tf.concat(
                    [st[:, 1:, :, :, :], fb[:, tf.newaxis, :, :, :]],
                    axis=1,
                )
                last_pred = pred[1][:, -1]
                last_final_velocity = pred[0][:, -1]



            total_loss = total_loss / K
            momentum_loss = momentum_loss / K
            mass_loss = mass_loss / K
            depth_loss = depth_loss / K

            # arrival_time_loss disabled: log1p gradient explosion when depth > 0.5m
            # (sigmoid saturates -> grad = -1/1e-8 = -1e8 per step, NaN over K=7)
            # Fix: rework with hard soft-arrival using clip, not log-space survival

            # Add regularization losses once (not inside loop)
            if self.losses:
                total_loss += tf.add_n(self.losses)

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Metrics on last-step prediction
        self.loss_tracker.update_state(total_loss)
        self.momentum_loss_tracker.update_state(momentum_loss)
        self.mass_loss_tracker.update_state(mass_loss)
        self.depth_loss_tracker.update_state(depth_loss)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """Autoregressive unrolling validation step (mirrors train_step)."""
        x, y = data

        geospatial = x["geospatial"]
        temporal = x["temporal"]
        spatiotemporal = x["spatiotemporal"]

        Valid_Mask = geospatial[..., 1:2]
        DEM = geospatial[..., 0:1]
        Building = geospatial[..., 2:3]
        N = geospatial[..., 3:4] * 0.015 + 0.02
        last_final_velocity = None

        y_steps = self._normalize_labels(y)
        # Unstack along step axis → Python list with static length K.
        # Avoids tf.while_loop which causes XLA tracing errors with training=False.
        y_step_list = tf.unstack(y_steps, axis=1)
        K = len(y_step_list)
        has_temporal_per_step = len(temporal.shape) == 4

        batch_size = tf.shape(spatiotemporal)[0]

        total_loss = tf.constant(0.0)
        momentum_loss = tf.constant(0.0)
        mass_loss = tf.constant(0.0)
        depth_loss = tf.constant(0.0)

        st = spatiotemporal
        last_pred = tf.zeros(
            (batch_size, self._spatial_height, self._spatial_width, 1),
            dtype=tf.float32,
        )

        for k, yk in enumerate(y_step_list):
            temporal_k = temporal[:, k] if has_temporal_per_step else temporal
            pred = self(
                {
                    "geospatial": geospatial,
                    "temporal": temporal_k,
                    "spatiotemporal": st,
                },
                training=False,
            )

            step_loss, step_momentum, step_mass, step_depth = customloss.physical_loss(
                    velocity_sequence = pred[0],
                    depth_sequence = pred[1],
                    A = self.A,
                    B = self.B,
                    DEM = DEM,
                    Building = Building,
                    N = N,
                    label = yk,
                    last_final_velocity = last_final_velocity,
                    Valid_Mask = Valid_Mask
                )

            # pred_mass = tf.reduce_mean(pred, axis=[1, 2, 3])
            # gt_mass = tf.reduce_mean(yk, axis=[1, 2, 3])
            # mass_loss = tf.reduce_mean(tf.square(pred_mass - gt_mass))

            # time_weight = 1.0 + 0.3 * k
            # total_loss += time_weight * (step_loss + 0.2 * mass_loss)
            total_loss += step_loss
            momentum_loss += step_momentum
            mass_loss += step_mass
            depth_loss += step_depth

            # Feed corrected prediction back: clip at 4.0m (matches train_step)
            # pred_fb = tf.minimum(pred, 4.0)
            fb = pred[1][:, -1]

            st = tf.concat(
                [st[:, 1:, :, :, :], fb[:, tf.newaxis, :, :, :]],
                axis=1,
            )
            last_pred = pred[1][:, -1]
            last_final_velocity = pred[0][:, -1]

        total_loss = total_loss / K
        momentum_loss = momentum_loss / K
        mass_loss = mass_loss / K
        depth_loss = depth_loss / K

        self.loss_tracker.update_state(total_loss)
        self.momentum_loss_tracker.update_state(momentum_loss)
        self.mass_loss_tracker.update_state(mass_loss)
        self.depth_loss_tracker.update_state(depth_loss)

        # Flooded-pixel MAE: MAE computed only where GT > 0.01m.
        # Overall MAE is dominated by ~90% dry pixels (always near zero), hiding
        # real flood prediction errors. This metric isolates the flood signal.
        gt_last = y_steps[:, -1]
        flooded_mask = tf.cast(gt_last > 0.01, tf.float32)
        flooded_mae = tf.reduce_sum(tf.abs(last_pred - gt_last) * flooded_mask) / (
            tf.reduce_sum(flooded_mask) + 1e-6
        )

        result = {m.name: m.result() for m in self.metrics}
        result["flooded_mae"] = flooded_mae
        return result

    def get_config(self):
        """Get_config."""
        return {
            "params": self._params.to_dict(),
            "spatial_dims": (self._spatial_height, self._spatial_width),
        }

    @classmethod
    def from_config(cls, config):
        """From_config."""
        return cls(
            params=FloodModel.Params.from_dict(config["params"]),
            spatial_dims=tuple(config["spatial_dims"]),
        )

if __name__ == "__main__":
    PATCH_SIZE = 256
    STEPS_PER_EPOCH = 2000
    EPOCHS = 10
    cosine_lr = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-6,
        decay_steps=STEPS_PER_EPOCH * EPOCHS,
        alpha=1e-6,
        warmup_target=2e-5,                         # ROUND 6: V3 R3 baseline LR (back to proven)
        warmup_steps=500,
    )
    params = FloodModel.Params(
        lstm_units=128,
        lstm_kernel_size=5,
        lstm_dropout=0.0,
        lstm_recurrent_dropout=0.0,
        n_flood_maps=N_FLOOD_MAPS,
        m_rainfall=M_RAINFALL,
        use_rain_broadcast=False,
        use_dilated_geo=False,
        use_storm_embed=False,
        use_deep_decoder=False,
        use_group_norm=False,
        loss_version="v1",                           # ROUND 6: V1 loss — only feedback clip changed
        optimizer=keras.optimizers.Adam(learning_rate=cosine_lr, global_clipnorm=1.0),
    )
    model = FloodModel(params=params, spatial_dims=(PATCH_SIZE, PATCH_SIZE))

    FILECACHE_DIR = pathlib.Path("/scratch/hw4402/climateiq_filecache_us")
    PATCH_STRIDE = 128
    BATCH_SIZE = 1   # reverted from 6 — batch 4 had 16x better MAE (0.012 vs 0.197)
    EPOCHS = 100  # ROUND 5 ABLATION — V3 loss only, validate before long run
    N_FLOOD_MAPS = 5
    M_RAINFALL = 6
    N_FUTURE_STEPS = 6  # 13 → 0 windows (last_t = T - n_future = 13 - 13 = 0); 12 gives last_t=1 per chunk
    DEPTH_CAP = 4.0  # metres — raised from 2.5; covers Atlanta peak (~4.4m capped at 4m)
                    # Manhattan 5-7m ponding in closed canyons above this are outliers
    DRY_TIMESTEP_FRACTION = 0.12  # calibrated from prior best run to avoid dry-step dominance

    old_sims = [
        # Atlanta (3 scenarios)
        "Atlanta-Atlanta_config/Rainfall_Data_1.txt",
        "Atlanta-Atlanta_config/Rainfall_Data_2.txt",
        "Atlanta-Atlanta_config/Rainfall_Data_5.txt",
        # Phoenix SM (4 scenarios)
        "Phoenix_SM-PHX_SM/Rainfall_Data_1.txt",
        "Phoenix_SM-PHX_SM/Rainfall_Data_5.txt",
        "Phoenix_SM-PHX_SM/Rainfall_Data_15.txt",
        "Phoenix_SM-PHX_SM/Rainfall_Data_16.txt",
        # Manhattan (8 scenarios; 7 and 15 reserved for test)
        "Manhattan-Manhattan_config/Rainfall_Data_1.txt",
        "Manhattan-Manhattan_config/Rainfall_Data_2.txt",
        "Manhattan-Manhattan_config/Rainfall_Data_5.txt",
        "Manhattan-Manhattan_config/Rainfall_Data_6.txt",
        "Manhattan-Manhattan_config/Rainfall_Data_10.txt",
        "Manhattan-Manhattan_config/Rainfall_Data_13.txt",
        "Manhattan-Manhattan_config/Rainfall_Data_16.txt",
        "Manhattan-Manhattan_config/Rainfall_Data_19.txt",
        # Phoenix PV (2 scenarios)
        "Phoenix_PV-PHX_PV/Rainfall_Data_1.txt",
        "Phoenix_PV-PHX_PV/Rainfall_Data_5.txt",]

    # ── New cities (never seen by model) ─────────────────────────────────
    new_sims = [
            # Denton TX (13 timesteps, 1 chunk each)
        "Denton_TX-Denton_config/Rainfall_Data_1.txt",
        "Denton_TX-Denton_config/Rainfall_Data_2.txt",
        "Denton_TX-Denton_config/Rainfall_Data_5.txt",
        # Lafayette LA (13 timesteps, 1 chunk each)
        "Lafayette_LA-Lafayette_config/Rainfall_Data_1.txt",
        "Lafayette_LA-Lafayette_config/Rainfall_Data_2.txt",
        "Lafayette_LA-Lafayette_config/Rainfall_Data_5.txt",
        # Boise ID (13 timesteps, 1 chunk each)
        "Boise_ID-Boise_config/Rainfall_Data_1.txt",
        "Boise_ID-Boise_config/Rainfall_Data_2.txt",
        "Boise_ID-Boise_config/Rainfall_Data_5.txt",
        # Navarre FL (13 timesteps, 1-2 chunks each)
        "Navarre_FL-Navarre_config/Rainfall_Data_1.txt",
        "Navarre_FL-Navarre_config/Rainfall_Data_2.txt",
        "Navarre_FL-Navarre_config/Rainfall_Data_5.txt",
        # New Orleans
        "New_Orleans-New_Orleans_config/Rainfall_Data_1.txt",
        "New_Orleans-New_Orleans_config/Rainfall_Data_2.txt",
        "New_Orleans-New_Orleans_config/Rainfall_Data_5.txt",]

    # BASELINE TRAINING: Only old cities for clean comparison
    # Filter to only existing sims
    old_sims = [s for s in old_sims if (FILECACHE_DIR / s).exists()]
    new_sims = [s for s in new_sims if (FILECACHE_DIR / s).exists()]
    sim_names = old_sims + new_sims  # All cities — learning new flow channels
    # print(f"BASELINE TRAINING: {len(sim_names)} old city simulations (batch_size={BATCH_SIZE})")
    # for s in sim_names:
    #     print(f"  {s}")

    split = "train"

    ds = load_dataset_windowed_patches(
        filecache_dir=FILECACHE_DIR,
        sim_names=sim_names,
        dataset_split="train",
        patch_size=PATCH_SIZE,
        stride=PATCH_STRIDE,
        batch_size=BATCH_SIZE,
        n_flood_maps=N_FLOOD_MAPS,
        m_rainfall=M_RAINFALL,
        max_patches_per_chunk=None,
        min_flood_fraction=0.12,    # ≥1% flooded pixels — excludes all-dry patches
        min_max_depth=0.05,         # lowered from 0.1 — include early onset patches
        min_label_max_depth=0.001,  # lowered from 0.01 — include very early flood onset
        n_future_steps=N_FUTURE_STEPS,
        dry_timestep_fraction=DRY_TIMESTEP_FRACTION if split == "train" else 0.0,
        shuffle=True,
        temporal_feature_version=2,
    )

    i = 0
    for d in ds:
        x, y = d
        geospatial = x["geospatial"]
        temporal = x["temporal"]
        spatiotemporal = x["spatiotemporal"]
        model._model(
            {
                "geospatial": geospatial,
                "temporal": temporal[:, 0],
                "spatiotemporal": spatiotemporal,
            },
        )
        print("done!")
        break
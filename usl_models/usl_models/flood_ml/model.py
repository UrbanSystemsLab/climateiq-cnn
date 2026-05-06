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

from usl_models.flood_ml import constants
from usl_models.shared import keras_dataclasses
from usl_models.flood_ml import customloss


Activation: TypeAlias = Literal["relu", "sigmoid", "tanh", "softmax", "linear"]
PadMode: TypeAlias = Literal["REFLECT", "CONSTANT"]


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
        model = FloodConvLSTM(self._params, spatial_dims=self._spatial_dims)
        loss_fn = (
            customloss.make_hybrid_loss_v3
            if getattr(self._params, "loss_version", "v1") == "v3"
            else customloss.make_hybrid_loss
        )
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


###############################################################################
#                       Custom Keras Model definitions
#
# The following are keras.Model class implementations for flood model
# architecture(s). They can be used within the generic FloodModel class above
# for training and evaluation, and only define basic components of the model,
# such as layers and the forward pass. While these models are callable, all
# data pre- and post-processing are expected to be handled externally.
###############################################################################


@register_keras_serializable()
class GreenAmptGate(keras.layers.Layer):
    """Physics-based Green-Ampt infiltration correction.

    No trainable parameters. Uses soil hydraulic properties pre-encoded in
    geospatial channels 4-7 by the data pipeline (feature_raster_transformers):

        ch4: K_s  / MAX_K_S_CM_HR   — hydraulic conductivity (cm/hr, normalised)
        ch5: ψ_f  / MAX_PSI_F_CM    — wetting front suction head (cm, normalised)
        ch6: θ_e  / MAX_THETA_E     — effective porosity (dimensionless, normalised)
        ch7: θ_i  / MAX_THETA_I     — initial effective saturation (normalised)

    Applied once per autoregressive timestep. Cumulative infiltration F is
    tracked as an external state variable (passed in/out of the call) so that
    it works correctly inside tf.while_loop without needing internal layer state.

    Green-Ampt formula per pixel per step:
        fc = K_s * (1 + ψ_f * Δθ / (F + ε))        [infiltration capacity, m/s]
        infil = min(fc * dt, available_water)         [actual infiltration, m]
        corrected_depth = relu(pred - infil)          [surface depth after loss]
        F_new = F + infil                             [updated cumulative state]

    Pixels with K_s = 0 (impervious: roads, buildings, water) are unaffected.
    """

    # Denormalisation constants — must match feature_raster_transformers.py
    MAX_K_S_CM_HR = 11.78
    MAX_PSI_F_CM = 20.88
    MAX_THETA_E = 0.486
    MAX_THETA_I = 0.99
    DT_S = 300.0  # timestep: 5 minutes in seconds

    def call(
        self,
        pred: tf.Tensor,
        geospatial: tf.Tensor,
        cumul_F: tf.Tensor,
    ):
        """Apply Green-Ampt infiltration correction.

        Args:
            pred:       (B, H, W, 1) predicted flood depth (m), already relu'd.
            geospatial: (B, H, W, 9) static features from the data pipeline.
            cumul_F:    (B, H, W, 1) cumulative infiltration depth (m) so far.

        Returns:
            corrected:    (B, H, W, 1) flood depth after infiltration removed.
            new_cumul_F:  (B, H, W, 1) updated cumulative infiltration.
        """
        # Denormalise soil parameters to SI units
        K_s = geospatial[:, :, :, 4:5] * (self.MAX_K_S_CM_HR * 0.01 / 3600.0)  # m/s
        psi_f = geospatial[:, :, :, 5:6] * (self.MAX_PSI_F_CM * 0.01)  # m
        theta_e = geospatial[:, :, :, 6:7] * self.MAX_THETA_E
        theta_i = geospatial[:, :, :, 7:8] * self.MAX_THETA_I

        # Moisture deficit following Ahmed's Green-Ampt convention (theta_s × Se)
        delta_theta = theta_e * theta_i

        # Green-Ampt infiltration capacity at current saturation state (m/s)
        fc = K_s * (1.0 + (psi_f * delta_theta) / (cumul_F + 1e-9))

        # Actual infiltration this timestep: bounded by available surface water
        actual_infil = tf.minimum(fc * self.DT_S, tf.nn.relu(pred))

        # Zero out impervious / no-soil pixels (K_s == 0 means no infiltration)
        green_mask = tf.cast(geospatial[:, :, :, 4:5] > 0, tf.float32)
        actual_infil = actual_infil * green_mask

        new_cumul_F = cumul_F + actual_infil
        corrected = tf.nn.relu(pred - actual_infil)
        return corrected, new_cumul_F

    def get_config(self):
        """Return Keras serialisation config (no extra params)."""
        return super().get_config()


@register_keras_serializable()
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
        self._sampling_prob = tf.Variable(0.0, trainable=False, name="sampling_prob")

        # CNN padding config
        K_PAD = 2  # 5x5 kernel means 2-pixel padding
        cnn_pad = (K_PAD, K_PAD)
        activation = "relu"

        # === Spatiotemporal CNN (split into 2 stages for skip connections) ===
        # V3: rain_broadcast -> 3-channel ST input (depth + rate + cumulative)
        st_in_channels = 3 if self._params.use_rain_broadcast else 1
        self._st_in_channels = st_in_channels
        # Stage 1: 2x downsample -> [B, N, H/2, W/2, 8]
        st_input_shape = (
            None,
            self._spatial_height,
            self._spatial_width,
            st_in_channels,
        )
        self.st_cnn_stage1 = keras.Sequential(
            [
                layers.InputLayer(st_input_shape),
                layers.TimeDistributed(pad_layers.Pad2D(cnn_pad, mode="REFLECT")),
                layers.TimeDistributed(
                    layers.Conv2D(
                        8, 5, strides=2, padding="valid", activation=activation
                    )
                ),
                layers.TimeDistributed(
                    layers.MaxPool2D(pool_size=2, strides=1, padding="same")
                ),
            ],
            name="st_cnn_stage1",
        )
        # Stage 2: 4x downsample -> [B, N, H/4, W/4, 16]
        half_h = self._spatial_height // 2
        half_w = self._spatial_width // 2
        self.st_cnn_stage2 = keras.Sequential(
            [
                layers.InputLayer((None, half_h, half_w, 8)),
                layers.TimeDistributed(pad_layers.Pad2D(cnn_pad, mode="REFLECT")),
                layers.TimeDistributed(
                    layers.Conv2D(
                        16, 5, strides=2, padding="valid", activation=activation
                    )
                ),
                layers.TimeDistributed(
                    layers.MaxPool2D(pool_size=2, strides=1, padding="same")
                ),
            ],
            name="st_cnn_stage2",
        )

        # === V3.2: Dilated conv refinement (optional, residual) ===
        # Expands effective receptive field to ~200m before standard geo_cnn,
        # giving model long-range context for flow routing.
        if self._params.use_dilated_geo:
            geo_in_shape = (
                self._spatial_height,
                self._spatial_width,
                constants.GEO_FEATURES,
            )
            self.geo_dilated = keras.Sequential(
                [
                    layers.InputLayer(geo_in_shape),
                    layers.Conv2D(
                        32, 3, padding="same", dilation_rate=2, activation="relu"
                    ),
                    layers.Conv2D(
                        32, 3, padding="same", dilation_rate=4, activation="relu"
                    ),
                    # Project back to GEO_FEATURES so we can add residually
                    layers.Conv2D(
                        constants.GEO_FEATURES, 3, padding="same", activation=None
                    ),
                ],
                name="geo_dilated",
            )
        else:
            self.geo_dilated = None

        # === V3.3: Global storm-intensity embedding (optional) ===
        # Scalar storm intensity (peak rain) → 32-dim embedding tiled spatially
        if self._params.use_storm_embed:
            self.storm_embed_dim = 32
            self.storm_embed = keras.Sequential(
                [
                    layers.InputLayer((self._params.m_rainfall,)),
                    layers.Dense(64, activation="relu"),
                    layers.Dense(self.storm_embed_dim, activation=None),
                ],
                name="storm_embed",
            )
        else:
            self.storm_embed = None
            self.storm_embed_dim = 0

        # === Geospatial CNN ===
        self.geo_cnn = keras.Sequential(
            [
                # Input shape: (height, width, channels)
                layers.InputLayer(
                    (self._spatial_height, self._spatial_width, constants.GEO_FEATURES)
                ),
                pad_layers.Pad2D(cnn_pad, mode="REFLECT"),
                layers.Conv2D(16, 5, strides=2, padding="valid", activation=activation),
                layers.MaxPool2D(pool_size=2, strides=1, padding="same"),
                pad_layers.Pad2D(cnn_pad, mode="REFLECT"),
                layers.Conv2D(64, 5, strides=2, padding="valid", activation=activation),
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
        conv_lstm_channels = 16 + 64 + self._params.m_rainfall + self.storm_embed_dim

        self.conv_lstm = keras.Sequential(
            [
                # Input shape: (time_steps, height, width, channels)
                layers.InputLayer(
                    (None, conv_lstm_height, conv_lstm_width, conv_lstm_channels)
                ),
                # First ConvLSTM: captures local flood propagation
                layers.ConvLSTM2D(
                    self._params.lstm_units,
                    self._params.lstm_kernel_size,
                    strides=1,
                    padding="same",
                    activation="tanh",
                    dropout=self._params.lstm_dropout,
                    recurrent_dropout=self._params.lstm_recurrent_dropout,
                    return_sequences=True,
                ),
                layers.BatchNormalization(),
                # Second ConvLSTM: integrates wider neighbourhood
                layers.ConvLSTM2D(
                    self._params.lstm_units,
                    self._params.lstm_kernel_size,
                    strides=1,
                    padding="same",
                    activation="tanh",
                    dropout=self._params.lstm_dropout,
                    recurrent_dropout=self._params.lstm_recurrent_dropout,
                    return_sequences=False,
                ),
            ],
            name="conv_lstm",
        )

        self.attention = SpatialAttention()  # attention after ConvLSTM

        # === Decoder layers (called explicitly for skip connections) ===
        # Bilinear upsample + Conv2D avoids checkerboard artifacts.
        self.decoder_up1 = layers.UpSampling2D(size=2, interpolation="bilinear")
        # After concat with skip1 (8ch), input channels = lstm_units + 8
        self.decoder_conv1 = layers.Conv2D(32, 3, padding="same", activation="relu")
        # v6: GroupNorm has no train/eval gap; BN running stats caused 2.5m ceiling
        if self._params.use_group_norm:
            # 32ch / 8 = 4ch/group
            self.decoder_bn1 = layers.GroupNormalization(groups=8)
        else:
            self.decoder_bn1 = layers.BatchNormalization()
        self.decoder_up2 = layers.UpSampling2D(size=2, interpolation="bilinear")
        self.decoder_conv2 = layers.Conv2D(16, 3, padding="same", activation="relu")
        if self._params.use_group_norm:
            # 16ch / 4 = 4ch/group
            self.decoder_bn2 = layers.GroupNormalization(groups=4)
        else:
            self.decoder_bn2 = layers.BatchNormalization()

        # === V3.4: Deep decoder — extra refinement convs per stage ===
        # Gives the model explicit capacity to reconstruct fine-grained depth
        # detail at full resolution. Only built when flag is on so V1/round-3
        # weights still load cleanly when the flag is off.
        if self._params.use_deep_decoder:
            self.decoder_conv1b = layers.Conv2D(
                32, 3, padding="same", activation="relu"
            )
            self.decoder_bn1b = layers.BatchNormalization()
            self.decoder_conv2b = layers.Conv2D(
                16, 3, padding="same", activation="relu"
            )
            self.decoder_bn2b = layers.BatchNormalization()
            # Extra final-resolution refinement before output
            self.decoder_refine = layers.Conv2D(
                16, 3, padding="same", activation="relu"
            )
            self.decoder_refine_bn = layers.BatchNormalization()
        else:
            self.decoder_conv1b = None
            self.decoder_bn1b = None
            self.decoder_conv2b = None
            self.decoder_bn2b = None
            self.decoder_refine = None
            self.decoder_refine_bn = None

        # Final 3×3 to single output channel. Linear activation lets
        # the loss gradient flow unconstrained; clip to ≥0 at inference.
        self.output_conv = layers.Conv2D(1, 3, padding="same", activation="linear")

        # Physics-based infiltration correction (no trainable weights)
        self.green_ampt_gate = GreenAmptGate(name="green_ampt_gate")

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

        N = self._params.n_flood_maps

        # === V3.1: Rain broadcast — tile rain_rate + rain_cum per-timestep
        # into spatiotemporal, so the spatial encoder sees rainfall context. ===
        if self._params.use_rain_broadcast:
            # temporal[:, :, 0] = rate (v2), temporal[:, :, 1] = cumul/tot (v2)
            H = self._spatial_height
            W = self._spatial_width
            rain_rate = temporal[:, :, tf.newaxis, tf.newaxis, 0:1]  # [B, N, 1, 1, 1]
            rain_cum = temporal[:, :, tf.newaxis, tf.newaxis, 1:2]
            rain_rate = tf.tile(rain_rate, [1, 1, H, W, 1])  # [B, N, H, W, 1]
            rain_cum = tf.tile(rain_cum, [1, 1, H, W, 1])
            spatiotemporal = tf.concat(
                [spatiotemporal, rain_rate, rain_cum], axis=-1  # [B, N, H, W, 3]
            )

        # Spatiotemporal CNN (two stages for skip connection)
        # Stage 1: [B, n, H, W, st_in] -> [B, n, H/2, W/2, 8]
        skip1 = self.st_cnn_stage1(spatiotemporal)
        # Stage 2: [B, n, H/2, W/2, 8] -> [B, n, H/4, W/4, 16]
        st_cnn_output = self.st_cnn_stage2(skip1)

        # === V3.2: Dilated geo refinement (residual) ===
        if self.geo_dilated is not None:
            geospatial = geospatial + self.geo_dilated(geospatial)

        # Geospatial CNN
        # [B, H, W, f ]-> [B, H', W', k2]
        # Add a new time axis and repeat n times -> [B, n, H', W', k2].
        geo_cnn_output = self.geo_cnn(geospatial)
        geo_cnn_output = geo_cnn_output[:, tf.newaxis, :, :, :]
        geo_cnn_output = tf.repeat(geo_cnn_output, N, axis=1)

        # Expand temporal inputs into maps
        # [B, n, m] -> [B, n, H', W', m]
        H_out = self._spatial_height // 4
        W_out = self._spatial_width // 4
        temp_input = temporal[:, :, tf.newaxis, tf.newaxis, :]
        temp_input = tf.tile(temp_input, [1, 1, H_out, W_out, 1])

        tensors_to_concat = [st_cnn_output, geo_cnn_output, temp_input]

        # === V3.3: Storm-intensity embedding broadcast ===
        if self.storm_embed is not None:
            # peak rain feature vector per sample across N timesteps [B, m]
            peak_temporal = tf.reduce_max(temporal, axis=1)  # [B, m]
            storm_vec = self.storm_embed(peak_temporal)  # [B, 32]
            # Broadcast to [B, N, H', W', 32]
            storm_map = storm_vec[:, tf.newaxis, tf.newaxis, tf.newaxis, :]
            storm_map = tf.tile(storm_map, [1, N, H_out, W_out, 1])
            tensors_to_concat.append(storm_map)

        lstm_input = tf.concat(tensors_to_concat, axis=-1)
        lstm_output = self.conv_lstm(lstm_input)
        lstm_output = self.attention(lstm_output)

        # Decoder with skip connection from encoder stage 1
        x = self.decoder_up1(lstm_output)  # [B, H/2, W/2, units]
        skip = skip1[:, -1]  # [B, H/2, W/2, 8]
        x = tf.concat([x, skip], axis=-1)  # [B, H/2, W/2, units+8]
        x = self.decoder_conv1(x)  # [B, H/2, W/2, 32]
        x = self.decoder_bn1(x)
        # V3.4: extra refinement at half resolution
        if self.decoder_conv1b is not None:
            x = self.decoder_conv1b(x)
            x = self.decoder_bn1b(x)
        x = self.decoder_up2(x)  # [B, H, W, 32]
        x = self.decoder_conv2(x)  # [B, H, W, 16]
        x = self.decoder_bn2(x)
        # V3.4: extra refinement at full resolution
        if self.decoder_conv2b is not None:
            x = self.decoder_conv2b(x)
            x = self.decoder_bn2b(x)
            x = self.decoder_refine(x)
            x = self.decoder_refine_bn(x)
        output = self.output_conv(x)  # [B, H, W, 1]

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
            prediction, cumul_F = self.green_ampt_gate(prediction, geospatial, cumul_F)

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

    # ------------------------------------------------------------------
    # PREVIOUS: Scheduled sampling train_step (commented for backtracking)
    # ------------------------------------------------------------------
    # def train_step_scheduled_sampling(self, data):
    #     """Custom train step with scheduled sampling.
    #
    #     Always runs two forward passes:
    #     - Pass 1: standard teacher-forced (GT spatiotemporal input)
    #     - Pass 2: spatiotemporal input where the most recent GT map is
    #       replaced with the model's own prediction (probability controlled
    #       by self._sampling_prob, ramped via ScheduledSamplingCallback).
    #
    #     When sampling_prob=0, pass 2 is identical to pass 1 (no-op).
    #     """
    #     x, y = data
    #     with tf.GradientTape() as tape:
    #         y_pred = self(x, training=True)
    #         loss_tf = self.compute_loss(y=y, y_pred=y_pred)
    #         st = x["spatiotemporal"]
    #         pred_detached = tf.stop_gradient(y_pred)
    #         corrupted_st = tf.concat(
    #             [st[:, 1:, :, :, :], pred_detached[:, tf.newaxis, :, :, :]],
    #             axis=1,
    #         )
    #         batch_size = tf.shape(st)[0]
    #         use_pred = tf.cast(
    #             tf.random.uniform([batch_size, 1, 1, 1, 1])
    #             < self._sampling_prob,
    #             tf.float32,
    #         )
    #         mixed_st = corrupted_st * use_pred + st * (1.0 - use_pred)
    #         x_ss = {
    #             "geospatial": x["geospatial"],
    #             "temporal": x["temporal"],
    #             "spatiotemporal": mixed_st,
    #         }
    #         y_pred_ss = self(x_ss, training=True)
    #         loss_ss = self.compute_loss(y=y, y_pred=y_pred_ss)
    #         loss = 0.5 * (loss_tf + loss_ss)
    #     gradients = tape.gradient(loss, self.trainable_variables)
    #     self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    #     for metric in self.metrics:
    #         if metric.name == "loss":
    #             metric.update_state(loss)
    #         else:
    #             metric.update_state(y, y_pred)
    #     return {m.name: m.result() for m in self.metrics}

    # ------------------------------------------------------------------
    # CURRENT: Autoregressive unrolling train_step (K-step)
    # ------------------------------------------------------------------
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

        # Scheduled sampling probability.
        # With temporal_feature_version=2, temporal channels already encode
        # storm progression (cumulative/time channels), so additional random
        # zero-context corruption is disabled by default.
        # Curriculum AR sampling: ramp up zero-context fraction over training.
        # Early epochs: learn flood physics from GT context.
        # Later epochs: force geospatial-only routing for cold-start.
        _ar_step = tf.cast(self.optimizer.iterations, tf.float32)
        _ar_prob = tf.minimum(0.20, 0.0 + _ar_step / 50000.0 * 0.20)
        spatiotemporal = tf.cond(
            tf.random.uniform(()) < _ar_prob,
            lambda: tf.zeros_like(spatiotemporal),
            lambda: spatiotemporal,
        )

        y_steps = self._normalize_labels(y)
        # Unstack along step axis → Python list with static length K.
        # Avoids tf.while_loop dynamic-shape XLA issues.
        y_step_list = tf.unstack(y_steps, axis=1)
        K = len(y_step_list)
        has_temporal_per_step = len(temporal.shape) == 4

        with tf.GradientTape() as tape:
            batch_size = tf.shape(spatiotemporal)[0]
            # Cumulative infiltration F starts at zero each simulation
            cumul_F = tf.zeros(
                (batch_size, self._spatial_height, self._spatial_width, 1),
                dtype=tf.float32,
            )

            total_loss = tf.constant(0.0)
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
                pred = tf.nn.relu(pred)

                # Green-Ampt infiltration correction
                pred, cumul_F = self.green_ampt_gate(pred, geospatial, cumul_F)

                step_loss = self.compute_loss(y=yk, y_pred=pred)

                # Mass preservation: penalise mismatch in mean depth
                pred_mass = tf.reduce_mean(pred, axis=[1, 2, 3])
                gt_mass = tf.reduce_mean(yk, axis=[1, 2, 3])
                mass_loss = tf.reduce_mean(tf.square(pred_mass - gt_mass))

                # Later steps matter more (error should not accumulate)
                time_weight = 1.0 + 0.3 * k

                total_loss += time_weight * (step_loss + 0.2 * mass_loss)

                # Feed corrected prediction back: clip at 4.0m to match GT depth cap
                # (was 2.5m which caused hard ceiling — pred never learned >2.5m)
                fb = tf.minimum(tf.stop_gradient(pred), 4.0)
                # Feedback noise: teaches robustness to imperfect context
                fb = fb + tf.random.normal(tf.shape(fb), stddev=0.02)
                fb = tf.nn.relu(fb)  # keep non-negative after noise
                st = tf.concat(
                    [st[:, 1:, :, :, :], fb[:, tf.newaxis, :, :, :]],
                    axis=1,
                )
                last_pred = pred

            total_loss = total_loss / K

            # arrival_time_loss disabled: log1p gradient explosion when depth > 0.5m
            # (sigmoid saturates -> grad = -1/1e-8 = -1e8 per step, NaN over K=7)
            # Fix: rework with hard soft-arrival using clip, not log-space survival

            # Add regularization losses once (not inside loop)
            if self.losses:
                total_loss += tf.add_n(self.losses)

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Metrics on last-step prediction
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(total_loss)
            else:
                metric.update_state(y_steps[:, -1], last_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """Autoregressive unrolling validation step (mirrors train_step)."""
        x, y = data

        geospatial = x["geospatial"]
        temporal = x["temporal"]
        spatiotemporal = x["spatiotemporal"]

        y_steps = self._normalize_labels(y)
        # Unstack along step axis → Python list with static length K.
        # Avoids tf.while_loop which causes XLA tracing errors with training=False.
        y_step_list = tf.unstack(y_steps, axis=1)
        K = len(y_step_list)
        has_temporal_per_step = len(temporal.shape) == 4

        batch_size = tf.shape(spatiotemporal)[0]
        cumul_F = tf.zeros(
            (batch_size, self._spatial_height, self._spatial_width, 1),
            dtype=tf.float32,
        )

        total_loss = tf.constant(0.0)
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
            pred = tf.nn.relu(pred)

            # Green-Ampt infiltration correction
            pred, cumul_F = self.green_ampt_gate(pred, geospatial, cumul_F)

            step_loss = self.compute_loss(y=yk, y_pred=pred)

            pred_mass = tf.reduce_mean(pred, axis=[1, 2, 3])
            gt_mass = tf.reduce_mean(yk, axis=[1, 2, 3])
            mass_loss = tf.reduce_mean(tf.square(pred_mass - gt_mass))

            time_weight = 1.0 + 0.3 * k
            total_loss += time_weight * (step_loss + 0.2 * mass_loss)

            # Feed corrected prediction back: clip at 4.0m (matches train_step)
            pred_fb = tf.minimum(pred, 4.0)
            st = tf.concat(
                [st[:, 1:, :, :, :], pred_fb[:, tf.newaxis, :, :, :]],
                axis=1,
            )
            last_pred = pred

        total_loss = total_loss / K

        if self.losses:
            total_loss += tf.add_n(self.losses)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(total_loss)
            else:
                metric.update_state(y_steps[:, -1], last_pred)

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


class ScheduledSamplingCallback(keras.callbacks.Callback):
    """Linearly ramps scheduled sampling probability over training.

    During training, FloodConvLSTM.train_step uses self._sampling_prob
    to decide whether to corrupt the spatiotemporal input with the model's
    own prediction. This callback ramps that probability from 0 to max_prob
    over warmup_epochs, then holds it constant.
    """

    def __init__(self, max_prob=0.5, warmup_epochs=15):
        """Configure peak sampling probability and the warmup ramp length."""
        super().__init__()
        self.max_prob = max_prob
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        """Update the model's scheduled-sampling probability at each epoch."""
        prob = min(epoch / max(self.warmup_epochs, 1), 1.0) * self.max_prob
        self.model._sampling_prob.assign(prob)
        print(f"  Scheduled sampling prob: {prob:.3f}")

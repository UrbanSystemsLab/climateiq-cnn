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
        # Full-resolution residual head (2026-10-12 architecture audit):
        # the decoder reconstructs output from H/4 with only an 8-channel
        # skip (depth/rain stream only, no full-res geospatial or previous-
        # depth input). A persistence-baseline test confirmed this directly:
        # given REAL ground-truth context (no rollout error), the model
        # still scored far below a trivial "copy the last known frame"
        # baseline on CSI and MAE in every city tested -- the decoder is
        # discarding information it already has, not just struggling with
        # uncertainty. This flag adds a small full-resolution refinement
        # head -- out = output_conv(x) + head(concat[x, geospatial,
        # prev_depth]) -- with the head's final layer zero-initialised, so
        # with the flag OFF (default) or immediately after turning it on but
        # before training, behaviour is byte-identical to today's model.
        use_full_res_residual_head: bool = False
        # v6: replace decoder BN with GroupNorm (no train/eval gap)
        use_group_norm: bool = False
        # Explicit steepest-descent flow-routing correction after Green-Ampt,
        # each autoregressive step (see FlowRoutingGate) -- gives the model a
        # physical prior for WHERE water moves instead of leaving spatial
        # placement (what CSI measures) purely to learned convolutions.
        use_flow_routing: bool = False
        # CFL-like cap on FlowRoutingGate -- max fraction of a cell's depth
        # moved to one neighbour per timestep. First smoke test (0.5)
        # improved flood CSI 20-40% relative but regressed dangerous-depth
        # CSI and peak correlation on Manhattan -- plausibly over-smoothing
        # legitimate deep-water peaks. Only used when use_flow_routing=True.
        flow_routing_max_fraction: float = 0.5
        # Weight on gravity_consistency_loss (customloss.py) -- penalises the
        # network's OWN prediction for isolated water-surface-elevation peaks
        # unsupported by terrain (water "appearing from nowhere"), a training
        # signal complementary to FlowRoutingGate's post-hoc correction. 0.0 =
        # off (default, unchanged behaviour).
        gravity_loss_weight: float = 0.0
        # Weight on infiltration_consistency_loss (customloss.py) -- penalises
        # FALSE-POSITIVE flood predictions (predicted wet, actually dry) in
        # proportion to the terrain's soil infiltration capacity (hydraulic
        # conductivity + porosity). A false-positive-channel analysis found
        # the model's false positives sit on measurably higher-infiltration
        # terrain than its true positives, despite that data already being
        # available in the geospatial input -- nothing previously penalised
        # ignoring it. 0.0 = off (default, unchanged behaviour).
        infiltration_loss_weight: float = 0.0
        # LearnedFlowRoutingGate: replaces FlowRoutingGate's fixed heuristic
        # (single steepest-descent neighbour, hand-picked Manning's-inspired
        # speed) with a small TRAINABLE routing head -- softmax over 9
        # outcomes (retain + 8 neighbour directions) per pixel, so mass
        # conservation is structural (softmax sums to 1, nothing created or
        # destroyed) rather than a soft loss penalty, while the actual
        # fractions are learned from context instead of fixed by hand. Can
        # split flow across multiple downhill neighbours simultaneously,
        # which the fixed single-neighbour heuristic architecturally cannot.
        # Mutually exclusive with use_flow_routing in practice (both applying
        # would double-route); kept as a separate flag for a clean A/B against
        # the fixed heuristic on the same checkpoint/eval.
        use_learned_routing: bool = False
        # "v1" = log-depth hybrid (default) | "v3" = depth-weighted linear MSE
        # | "v4" = v3 + smooth depth-weight ramp + down-weighted peak penalty
        # | "v5" = v4 + depth-weight fixed to also weight shallow/moderate
        #   floods (0.05-0.4m), not just deep water
        # | "v6" = v5 rebalanced -- restores stronger relative emphasis on
        #   deep water (1-4m) that v5 traded away -- see customloss.py
        # | "v7" = v6 with boundary-aware Dice (ABeDice) restored in place of
        #   plain focal -- directly targets flood-extent/CSI, which focal
        #   alone doesn't emphasize -- see customloss.py
        # | "v8" = v6 with the focal term's false-negative weight rebalanced
        #   10x -> 3x -- targets over-prediction of flood EXTENT found via
        #   true-AR visual inspection (predicted flooding spreads across the
        #   whole street network, not just where it actually floods)
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
                "use_full_res_residual_head": self.use_full_res_residual_head,
                "use_group_norm": self.use_group_norm,
                "use_flow_routing": self.use_flow_routing,
                "flow_routing_max_fraction": self.flow_routing_max_fraction,
                "gravity_loss_weight": self.gravity_loss_weight,
                "infiltration_loss_weight": self.infiltration_loss_weight,
                "use_learned_routing": self.use_learned_routing,
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
        _loss_version = getattr(self._params, "loss_version", "v1")
        _loss_fns = {
            # v3-v5, v7-v11, v11_fa3/fa4 removed: a sequential sweep where
            # each was superseded by the next, with zero remaining callers.
            # v6 and v1 stay because sibling scripts still reference them by
            # name; v11_fa2 is the actual best (confirmed via the best
            # checkpoint's own saved config.json).
            "v1": customloss.make_hybrid_loss,
            "v6": customloss.make_hybrid_loss_v6,
            "v11_fa2": customloss.make_hybrid_loss_v11_fa2,
            # v12_bandcsi: v11_fa2 + a soft-CSI surrogate for the low/med
            # depth bands -- see customloss.py docstring. Untested as of
            # 2026-10-21; smoke-test before trusting it.
            "v12_bandcsi": customloss.make_hybrid_loss_v12_bandcsi,
        }
        if _loss_version not in _loss_fns:
            # Was a silent fallback to v1 (the already-known-bad log-depth
            # loss) on ANY typo/unset value -- a latent footgun given several
            # sibling scripts still hard-code loss_version="v1" for baseline
            # comparisons. Fail loudly instead.
            raise ValueError(
                f"Unrecognized loss_version={_loss_version!r}; must be one of "
                f"{sorted(_loss_fns)}"
            )
        loss_fn = _loss_fns[_loss_version]
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
        #
        # BUG (found in launch-readiness audit): at cumul_F=0 -- true at the
        # start of EVERY training batch, EVERY call_n inference run, and (via
        # the dataset.py last_t off-by-one, separately fixed) EVERY training
        # window before that fix -- the old epsilon (1e-9 m, ~1 nanometer) let
        # this ratio reach ~1e8, so fc*DT_S could reach ~1e6 m: astronomically
        # above any real predicted depth. `tf.minimum(fc*DT_S, pred)` then
        # always equalled `pred` exactly, so `corrected = relu(pred-pred) = 0`
        # for every pervious pixel, with ZERO gradient through this path, on
        # literally the first step of every rollout. Fix: a physically
        # meaningful epsilon floor (1mm, like a minimal initial-abstraction
        # depth, not a numerical-stability nanometer) plus an explicit cap on
        # the correction factor so fc can't blow up regardless of the
        # specific psi_f/delta_theta/K_s combination.
        F_EPS = 1e-3  # 1 mm -- physically meaningful floor, not division-by-zero guard
        correction_factor = tf.minimum((psi_f * delta_theta) / (cumul_F + F_EPS), 10.0)
        fc = K_s * (1.0 + correction_factor)

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
class FlowRoutingGate(keras.layers.Layer):
    """Physics-inspired shallow-flow routing correction.

    Why this exists: the network currently has no explicit mechanism for
    WHERE water moves between neighbouring pixels -- that has to be learned
    entirely implicitly, through convolutional pattern-matching over
    rainfall/terrain/history. That's exactly the failure mode CSI measures
    (correctly placing the flood extent, not just getting aggregate depth
    magnitude right), and it's the one architectural gap identified this
    session that no loss/curriculum tweak can fix, because none of them give
    the model an explicit notion of "downhill" at all.

    This gate approximates one kinematic-wave overland-flow step: each pixel
    moves a bounded fraction of its depth toward its steepest-descent
    neighbour (computed from the elevation channel, ch0) at a depth- and
    surface-dependent speed, every autoregressive timestep. No trainable
    parameters -- a fixed physical calculation layered after GreenAmptGate,
    same pattern.

    KNOWN LIMITATION (documented here, not hidden): ch0 is elevation
    normalised per STUDY AREA (not per chunk -- confirmed via
    usl_pipeline/usl_lib/storage/metastore.py's elevation_min/max), so
    steepest-DESCENT DIRECTION is physically valid within one city, but the
    true slope MAGNITUDE in metres/metre can't be recovered here because the
    real elevation range isn't passed into the model. The speed term below is
    therefore deliberately NOT full Manning's-equation slope^0.5 -- it uses
    only depth and surface type (K_s channel, already used by GreenAmptGate
    to distinguish pervious/impervious). Passing elevation_min/max as an
    additional model input would let a future version use true slope and
    make this fully physically rigorous; flagged as the natural next step,
    not silently approximated as if it were already exact.
    """

    DT_S = 300.0  # 5-minute timestep, matches GreenAmptGate
    CELL_SIZE_M = 2.0  # matches _compute_all_geo_features default
    MANNING_N_IMPERVIOUS = 0.02
    MANNING_N_PERVIOUS = 0.06

    # 8-neighbour offsets (dy, dx): N, S, E, W, NE, NW, SE, SW
    _OFFSETS = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (1, 1), (1, -1)]

    def __init__(self, max_fraction: float = 0.5, **kwargs):
        """Builds the gate.

        Args:
            max_fraction: CFL-like cap -- never move more than this fraction
                of a cell's depth to one neighbour in a single timestep.
                First smoke test (max_fraction=0.5) improved flood CSI
                20-40% relative on both test cities but regressed
                dangerous-depth CSI and Manhattan peak correlation --
                plausibly over-redistributing depth away from legitimate
                deep-water peaks. A smaller cap trades some of the CSI gain
                for less peak smoothing; tune based on which trade-off the
                deployment actually needs.
        """
        super().__init__(**kwargs)
        self.max_fraction = max_fraction

    def get_config(self):
        """Returns this layer's constructor kwargs for serialization."""
        config = super().get_config()
        config.update({"max_fraction": self.max_fraction})
        return config

    @staticmethod
    def _shift(t: tf.Tensor, dy: int, dx: int, mode: str = "SYMMETRIC") -> tf.Tensor:
        """Read `t` as if shifted by (dy, dx).

        mode="SYMMETRIC" (edge-replicated) is for reading NEIGHBOUR VALUES
        (elevation/validity) -- safe at boundaries because it makes an
        off-grid neighbour look identical to the cell itself, giving drop=0,
        which correctly disqualifies routing off the domain edge.

        mode="CONSTANT" (zero-padded) MUST be used for routing mass (the
        inflow gather) -- SYMMETRIC padding duplicates the edge row/column,
        which silently double-counts a boundary cell's outflow as inflow to
        itself (confirmed via a synthetic mass-conservation test: using
        SYMMETRIC for both purposes inflated total depth by 2.5% on a simple
        tilted-plane case with zero real gain/loss expected). CONSTANT
        padding makes mass that would leave the domain simply vanish, which
        is physically correct for this per-chunk (not cross-chunk) routing.
        """
        padded = tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]], mode=mode)
        h = tf.shape(t)[1]
        w = tf.shape(t)[2]
        return padded[:, 1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w, :]

    def call(self, pred: tf.Tensor, geospatial: tf.Tensor) -> tf.Tensor:
        """Apply one step of steepest-descent flow routing.

        Args:
            pred:       (B, H, W, 1) predicted flood depth (m), post-infiltration.
            geospatial: (B, H, W, 12) static features.

        Returns:
            (B, H, W, 1) routed flood depth.
        """
        elev = geospatial[:, :, :, 0:1]  # normalised [0,1] per study area
        valid = geospatial[:, :, :, 1:2]
        k_s = geospatial[:, :, :, 4:5]  # 0 = impervious

        pervious = tf.cast(k_s > 0, tf.float32)
        manning_n = (
            self.MANNING_N_IMPERVIOUS
            + (self.MANNING_N_PERVIOUS - self.MANNING_N_IMPERVIOUS) * pervious
        )

        depth = tf.nn.relu(pred)

        # Steepest-descent direction: among the 8 neighbours that are both
        # VALID and LOWER than this pixel, pick the one with the largest
        # (elev - neighbour_elev) drop. Distance-normalise so diagonals
        # aren't spuriously favoured over orthogonal neighbours.
        drops = []
        for dy, dx in self._OFFSETS:
            n_elev = self._shift(elev, dy, dx)
            n_valid = self._shift(valid, dy, dx)
            dist = self.CELL_SIZE_M * (1.41421356 if dy != 0 and dx != 0 else 1.0)
            drop = (elev - n_elev) / dist
            # Disqualify neighbours outside the valid domain or uphill.
            drop = tf.where(n_valid > 0.5, drop, -tf.ones_like(drop))
            drops.append(drop)
        drops_stack = tf.concat(drops, axis=-1)  # (B, H, W, 8)
        max_drop = tf.reduce_max(drops_stack, axis=-1, keepdims=True)
        has_downhill = tf.cast(max_drop > 0.0, tf.float32)

        # Speed: depth-dependent (deeper water moves faster), surface-
        # dependent (impervious sheds faster than pervious) -- NOT slope-
        # dependent, see class docstring for why.
        speed = (1.0 / manning_n) * tf.pow(depth + 1e-6, 2.0 / 3.0)
        frac = tf.clip_by_value(
            speed * self.DT_S / self.CELL_SIZE_M, 0.0, self.max_fraction
        )
        frac = frac * has_downhill * valid

        outflow = depth * frac

        # Route each cell's outflow to its single steepest-descent neighbour:
        # for offset (dy, dx), a cell sends outflow to the cell AT (dy, dx)
        # relative to itself -- equivalently, that target cell RECEIVES it
        # via the inverse shift (-dy, -dx).
        argmax_idx = tf.argmax(drops_stack, axis=-1)  # (B, H, W)
        one_hot = tf.one_hot(argmax_idx, depth=8, axis=-1)  # (B, H, W, 8)
        inflow = tf.zeros_like(depth)
        for i, (dy, dx) in enumerate(self._OFFSETS):
            contrib = outflow * one_hot[:, :, :, i : i + 1]
            inflow += self._shift(contrib, -dy, -dx, mode="CONSTANT")

        routed = depth - outflow + inflow
        # Outside the valid domain, pass the prediction through unchanged --
        # routing only applies within the real study area.
        routed = tf.nn.relu(routed) * valid + depth * (1.0 - valid)
        return routed


@register_keras_serializable()
class LearnedFlowRoutingGate(keras.layers.Layer):
    """Trainable alternative to FlowRoutingGate's fixed steepest-descent rule.

    FlowRoutingGate always sends a hand-computed fraction of a cell's depth
    to its single steepest-descent neighbour, using a fixed Manning's-
    inspired speed formula -- correct in kind (gives the model an explicit
    notion of "downhill"), but every quantity in it (which neighbour, how
    much, how fast) is hand-picked rather than learned, and it can only ever
    split flow toward exactly one neighbour.

    This layer keeps the same hard-conservation property (nothing is created
    or destroyed -- a softmax over "retain + 8 directions" always sums to 1,
    so whatever leaves a cell is fully accounted for, exactly like
    FlowRoutingGate's outflow/inflow bookkeeping) but makes the split
    TRAINABLE: a small 1x1 conv over [elevation drop toward each of the 8
    neighbours, current depth, pervious/impervious] predicts logits for how
    much of a cell's water stays vs. moves each direction, and can distribute
    across MULTIPLE downhill neighbours at once with a per-context-dependent
    (not fixed-formula) amount.

    Initialisation is deliberately conservative: the kernel starts as small
    random noise and the bias strongly favours "retain" (softmax(0,...,0,5)
    puts ~99.3% of mass on retain) so that, warm-started onto an already-good
    checkpoint, this layer starts close to a no-op and has to EARN any
    routing behaviour through gradient descent -- rather than starting from
    an architecturally-plausible-looking but factually random redistribution
    of real predicted depth.
    """

    CELL_SIZE_M = 2.0  # matches FlowRoutingGate / _compute_all_geo_features default
    _OFFSETS = FlowRoutingGate._OFFSETS  # same 8-neighbour order: N,S,E,W,NE,NW,SE,SW
    _shift = staticmethod(FlowRoutingGate._shift)

    def __init__(self, retain_bias_init: float = 5.0, **kwargs):
        """Builds the gate.

        Args:
            retain_bias_init: Initial bias on the "retain" logit -- higher
                values start the layer closer to a no-op (see class
                docstring).
            **kwargs: Forwarded to keras.layers.Layer.
        """
        super().__init__(**kwargs)
        self.retain_bias_init = retain_bias_init

    def build(self, input_shape):
        """Creates the 1x1 conv that predicts per-direction routing logits."""

        def _bias_init(shape, dtype=None):
            # 8 direction logits at 0, retain logit at retain_bias_init.
            return tf.concat(
                [
                    tf.zeros((8,), dtype=dtype),
                    tf.constant([self.retain_bias_init], dtype=dtype),
                ],
                axis=0,
            )

        self.route_conv = keras.layers.Conv2D(
            filters=9,
            kernel_size=1,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer=_bias_init,
            name="learned_route_logits",
        )
        super().build(input_shape)

    def get_config(self):
        """Returns this layer's constructor kwargs for serialization."""
        config = super().get_config()
        config.update({"retain_bias_init": self.retain_bias_init})
        return config

    def call(
        self,
        pred: tf.Tensor,
        geospatial: tf.Tensor,
        elevation_range_m: tf.Tensor = None,
    ) -> tf.Tensor:
        """Apply one learned, mass-conservative routing step.

        Args:
            pred:       (B, H, W, 1) predicted flood depth (m), post-infiltration.
            geospatial: (B, H, W, 12) static features.
            elevation_range_m: Optional (B,) or (B, 1) tensor -- the real
                per-study-area (elevation_max - elevation_min) in metres,
                looked up from Firestore study_area metadata (see
                train_data/study_area_elevation_ranges.json). ch0's
                elevation is normalised PER STUDY AREA to [0,1], so a raw
                normalised drop of e.g. 0.01 means a real ~1.7m slope in
                flat Lafayette (range 28.6m) but a real ~11m slope in
                mountainous Cape Town (range 1086m) -- the same routing
                decision is physically wrong for one of those two cities
                unless the network is told which city it's looking at.
                When provided, drops are rescaled into true metres before
                the routing head sees them, so the same input value means
                the same physical slope everywhere. Defaults to None
                (unscaled, normalised drops -- the original behaviour,
                fully backward compatible with any checkpoint/caller that
                doesn't know about this).
        """
        elev = geospatial[:, :, :, 0:1]
        valid = geospatial[:, :, :, 1:2]
        k_s = geospatial[:, :, :, 4:5]
        pervious = tf.cast(k_s > 0, tf.float32)
        depth = tf.nn.relu(pred)

        drops = []
        for dy, dx in self._OFFSETS:
            n_elev = self._shift(elev, dy, dx)
            n_valid = self._shift(valid, dy, dx)
            dist = self.CELL_SIZE_M * (1.41421356 if dy != 0 and dx != 0 else 1.0)
            drop = (elev - n_elev) / dist
            drop = tf.where(n_valid > 0.5, drop, tf.zeros_like(drop))
            drops.append(drop)
        drops_stack = tf.concat(drops, axis=-1)  # (B, H, W, 8)
        if elevation_range_m is not None:
            scale = tf.reshape(elevation_range_m, [-1, 1, 1, 1])
            drops_stack = drops_stack * scale

        features = tf.concat([drops_stack, depth, pervious], axis=-1)  # (B, H, W, 10)
        logits = self.route_conv(features)  # (B, H, W, 9)
        probs = tf.nn.softmax(logits, axis=-1)
        frac_dirs = probs[:, :, :, :8]  # (B, H, W, 8)
        frac_retain = probs[:, :, :, 8:9]  # (B, H, W, 1)

        outflow_per_dir = depth * frac_dirs
        retained = depth * frac_retain

        inflow = tf.zeros_like(depth)
        for i, (dy, dx) in enumerate(self._OFFSETS):
            inflow += self._shift(
                outflow_per_dir[:, :, :, i : i + 1], -dy, -dx, mode="CONSTANT"
            )

        routed = retained + inflow
        routed = tf.nn.relu(routed) * valid + depth * (1.0 - valid)
        return routed


@register_keras_serializable()
class FloodRoutingCNN(keras.Model):
    """A structurally different model: mass conservation IS the recurrence.

    Not a correction applied to it. FloodConvLSTM predicts depth directly
    via an opaque ConvLSTM hidden
    state, then patches the result with GreenAmptGate/FlowRoutingGate/
    LearnedFlowRoutingGate afterward. Those patches are real and help, but
    the network's PRIMARY mechanism for "where does water go" is still
    whatever an RNN hidden state implicitly learned -- physics is bolted on,
    not architectural.

    Here, the ONLY thing carried between timesteps is the actual physical
    depth field (H, W, 1) plus cumulative infiltration state -- no ConvLSTM
    cell state at all. Every single step MUST go through the same hard,
    exactly-conservative routing update verified for LearnedFlowRoutingGate
    (softmax over retain + 8 directions, reusing FlowRoutingGate._shift's
    validated CONSTANT-padding mass bookkeeping). A deep dilated-CNN encoder
    (dilation 1/2/4/8/16, full spatial resolution, no 4x downsampling the
    way FloodConvLSTM's geo_cnn uses) produces the per-pixel routing
    decision and a learned rainfall-coupling gain from [geospatial, current
    depth, rain rate, rain cumulative] -- so the receptive field per step is
    much larger than FloodConvLSTM's, and it operates at full resolution
    throughout rather than only after a decoder upsamples back.

    This directly targets the two structural gaps flow-routing/gravity-loss
    couldn't: (1) mass conservation as the core update rule, not a patch
    after a free-form prediction, and (2) a bigger per-step receptive field
    without needing multi-timestep ConvLSTM memory to compensate for a
    small one.

    Trained from scratch (a genuinely different architecture -- no warm
    start possible from FloodConvLSTM checkpoints) via a custom script that
    unrolls call_n's own recurrence under a GradientTape and applies the
    existing per-step loss functions from customloss.py -- see
    train_scratch_floodrouting.py.
    """

    _OFFSETS = FlowRoutingGate._OFFSETS
    _shift = staticmethod(FlowRoutingGate._shift)

    def __init__(
        self,
        params: FloodModel.Params,
        spatial_dims: tuple[int, int] = (constants.MAP_HEIGHT, constants.MAP_WIDTH),
    ):
        """Builds the encoder and routing/source heads.

        Args:
            params: Shared FloodModel.Params (loss_version, optimizer, etc.).
            spatial_dims: (height, width) of the patches this model runs on.
        """
        super().__init__()
        self._params = params
        self._spatial_height, self._spatial_width = spatial_dims

        in_ch = (
            constants.GEO_FEATURES + 1 + 2 + 8
        )  # geospatial + depth + rain_rate + rain_cum + 8 true-slope drops
        # 8 extra channels: true-metre elevation drop toward each of the 8
        # neighbours (raw ch0 * elevation_range_m), not just the normalised-
        # per-study-area value -- same fix already validated for
        # LearnedFlowRoutingGate (Manhattan train_smoke_trueslope.py result:
        # CSI 0.172->0.181, peak_corr 0.769->0.772, false-alarm 0.723->0.656).
        # Given here as an ENCODER INPUT (not just the final routing decision)
        # so the whole network reasons with real slope throughout, not only
        # at the last layer.
        # GroupNorm (not BatchNorm) after every conv -- same lesson already
        # learned for FloodConvLSTM's decoder (use_group_norm comment above):
        # BatchNorm's running-stats introduce a train/eval gap. Added after
        # the from-scratch training attempt collapsed to predicting near-
        # zero depth everywhere (flat 0.20-0.34 loss for all 2000 steps,
        # never converging) -- five stacked dilated convs with NO
        # normalisation at increasing dilation is exactly the kind of stack
        # where activation scale can drift step to step, making gradients
        # too noisy for an undertrained network to escape the trivial
        # "predict nothing" minimum in a short dose.
        self.encoder = keras.Sequential(
            [
                layers.InputLayer((self._spatial_height, self._spatial_width, in_ch)),
                layers.Conv2D(
                    32, 3, padding="same", dilation_rate=1, activation="relu"
                ),
                layers.GroupNormalization(groups=8),
                layers.Conv2D(
                    32, 3, padding="same", dilation_rate=2, activation="relu"
                ),
                layers.GroupNormalization(groups=8),
                layers.Conv2D(
                    32, 3, padding="same", dilation_rate=4, activation="relu"
                ),
                layers.GroupNormalization(groups=8),
                layers.Conv2D(
                    32, 3, padding="same", dilation_rate=8, activation="relu"
                ),
                layers.GroupNormalization(groups=8),
                layers.Conv2D(
                    32, 3, padding="same", dilation_rate=16, activation="relu"
                ),
                layers.GroupNormalization(groups=8),
            ],
            name="routing_encoder",
        )

        def _retain_bias_init(shape, dtype=None):
            return tf.concat(
                [tf.zeros((8,), dtype=dtype), tf.constant([5.0], dtype=dtype)], axis=0
            )

        self.route_head = layers.Conv2D(
            9,
            1,
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer=_retain_bias_init,
            name="route_head",
        )
        self.source_head = layers.Conv2D(
            1,
            1,
            activation="sigmoid",
            kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
            name="source_gain",
        )
        # BUG FOUND after 3 from-scratch training attempts all collapsed to
        # predicting near-zero depth everywhere (CSI=0.000 on both cities,
        # every time, regardless of normalisation or flood-content-filtered
        # sampling): the raw rain_rate temporal feature is scaled to ~1e-5
        # (confirmed empirically: max 4.86e-5 across Atlanta's rainfall
        # data), and source_gain is sigmoid-bounded to [0,1] -- so the
        # maximum possible depth added per step via source_gain*rain_map
        # was ~0.00005m, FIVE ORDERS OF MAGNITUDE below the 0.05m flood
        # threshold. This was architecturally impossible to learn past, not
        # a training or data problem -- no amount of gradient descent could
        # ever produce meaningful flooding through a channel capped at
        # 5e-5m/step. Fixed with a learnable, unbounded-positive rescale
        # applied to rain_map before combining with source_gain, so the
        # network can find the real conversion factor via training instead
        # of being capped by construction.
        self.rain_scale = tf.Variable(1000.0, trainable=True, name="rain_scale")
        self.green_ampt_gate = GreenAmptGate(name="green_ampt_gate")

    def get_config(self):
        """Returns this model's constructor kwargs for serialization."""
        return {
            "params": self._params.to_dict(),
            "spatial_dims": (self._spatial_height, self._spatial_width),
        }

    def _route_step(
        self, depth, geospatial, rain_rate, rain_cum, cumul_F, elevation_range_m=None
    ):
        """One physically-updated timestep. depth/cumul_F are (B,H,W,1).

        elevation_range_m: optional (B,) or (B,1) tensor, real per-study-
            area (elevation_max - elevation_min) in metres. When given, the
            8 directional elevation drops fed to the encoder are rescaled
            from ch0's per-study-area-normalised units into true metres, so
            the same input value means the same physical slope in every
            city. Defaults to None (raw normalised drops -- backward
            compatible with Stage 1/2/3 checkpoints that trained without it).
        """
        H, W = self._spatial_height, self._spatial_width
        rain_map = tf.tile(tf.reshape(rain_rate, [-1, 1, 1, 1]), [1, H, W, 1])
        cum_map = tf.tile(tf.reshape(rain_cum, [-1, 1, 1, 1]), [1, H, W, 1])

        elev = geospatial[:, :, :, 0:1]
        valid = geospatial[:, :, :, 1:2]
        drops = []
        for dy, dx in self._OFFSETS:
            n_elev = self._shift(elev, dy, dx)
            n_valid = self._shift(valid, dy, dx)
            dist = 2.0 * (1.41421356 if dy != 0 and dx != 0 else 1.0)  # 2.0m cell size
            drop = (elev - n_elev) / dist
            drop = tf.where(n_valid > 0.5, drop, tf.zeros_like(drop))
            drops.append(drop)
        drops_stack = tf.concat(drops, axis=-1)  # (B, H, W, 8)
        if elevation_range_m is not None:
            drops_stack = drops_stack * tf.reshape(elevation_range_m, [-1, 1, 1, 1])

        x = tf.concat([geospatial, depth, rain_map, cum_map, drops_stack], axis=-1)
        feat = self.encoder(x)
        logits = self.route_head(feat)

        # Soft physical prior: bias the 8 directional logits toward true
        # downhill flow, proportional to the real (metre-scale) elevation
        # drop. Trained WITH this term present so gradients rebalance the
        # whole network around it. This is the mechanism the `downhill`
        # production checkpoint was actually trained under -- a stricter
        # HARD mask variant was also tried (uphill directions forced to
        # ~0 probability regardless of learned logits) and, on real
        # evaluation, underperformed this soft version on every metric
        # (lower flood detection, lower dangerous-depth detection, worse
        # false-alarm rate) -- rejected, not adopted. Restored here so a
        # warm-start from `downhill` continues under the SAME mechanism
        # its weights were actually trained with.
        DOWNHILL_BIAS = 20.0
        dir_logits = logits[:, :, :, :8] + DOWNHILL_BIAS * drops_stack
        retain_logit = logits[:, :, :, 8:9]
        logits = tf.concat([dir_logits, retain_logit], axis=-1)

        probs = tf.nn.softmax(logits, axis=-1)
        frac_dirs = probs[:, :, :, :8]
        frac_retain = probs[:, :, :, 8:9]

        depth_relu = tf.nn.relu(depth)
        depth_after_infil, cumul_F_new = self.green_ampt_gate(
            depth_relu, geospatial, cumul_F
        )

        outflow_per_dir = depth_after_infil * frac_dirs
        retained = depth_after_infil * frac_retain
        inflow = tf.zeros_like(depth_after_infil)
        for i, (dy, dx) in enumerate(self._OFFSETS):
            inflow += self._shift(
                outflow_per_dir[:, :, :, i : i + 1], -dy, -dx, mode="CONSTANT"
            )
        routed = retained + inflow

        valid = geospatial[:, :, :, 1:2]
        routed = tf.nn.relu(routed) * valid + depth_after_infil * (1.0 - valid)

        source_gain = self.source_head(feat)
        rain_map_scaled = rain_map * tf.nn.softplus(self.rain_scale)
        new_depth = routed + source_gain * rain_map_scaled * valid
        return new_depth, cumul_F_new

    def call_n(self, full_input: FloodModel.Input, n: int = 1) -> tf.Tensor:
        """Same contract as FloodConvLSTM.call_n: returns (B, n, H, W).

        Always starts from zero depth (matches real deployment exactly --
        no separate history-window input needed, since the only state this
        model carries IS the physical depth field, which genuinely starts
        at zero for a fresh storm).
        """
        geospatial = full_input["geospatial"]
        temporal = full_input["temporal"]
        elevation_range_m = full_input.get("elevation_range_m")
        B = tf.shape(geospatial)[0]
        H, W = self._spatial_height, self._spatial_width

        depth = tf.zeros((B, H, W, 1), dtype=tf.float32)
        cumul_F = tf.zeros((B, H, W, 1), dtype=tf.float32)
        predictions = tf.TensorArray(tf.float32, size=n)
        for t in range(n):
            rain_rate = temporal[:, t, 0]
            rain_cum = temporal[:, t, 1]
            depth, cumul_F = self._route_step(
                depth, geospatial, rain_rate, rain_cum, cumul_F, elevation_range_m
            )
            predictions = predictions.write(t, depth)
        predictions = tf.stack(tf.unstack(predictions.stack()), axis=1)
        return tf.squeeze(predictions, axis=-1)


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

        # Full-resolution residual head (see use_full_res_residual_head
        # above). Zero-initialised final layer: with zero weights AND zero
        # bias, this head outputs exactly 0 everywhere regardless of its
        # input, so `output_conv(x) + head(...)` equals `output_conv(x)`
        # alone until the head is actually trained -- a true no-op, not an
        # approximate one.
        if self._params.use_full_res_residual_head:
            self.full_res_head_conv1 = layers.Conv2D(
                16, 3, padding="same", activation="relu"
            )
            self.full_res_head_conv2 = layers.Conv2D(
                1,
                3,
                padding="same",
                activation="linear",
                kernel_initializer="zeros",
                bias_initializer="zeros",
            )
        else:
            self.full_res_head_conv1 = None
            self.full_res_head_conv2 = None

        # Physics-based infiltration correction (no trainable weights)
        self.green_ampt_gate = GreenAmptGate(name="green_ampt_gate")
        self.flow_routing_gate = (
            FlowRoutingGate(
                max_fraction=self._params.flow_routing_max_fraction,
                name="flow_routing_gate",
            )
            if self._params.use_flow_routing
            else None
        )
        self.learned_routing_gate = (
            LearnedFlowRoutingGate(name="learned_routing_gate")
            if self._params.use_learned_routing
            else None
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

        # Captured before the rain-broadcast concat below reassigns
        # spatiotemporal -- the full-resolution residual head needs the
        # most recent depth frame on its own, not mixed with rain channels.
        prev_depth = spatiotemporal[:, -1]  # [B, H, W, 1]

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

        if self.full_res_head_conv1 is not None:
            # Full-resolution residual correction -- gives the output a
            # direct path to full-res geospatial (buildings, DEM_sink,
            # flow_accumulation) and the previous depth frame, neither of
            # which otherwise reach the decoder past the H/2 skip. Additive,
            # not a replacement: output = output_conv(x) + head(...).
            residual_input = tf.concat([x, geospatial, prev_depth], axis=-1)
            residual = self.full_res_head_conv1(residual_input)
            residual = self.full_res_head_conv2(residual)
            output = output + residual

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
        elevation_range_m = full_input.get("elevation_range_m")

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
            if self.flow_routing_gate is not None:
                prediction = self.flow_routing_gate(prediction, geospatial)
            if self.learned_routing_gate is not None:
                prediction = self.learned_routing_gate(
                    prediction, geospatial, elevation_range_m
                )

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
        elevation_range_m = x.get("elevation_range_m")

        # Scheduled sampling probability.
        # With temporal_feature_version=2, temporal channels already encode
        # storm progression (cumulative/time channels), so additional random
        # zero-context corruption is disabled by default.
        # Curriculum AR sampling: ramp up zero-context fraction over training.
        # Early epochs: learn flood physics from GT context.
        # Later epochs: force geospatial-only routing for cold-start.
        # Raised 0.20 -> 0.35 cap: today's true-AR investigation found the
        # model over-predicts flood extent from the very first non-zero
        # timestep (Atlanta t=1: predicted max 0.97m vs GT max 0.09m). More
        # cold-start training exposure is a direct, complementary lever to
        # the AR-rollout curriculum (which targets rollout LENGTH, not how
        # often training sees a true zero-context start at all).
        _ar_step = tf.cast(self.optimizer.iterations, tf.float32)
        _ar_prob = tf.minimum(0.35, 0.0 + _ar_step / 50000.0 * 0.35)
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
                if self.flow_routing_gate is not None:
                    pred = self.flow_routing_gate(pred, geospatial)
                if self.learned_routing_gate is not None:
                    pred = self.learned_routing_gate(
                        pred, geospatial, elevation_range_m
                    )

                step_loss = self.compute_loss(y=yk, y_pred=pred)

                # Mass preservation: penalise mismatch in mean depth
                pred_mass = tf.reduce_mean(pred, axis=[1, 2, 3])
                gt_mass = tf.reduce_mean(yk, axis=[1, 2, 3])
                mass_loss = tf.reduce_mean(tf.square(pred_mass - gt_mass))

                # Gravity consistency: penalise isolated WSE peaks unsupported
                # by terrain (water appearing from nowhere) -- see customloss.py
                if self._params.gravity_loss_weight > 0:
                    gravity_loss = customloss.gravity_consistency_loss(pred, geospatial)
                else:
                    gravity_loss = 0.0

                # Infiltration consistency: penalise false-positive flooding
                # on high-infiltration soil -- see customloss.py
                if self._params.infiltration_loss_weight > 0:
                    infiltration_loss = customloss.infiltration_consistency_loss(
                        yk, pred, geospatial
                    )
                else:
                    infiltration_loss = 0.0

                # Later steps matter more (error should not accumulate)
                time_weight = 1.0 + 0.3 * k

                total_loss += time_weight * (
                    step_loss
                    + 0.2 * mass_loss
                    + self._params.gravity_loss_weight * gravity_loss
                    + self._params.infiltration_loss_weight * infiltration_loss
                )

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
        elevation_range_m = x.get("elevation_range_m")

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
            if self.flow_routing_gate is not None:
                pred = self.flow_routing_gate(pred, geospatial)
            if self.learned_routing_gate is not None:
                pred = self.learned_routing_gate(pred, geospatial, elevation_range_m)

            step_loss = self.compute_loss(y=yk, y_pred=pred)

            pred_mass = tf.reduce_mean(pred, axis=[1, 2, 3])
            gt_mass = tf.reduce_mean(yk, axis=[1, 2, 3])
            mass_loss = tf.reduce_mean(tf.square(pred_mass - gt_mass))

            if self._params.gravity_loss_weight > 0:
                gravity_loss = customloss.gravity_consistency_loss(pred, geospatial)
            else:
                gravity_loss = 0.0

            if self._params.infiltration_loss_weight > 0:
                infiltration_loss = customloss.infiltration_consistency_loss(
                    yk, pred, geospatial
                )
            else:
                infiltration_loss = 0.0

            time_weight = 1.0 + 0.3 * k
            total_loss += time_weight * (
                step_loss
                + 0.2 * mass_loss
                + self._params.gravity_loss_weight * gravity_loss
                + self._params.infiltration_loss_weight * infiltration_loss
            )

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

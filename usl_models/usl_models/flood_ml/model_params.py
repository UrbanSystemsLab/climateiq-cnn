"""Flood model parameters."""

from dataclasses import dataclass

import tensorflow as tf

from usl_models.flood_ml import constants


@dataclass(kw_only=True)
class FloodModelParams:
    # General parameters.
    batch_size: int = 64
    m_rainfall: int = constants.M_RAINFALL
    n_flood_maps: int = constants.N_FLOOD_MAPS

    # Layer-specific parameters.
    lstm_units: int = 128
    lstm_kernel_size: int = 3
    lstm_dropout: float = 0.2
    lstm_recurrent_dropout: float = 0.2

    # Keras parameters.
    # While Keras models support (valid) string names, we choose to only accept
    # the appropriate object instances for better type checking and to be
    # explicit about hyperparameters (e.g., within the optimizer).
    optimizer: tf.keras.Optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    epochs: int = 10


# Used for testing.
test_model_params = FloodModelParams(
    batch_size=4,
    m_rainfall=3,
    n_flood_maps=3,
    lstm_units=32,
    lstm_kernel_size=3,
    epochs=1,
)

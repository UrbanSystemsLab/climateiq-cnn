"""Flood model parameters."""

from dataclasses import dataclass

import tensorflow as tf

from usl_models.flood_ml import constants


@dataclass(kw_only=True, slots=True)
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
    # Keras models support both Optimizer objects and (valid) string names.
    # It is the user's responsibility to pass in a valid optimizer value.
    optimizer: tf.keras.Optimizer | str = "adam"
    epochs: int = 10

"""AtmoML model parameters."""

from dataclasses import dataclass

import tensorflow as tf


@dataclass(kw_only=True, slots=True)
class AtmoModelParams:
    # General parameters.
    batch_size: int = 64

    # Layer-specific parameters.
    lstm_units: int = 512
    lstm_kernel_size: int = 5
    lstm_dropout: float = 0.2
    lstm_recurrent_dropout: float = 0.2

    # Keras parameters.
    # Keras models support both Optimizer objects and (valid) string names.
    # It is the user's responsibility to pass in a valid optimizer value.
    optimizer: tf.keras.Optimizer | str = "adam"
    epochs: int = 10
"""Flood model parameters."""

from typing import Any, Mapping, TypedDict

from usl_models.flood_ml import constants


class FloodModelParams(TypedDict):
    # General parameters.
    batch_size: int
    m_rainfall: int
    n_flood_maps: int

    # Layer-specific parameters.
    lstm_units: int
    lstm_kernel_size: int
    lstm_dropout: float
    lstm_recurrent_dropout: float

    # The optimizer configuration.
    # We use the dictionary definition to ensure the model is serializable.
    # This value is passed to tf.keras.optimizers.get to build the optimizer object.
    optimizer_config: Mapping[str, Any]


def default_params() -> FloodModelParams:
    """Creates default model parameter values."""
    return {
        "batch_size": 4,
        "m_rainfall": constants.M_RAINFALL,
        "n_flood_maps": constants.N_FLOOD_MAPS,
        "lstm_units": 128,
        "lstm_kernel_size": 3,
        "lstm_dropout": 0.2,
        "lstm_recurrent_dropout": 0.2,
        "optimizer_config": {
            "class_name": "Adam",
            "config": {"learning_rate": 1e-3},
        },
    }

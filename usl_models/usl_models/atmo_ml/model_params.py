"""AtmoML model parameters."""

from typing import Any, Mapping, TypedDict


class AtmoModelParams(TypedDict):
    # General parameters.
    batch_size: int

    # Layer-specific parameters.
    lstm_units: int
    lstm_kernel_size: int
    lstm_dropout: float
    lstm_recurrent_dropout: float

    # The optimizer configuration.
    # We use the dictionary definition to ensure the model is serializable.
    # This value is passed to tf.keras.optimizers.get to build the optimizer object.
    optimizer_config: Mapping[str, Any]

    epochs: int


def default_params() -> AtmoModelParams:
    """Creates default model parameter values."""
    return {
        "batch_size": 4,
        "lstm_units": 512,
        "lstm_kernel_size": 5,
        "lstm_dropout": 0.2,
        "lstm_recurrent_dropout": 0.2,
        "optimizer_config": {
            "class_name": "Adam",
            "config": {"learning_rate": 5e-3},
        },
        "epochs": 10,
    }

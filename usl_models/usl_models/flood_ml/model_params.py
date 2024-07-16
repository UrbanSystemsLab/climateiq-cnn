"""Flood model parameters."""

import dataclasses
from typing import Any, Mapping

from usl_models.flood_ml import constants


@dataclasses.dataclass(kw_only=True, slots=True)
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

    # The optimizer configuration.
    # We use the dictionary definition to ensure the model is serializable.
    optimizer_config: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "class_name": "Adam",
            "config": {"learning_rate": 1e-3},
        }
    )

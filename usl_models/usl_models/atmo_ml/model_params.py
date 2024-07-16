"""AtmoML model parameters."""

import dataclasses
from typing import Any, Mapping


@dataclasses.dataclass(kw_only=True, slots=True)
class AtmoModelParams:
    # General parameters.
    batch_size: int = 64

    # Layer-specific parameters.
    lstm_units: int = 512
    lstm_kernel_size: int = 5
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

    epochs: int = 10

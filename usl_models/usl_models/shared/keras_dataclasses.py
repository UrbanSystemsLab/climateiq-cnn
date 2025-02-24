"""Keras serializable dataclass."""

import copy
import dataclasses
from typing import Any

import keras


def get_config(self) -> dict[str, Any]:
    """Get config for keras serialization."""
    # Make a shallow copy so we can replace fields without
    # the mutating original.
    self_copy = copy.copy(self)

    serialized_config = {}
    # Update fields that need keras specific serialization.
    for f in dataclasses.fields(self):
        if issubclass(f.type, keras.optimizers.Optimizer):
            serialized_config[f.name] = keras.optimizers.serialize(
                getattr(self, f.name)
            )
            setattr(self_copy, f.name, None)

    # Run standard dataclass asdict.
    config = dataclasses.asdict(self_copy)
    config.update(serialized_config)
    return config


def from_config(cls, config: dict[str, Any]) -> Any:
    """Construct from config for keras deserialization."""
    # Make a shallow copy so we can replace fields without
    # the mutating original.
    config_copy = copy.copy(config)

    # Update fields that need keras specific deserialization.
    for f in dataclasses.fields(cls):
        if issubclass(f.type, keras.optimizers.Optimizer):
            config_copy[f.name] = keras.optimizers.get(config[f.name])

    # Run standard constructor.
    return cls(**config_copy)


def dataclass(package: str = "Custom", name: str | None = None, **kwargs):
    """Custom dataclass decorator that support keras serialization."""

    def decorator(cls):
        # Add dataclass functionality.
        cls = dataclasses.dataclass(**kwargs)(cls)

        # Support keras serialization / deserialization.
        cls.get_config = get_config
        cls.from_config = from_config
        cls = keras.saving.register_keras_serializable(package=package, name=name)(cls)

        return cls

    return decorator

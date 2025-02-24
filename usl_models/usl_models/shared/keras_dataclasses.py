"""Keras serializable dataclass."""

import copy
import dataclasses
from typing import Any, TypeVar, Type, Callable, ClassVar, Protocol, dataclass_transform

import keras


class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]


class KerasDataclass(DataclassInstance):
    def get_config(self) -> dict[str, Any]:
        """Get config for keras serialization."""
        # Make a shallow copy so we can replace fields without
        # the mutating original.
        self_copy = copy.copy(self)

        # Update fields that need keras specific serialization.
        serialized_config = {}
        for f in dataclasses.fields(self):
            if isinstance(f.type, type) and issubclass(
                f.type, keras.optimizers.Optimizer
            ):
                serialized_config[f.name] = keras.optimizers.serialize(
                    getattr(self, f.name)
                )
                setattr(self_copy, f.name, None)

        # Run standard dataclass asdict.
        config = dataclasses.asdict(self_copy)
        config.update(serialized_config)
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Any:
        """Construct from config for keras deserialization."""
        # Make a shallow copy so we can replace fields without
        # the mutating original.
        config_copy = copy.copy(config)

        # Update fields that need keras specific deserialization.
        for f in dataclasses.fields(cls):  # type: ignore
            if isinstance(f.type, type) and issubclass(
                f.type, keras.optimizers.Optimizer
            ):
                config_copy[f.name] = keras.optimizers.get(config[f.name])

        # Run standard constructor.
        return cls(**config_copy)


T = TypeVar("T")


@dataclass_transform(field_specifiers=(dataclasses.Field,))
def dataclass(
    package: str = "Custom", name: str | None = None, **kwargs
) -> Callable[[Type[T]], Type[T]]:
    """Custom dataclass decorator that support keras serialization."""

    def decorator(cls: Type[T]) -> Type[T]:
        cls = dataclasses.dataclass(**kwargs)(cls)
        cls = keras.saving.register_keras_serializable(package=package, name=name)(cls)
        return cls

    return decorator

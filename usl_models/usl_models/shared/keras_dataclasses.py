"""Keras serializable dataclass."""

import copy
import dataclasses
from typing import (
    Any,
    TypeVar,
    Type,
    Callable,
    ClassVar,
    Protocol,
    dataclass_transform,
)

import keras.src.saving.legacy.saved_model.utils

import keras


class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]


class Base(DataclassInstance):
    """Base Keras Dataclass.

    Usage:
    ```
    import keras_dataclasses

    @keras_dataclasses.dataclass()
    class Params(keras_dataclasses.Base):
      field1: int
      field2: float
    ```
    """

    def get_config(self) -> dict[str, Any]:
        """Get config for keras serialization."""
        config = {}

        # Force non-legacy serialization.
        with keras.src.saving.legacy.saved_model.utils.keras_option_scope(
            save_traces=False, in_tf_saved_model_scope=False
        ):
            for f in dataclasses.fields(self):
                config[f.name] = keras.saving.serialize_keras_object(
                    getattr(self, f.name)
                )

        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Any:
        """Construct from config for keras deserialization."""
        # Make a shallow copy so we can replace fields without
        # the mutating original.
        config_copy = copy.copy(config)

        # Update fields that need keras specific deserialization.
        for f in dataclasses.fields(cls):  # type: ignore
            config_copy[f.name] = keras.saving.deserialize_keras_object(
                config_copy[f.name]
            )

        # Run standard constructor.
        return cls(**config_copy)


T = TypeVar("T", bound=Base)


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

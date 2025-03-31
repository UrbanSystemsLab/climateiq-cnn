import os
from typing import Type, TypeVar


T = TypeVar("T")


def getenv(key: str, val_type: Type[T], default: T | None = None):
    """Gets a typed environment variable.

    Args:
      key: Environment varaible key.
      val_type: Type to cast the string value to.
      default: The default value to use if the environment variable isn't set.

    Returns: The environment variable cast to `val_type`.
    """
    return val_type(os.getenv(key, default=default))  # type: ignore

import builtins
import unittest
from typing import Any
import numpy as np
import tensorflow as tf


class TestCase(unittest.TestCase):
    """Testing utilss."""

    def assertShapesRecursive(self, obj: Any, expected: Any, path: str = ""):
        """Recursively checks the shapes of numpy arrays in a data structure."""
        if isinstance(obj, np.ndarray) or tf.is_tensor(obj):
            self.assertEqual(obj.shape,
                             expected, msg=f"Shape mismatch in path {path}")
            return

        self.assertEqual(type(obj), type(expected), msg=f"Type mismatch in path {path}")

        match type(obj):
            case builtins.list:
                self.assertEqual(len(obj), len(expected))
                for i, (o, e) in enumerate(zip(obj, expected)):
                    self.assertShapesRecursive(o, e, path + f"[{i}]")
            case builtins.tuple:
                for i, (o, e) in enumerate(zip(obj, expected)):
                    self.assertShapesRecursive(o, e, path + f"[{i}]")
            case builtins.dict:
                assert set(obj.keys()) == set(expected.keys())
                for k in obj:
                    self.assertShapesRecursive(obj[k], expected[k], path + f"['{k}']")

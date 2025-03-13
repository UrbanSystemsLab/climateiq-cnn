from keras import layers
import tensorflow as tf
from typing import Tuple


class Pad2D(layers.Layer):
    """Implements tf.pad as a layer for 2D tensors.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.
        mode: Padding mode.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(
        self, padding: Tuple[int, int] = (1, 1), mode: str = "REFLECT", **kwargs
    ):
        """Constructor."""
        self.padding = padding
        self.mode = mode
        super(Pad2D, self).__init__(**kwargs)

    def call(self, x: tf.Tensor, mask=None):
        """Call layer."""
        h_pad, w_pad = self.padding
        return tf.pad(
            x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], mode=self.mode
        )

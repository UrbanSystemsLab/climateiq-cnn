from keras import layers
import tensorflow as tf


class ReflectionPadding2D(layers.Layer):
    """Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), **kwargs):
        """Constructor."""
        self.padding = tuple(padding)
        # self.input_spec = [layers.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, shape):
        """Compute output shape."""
        h_pad, w_pad = self.padding
        B, H, W, *S = shape
        return (B, H + 2 * h_pad, W + 2 * w_pad, *S)

    def call(self, x, mask=None):
        """Call layer."""
        h_pad, w_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], "REFLECT")

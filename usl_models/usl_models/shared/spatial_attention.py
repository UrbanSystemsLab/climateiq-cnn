from keras.saving import register_keras_serializable
from keras import layers
import tensorflow as tf


@register_keras_serializable()
class SpatialAttention(layers.Layer):
    def __init__(self, **kwargs):
        """Initialize the spatial attention instance."""
        super().__init__(**kwargs)
        self.conv = layers.TimeDistributed(
            layers.Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")
        )

    def call(self, inputs):
        """Compute the attention weights."""
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)
        return inputs * attention

    def get_config(self):
        """Serialize from dict."""
        base_config = super().get_config()
        return base_config

    @classmethod
    def from_config(cls, config):
        """Deserialize from dict."""
        return cls(**config)

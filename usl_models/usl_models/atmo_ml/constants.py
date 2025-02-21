"""Constant definitions for the AtmoML model."""

import tensorflow as tf


# Geospatial constants
MAP_HEIGHT = 200
MAP_WIDTH = 200

# Other well-defined constants
INPUT_TIME_STEPS = 6
OUTPUT_CHANNELS = 5
OUTPUT_TIME_STEPS = 8
NUM_SAPTIAL_FEATURES = 22
NUM_SPATIOTEMPORAL_FEATURES = 12
TIME_STEPS_PER_DAY = 4
LU_INDEX_VOCAB_SIZE = 61
EMBEDDING_DIM = 8


def get_input_shape_batched(height, width):
    spec = get_input_spec(height, width)
    return {k: (None, *v.shape) for k, v in spec.items()}


def get_input_spec(height: int | None, width: int | None) -> dict[str, tf.TypeSpec]:
    return {
        "spatiotemporal": tf.TensorSpec(
            shape=(
                INPUT_TIME_STEPS,
                height,
                width,
                NUM_SPATIOTEMPORAL_FEATURES,
            ),
            dtype=tf.float32,
        ),
        "spatial": tf.TensorSpec(
            shape=(
                height,
                width,
                NUM_SAPTIAL_FEATURES,
            ),
            dtype=tf.float32,
        ),
        "lu_index": tf.TensorSpec(
            shape=(
                height,
                width,
            ),
            dtype=tf.int32,
        ),
        "sim_name": tf.TensorSpec(shape=(), dtype=tf.string),
        "date": tf.TensorSpec(shape=(), dtype=tf.string),
    }


def get_output_spec(height: int, width: int, timesteps: int) -> tf.TensorSpec:
    return tf.TensorSpec(
        shape=(
            timesteps,
            height,
            width,
            OUTPUT_CHANNELS,
        ),
        dtype=tf.float32,
    )

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

INPUT_SHAPE = {
    "spatiotemporal": (
        INPUT_TIME_STEPS,
        MAP_HEIGHT,
        MAP_WIDTH,
        NUM_SPATIOTEMPORAL_FEATURES,
    ),
    "spatial": (
        MAP_HEIGHT,
        MAP_WIDTH,
        NUM_SAPTIAL_FEATURES,
    ),
    "lu_index": (
        MAP_HEIGHT,
        MAP_WIDTH,
    ),
}

INPUT_SHAPE_BATCHED = {k: (None, *v) for k, v in INPUT_SHAPE.items()}

INPUT_SPEC = {
    "spatiotemporal": tf.TensorSpec(
        shape=(
            INPUT_TIME_STEPS,
            MAP_HEIGHT,
            MAP_WIDTH,
            NUM_SPATIOTEMPORAL_FEATURES,
        ),
        dtype=tf.float32,
    ),
    "spatial": tf.TensorSpec(
        shape=(
            MAP_HEIGHT,
            MAP_WIDTH,
            NUM_SAPTIAL_FEATURES,
        ),
        dtype=tf.float32,
    ),
    "lu_index": tf.TensorSpec(
        shape=(
            MAP_HEIGHT,
            MAP_WIDTH,
        ),
        dtype=tf.int32,
    ),
}

OUTPUT_SPEC = tf.TensorSpec(
    shape=(
        OUTPUT_TIME_STEPS,
        MAP_HEIGHT,
        MAP_WIDTH,
        OUTPUT_CHANNELS,
    ),
    dtype=tf.float32,
)

"""Constant definitions for the AtmoML model."""

# Geospatial constants
MAP_HEIGHT = 200
MAP_WIDTH = 200

# Other well-defined constants
INPUT_TIME_STEPS = 6
OUTPUT_TIME_STEPS = 2
NUM_SPATIAL_FEATURES = (
    22  # there is no change here just it was a typo so fixes the variable name
)
NUM_SAPTIAL_FEATURES = NUM_SPATIAL_FEATURES
NUM_SPATIOTEMPORAL_FEATURES = 12
TIME_STEPS_PER_DAY = 4
LU_INDEX_VOCAB_SIZE = 61
EMBEDDING_DIM = 8

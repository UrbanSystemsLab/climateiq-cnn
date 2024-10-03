"""Dataset processing for AtmoML model."""

import tensorflow as tf
from usl_models.atmo_ml import constants
from usl_models.atmo_ml import input_output_sequences


def process_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Processes the dataset and generates input/output sequences."""
    processed_data = []

    for data in dataset:
        inputs, labels = data

        # Extract individual components from inputs
        spatial_input = inputs["spatial"]
        spatiotemporal_input = inputs["spatiotemporal"]
        lu_index_input = inputs["lu_index"]

        # Process the sequence
        sequence_generator = input_output_sequences.create_input_output_sequences(
            spatiotemporal_input, labels
        )

        # Append the processed inputs (as a dictionary) and output sequences
        for input_sequence, output_sequence in sequence_generator:
            processed_data.append(
                (
                    {
                        "spatial": spatial_input,
                        "spatiotemporal": input_sequence,
                        "lu_index": lu_index_input,
                    },
                    output_sequence,
                )
            )

    return tf.data.Dataset.from_generator(
        lambda: iter(processed_data),
        output_signature=(
            {
                "spatial": tf.TensorSpec(
                    shape=(
                        None,
                        constants.MAP_HEIGHT,
                        constants.MAP_WIDTH,
                        constants.num_spatial_features,
                    ),
                    dtype=tf.float32,
                ),
                "spatiotemporal": tf.TensorSpec(
                    shape=(
                        None,
                        constants.INPUT_TIME_STEPS,
                        constants.MAP_HEIGHT,
                        constants.MAP_WIDTH,
                        constants.num_spatiotemporal_features,
                    ),
                    dtype=tf.float32,
                ),
                "lu_index": tf.TensorSpec(
                    shape=(None, constants.MAP_HEIGHT, constants.MAP_WIDTH),
                    dtype=tf.int32,
                ),
            },
            tf.TensorSpec(
                shape=(
                    None,
                    constants.OUTPUT_TIME_STEPS,
                    constants.MAP_HEIGHT,
                    constants.MAP_WIDTH,
                    constants.OUTPUT_CHANNELS,
                ),
                dtype=tf.float32,
            ),
        ),
    )

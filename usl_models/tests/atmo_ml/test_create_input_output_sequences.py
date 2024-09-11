# test_create_input_output_sequences.py

import tensorflow as tf
from usl_models.atmo_ml import input_output_sequences


def test_create_input_output_sequences():
    # Define test parameters
    batch_size = 1
    height = 1
    width = 1
    channels = 1
    time_steps_per_day = 4
    num_days = 2
    total_time_steps = time_steps_per_day * num_days

    # Create fake input data and labels for clarity
    inputs = tf.constant(
        [
            [
                [[[1]]],
                [[[2]]],
                [[[3]]],
                [[[4]]],  # Day 1: 0:00, 6:00, 12:00, 18:00
                [[[5]]],
                [[[6]]],
                [[[7]]],
                [[[8]]],  # Day 2: 0:00, 6:00, 12:00, 18:00
            ]
        ]
    )  # Shape [1, 8, 1, 1, 1]

    labels = tf.constant(
        [
            [
                [[[10]]],
                [[[11]]],
                [[[12]]],
                [[[13]]],  # Day 1 outputs
                [[[14]]],
                [[[15]]],
                [[[16]]],
                [[[17]]],  # Day 2 outputs
            ]
        ]
    )  # Shape [1, 8, 1, 1, 1]

    # Call the function to generate input-output sequences
    input_sequences, output_sequences = (
        input_output_sequences.create_input_output_sequences(inputs, labels)
    )

    # Expected input sequences:
    expected_input_sequences = tf.constant(
        [
            [
                [[[0]], [[1]], [[2]]],  # (X_{d-1}^{18}, X^0, X^6)
                [[[1]], [[2]], [[3]]],  # (X^0, X^6, X^{12})
                [[[2]], [[3]], [[4]]],  # (X^6, X^{12}, X^{18})
                [[[3]], [[4]], [[5]]],  # (X^{12}, X^{18}, X_{d+1}^0)
            ]
        ]
    )  # Shape [1, 4, 3, 1, 1, 1]

    # Expected output sequences:
    expected_output_sequences = tf.constant(
        [
            [
                [[[10]], [[11]]],  # (Y^0, Y^3)
                [[[11]], [[12]]],  # (Y^6, Y^9)
                [[[12]], [[13]]],  # (Y^{12}, Y^{15})
                [[[13]], [[14]]],  # (Y^{18}, Y^{21})
            ]
        ]
    )  # Shape [1, 4, 2, 1, 1, 1]

    # Check if the shapes and values are as expected
    assert (
        input_sequences.shape == expected_input_sequences.shape
    ), f"Expected input shape {expected_input_sequences.shape}, but got {input_sequences.shape}"
    assert (
        output_sequences.shape == expected_output_sequences.shape
    ), f"Expected output shape {expected_output_sequences.shape}, but got {output_sequences.shape}"

    # Verify specific values
    assert tf.reduce_all(
        tf.equal(input_sequences, expected_input_sequences)
    ), f"Expected input sequences {expected_input_sequences.numpy().tolist()}, but got {input_sequences.numpy().tolist()}"
    assert tf.reduce_all(
        tf.equal(output_sequences, expected_output_sequences)
    ), f"Expected output sequences {expected_output_sequences.numpy().tolist()}, but got {output_sequences.numpy().tolist()}"

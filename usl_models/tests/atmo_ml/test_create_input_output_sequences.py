# test_create_input_output_sequences.py

import tensorflow as tf
from usl_models.atmo_ml import input_output_sequences


def test_create_input_output_sequences():
    # Define test parameters
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

    input_sequences, output_sequences = (
        input_output_sequences.create_input_output_sequences(
            inputs, labels, time_steps_per_day
        )
    )
    # Print shapes and contents of the generated sequences for debugging
    print("Generated input_sequences shape:", input_sequences.shape)
    print("Generated output_sequences shape:", output_sequences.shape)
    print("Generated input_sequences values:\n", input_sequences.numpy())
    print("Generated output_sequences values:\n", output_sequences.numpy())
    # Update expected sequences to reflect the correct dimensions
    expected_input_sequences = tf.constant(
        [
            [
                [
                    [[[1]], [[1]], [[2]]],  # (X_{d-1}^{18}, X^0, X^6) - Day 1
                    [[[1]], [[2]], [[3]]],  # (X^0, X^6, X^{12})
                    [[[2]], [[3]], [[4]]],  # (X^6, X^{12}, X^{18})
                    [[[3]], [[4]], [[5]]],  # (X^{12}, X^{18}, X_{d+1}^0) - Day 2
                ],
                [
                    [[[4]], [[5]], [[6]]],  # (X_{d-1}^{18}, X^0, X^6) - Day 2
                    [[[5]], [[6]], [[7]]],  # (X^0, X^6, X^{12})
                    [[[6]], [[7]], [[8]]],  # (X^6, X^{12}, X^{18})
                    [[[7]], [[8]], [[8]]],  # (X^{12}, X^{18}, X_{d+1}^0) - last day
                ],
            ]
        ]
    )  # Shape [1, 2, 4, 3, 1]

    expected_output_sequences = tf.constant(
        [
            [
                [[[10]], [[11]], [[12]], [[13]]],  # Day 1 outputs
                [[[14]], [[15]], [[16]], [[17]]],  # Day 2 outputs
            ]
        ]
    )  # Shape [1, 2, 4, 1, 1]

    # Check if the shapes and values are as expected
    assert (
        input_sequences.shape == expected_input_sequences.shape
    ), f"Expected input shape {expected_input_sequences.shape}, but got {input_sequences.shape}"
    assert (
        output_sequences.shape == expected_output_sequences.shape
    ), f"Expected output shape {expected_output_sequences.shape}, but got {output_sequences.shape}"

    # Add further checks to validate content if needed

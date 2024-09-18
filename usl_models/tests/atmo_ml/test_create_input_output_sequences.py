from usl_models.atmo_ml import input_output_sequences

import tensorflow as tf


def test_create_input_output_sequences():
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
                [[[13]]],
                [[[14]]],
                [[[15]]],
                [[[16]]],
                [[[17]]],  # Day 1 outputs
                [[[18]]],
                [[[19]]],
                [[[20]]],
                [[[21]]],
                [[[22]]],
                [[[23]]],
                [[[24]]],
                [[[25]]],  # Day 2 outputs
            ]
        ]
    )  # Shape [1, 8, 1, 1, 1]

    # Updated expected input sequences for comparison based on generated output
    expected_inputs = [
        [1, 1, 2],  # Day 1
        [1, 2, 3],  # Day 1
        [2, 3, 4],  # Day 1
        [3, 4, 5],  # Day 1
        [4, 5, 6],  # Day 2
        [5, 6, 7],  # Day 2
        [6, 7, 8],  # Day 2
        [7, 8, 8],  # Day 2
    ]

    # Updated expected output sequences for comparison based on generated output
    expected_outputs = [
        [10, 11],  # Day 1
        [12, 13],  # Day 1
        [14, 15],  # Day 1
        [16, 17],  # Day 1
        [18, 19],  # Day 2
        [20, 21],  # Day 2
        [22, 23],  # Day 2
        [24, 25],  # Day 2
    ]

    generated_inputs = []
    generated_outputs = []

    # Call the function and collect sequences inside
    for input_seq, output_seq in input_output_sequences.create_input_output_sequences(
        inputs, labels
    ):
        # Simplify the sequences for direct comparison with expected values
        generated_inputs.append(input_seq.numpy().flatten().tolist())
        generated_outputs.append(output_seq.numpy().flatten().tolist())

    # Check if generated sequences match expected sequences
    for gen_input, exp_input in zip(generated_inputs, expected_inputs):
        assert (
            gen_input == exp_input
        ), f"Expected input {exp_input}, but got {gen_input}"

    for gen_output, exp_output in zip(generated_outputs, expected_outputs):
        assert (
            gen_output == exp_output
        ), f"Expected output {exp_output}, but got {gen_output}"

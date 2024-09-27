import tensorflow as tf
from usl_models.atmo_ml import cnn_inputs_outputs


def test_divide_into_days():
    # Sample input and label tensors
    inputs = tf.constant(
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
    )  # Shape [8, 1, 1, 1]

    labels = tf.constant(
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
    )  # Shape [16, 1, 1, 1]

    # Expected outputs
    expected_day_inputs_list = [
        tf.constant(
            [[[[1]]], [[[1]]], [[[2]]], [[[3]]], [[[4]]], [[[5]]]]
        ),  # Day 1 inputs
        tf.constant(
            [[[[4]]], [[[5]]], [[[6]]], [[[7]]], [[[8]]], [[[8]]]]
        ),  # Day 2 inputs
    ]

    expected_day_labels_list = [
        tf.constant(
            [
                [[[10]]],
                [[[10]]],
                [[[10]]],
                [[[11]]],
                [[[12]]],
                [[[13]]],
                [[[14]]],
                [[[15]]],
                [[[16]]],
                [[[17]]],
            ]
        ),  # Day 1 labels
        tf.constant(
            [
                [[[16]]],
                [[[17]]],
                [[[18]]],
                [[[19]]],
                [[[20]]],
                [[[21]]],
                [[[22]]],
                [[[23]]],
                [[[24]]],
                [[[25]]],
            ]
        ),  # Day 2 labels
    ]

    # Call the function to process inputs and labels
    day_inputs_list, day_labels_list = cnn_inputs_outputs.divide_into_days(
        inputs, labels, input_steps_per_day=4, label_steps_per_day=8
    )

    # Assert if generated inputs and labels match the expected results
    for i, (gen_inputs, exp_inputs) in enumerate(
        zip(day_inputs_list, expected_day_inputs_list)
    ):
        assert tf.reduce_all(
            tf.equal(gen_inputs, exp_inputs)
        ), f"Day {i + 1} Exp Inputs: {exp_inputs.numpy()}, got: {gen_inputs.numpy()}"

    for i, (gen_labels, exp_labels) in enumerate(
        zip(day_labels_list, expected_day_labels_list)
    ):
        assert tf.reduce_all(
            tf.equal(gen_labels, exp_labels)
        ), f"Day {i + 1} Exp labels: {exp_labels.numpy()}, got: {gen_labels.numpy()}"

    print("All tests passed!")

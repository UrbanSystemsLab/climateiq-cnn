from usl_models.atmo_ml import data_utils

from numpy import testing as np_testing
import tensorflow as tf


def test_boundary_pairs():
    """Tests the pairing on temporal data."""
    # Input:  [1, 3, 2, 2, 3]
    # [
    #     [[[0, 1, 2],    [3, 4, 5]],     [[6, 7, 8],     [9, 10, 11]]],
    #     [[[12, 13, 14], [15, 16, 17]],  [[18, 19, 20],  [21, 22, 23]]],
    #     [[[24, 25, 26], [27, 28, 29]],  [[30, 31, 32],  [33, 34, 35]]],
    # ]
    input = tf.reshape(tf.range(36), (1, 3, 2, 2, 3))

    # Output: [1, 2, 2, 2, 6]
    # Adding the batch dimension
    expected_output = tf.constant(
        [
            [
                [
                    [
                        [0, 1, 2, 12, 13, 14],
                        [3, 4, 5, 15, 16, 17],
                    ],
                    [
                        [6, 7, 8, 18, 19, 20],
                        [9, 10, 11, 21, 22, 23],
                    ],
                ],
                [
                    [
                        [12, 13, 14, 24, 25, 26],
                        [15, 16, 17, 27, 28, 29],
                    ],
                    [
                        [18, 19, 20, 30, 31, 32],
                        [21, 22, 23, 33, 34, 35],
                    ],
                ],
            ],
        ]
    )

    pairs = data_utils.boundary_pairs(input)
    np_testing.assert_array_equal(pairs, expected_output)


def test_split_time_step_pairs():
    """Tests the time step splitting."""
    # Input:  [1, 3, 2, 2, 4]
    # [
    #     [[[0, 1, 2, 3], [4, 5, 6, 7]], [[8, 9, 10, 11], [12, 13, 14, 15]]],
    #     [[[16, 17, 18, 19], [20, 21, 22, 23]], [[24, 25, 26, 27], [28, 29, 30, 31]]],
    #     [[[32, 33, 34, 35], [36, 37, 38, 39]], [[40, 41, 42, 43], [44, 45, 46, 47]]],
    # ]
    input = tf.reshape(tf.range(48), (1, 3, 2, 2, 4))

    # Output: [1, 6, 2, 2, 2]
    # adding batch dimension
    expected_output = tf.constant(
        [
            [
                [
                    [[0, 1], [4, 5]],
                    [[8, 9], [12, 13]],
                ],
                [
                    [[2, 3], [6, 7]],
                    [[10, 11], [14, 15]],
                ],
                [
                    [[16, 17], [20, 21]],
                    [[24, 25], [28, 29]],
                ],
                [
                    [[18, 19], [22, 23]],
                    [[26, 27], [30, 31]],
                ],
                [
                    [[32, 33], [36, 37]],
                    [[40, 41], [44, 45]],
                ],
                [
                    [[34, 35], [38, 39]],
                    [[42, 43], [46, 47]],
                ],
            ],
        ]
    )

    split_pairs = data_utils.split_time_step_pairs(input)
    np_testing.assert_array_equal(split_pairs, expected_output)

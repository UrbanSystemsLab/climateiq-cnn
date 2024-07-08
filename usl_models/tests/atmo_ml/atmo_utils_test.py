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
    expected_output = (
        [
            [
                [
                    [0, 12, 1, 13, 2, 14],
                    [3, 15, 4, 16, 5, 17],
                ],
                [
                    [6, 18, 7, 19, 8, 20],
                    [9, 21, 10, 22, 11, 23],
                ],
            ],
            [
                [
                    [12, 24, 13, 25, 14, 26],
                    [15, 27, 16, 28, 17, 29],
                ],
                [
                    [18, 30, 19, 31, 20, 32],
                    [21, 33, 22, 34, 23, 35],
                ],
            ],
        ],
    )

    pairs = data_utils.boundary_pairs(input)
    np_testing.assert_array_equal(pairs, expected_output)

from flood_ml import data_utils

from numpy import testing as np_testing
import tensorflow as tf


def test_temporal_window_view():
    """Tests the sliding window view on temporal data."""
    # Input:  [2, 4]
    # Output: [2, 4, 3]
    input = tf.reshape(tf.range(8), (2, 4))
    window = 3

    expected_output = [
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 2],
                [1, 2, 3],
            ],
            [
                [0, 0, 4],
                [0, 4, 5],
                [4, 5, 6],
                [5, 6, 7],
            ],
        ]

    window_output = data_utils.temporal_window_view(input, window)
    np_testing.assert_array_equal(window_output, expected_output)

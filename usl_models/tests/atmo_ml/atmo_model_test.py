"""Tests for Atmo model."""

import tensorflow as tf

from usl_models.atmo_ml import model as atmo_model
from usl_models.atmo_ml import constants
from usl_models.atmo_ml import model_params

_TEST_MAP_HEIGHT = 100
_TEST_MAP_WIDTH = 100
_TEST_SPATIAL_FEATURES = 20
_TEST_SPATIOTEMPORAL_FEATURES = 6


def pytest_model_params() -> model_params.AtmoModelParams:
    """Defines AtmoModelParams for testing."""
    params = model_params.default_params()
    params.update(
        {
            "batch_size": 4,
            "lstm_units": 32,
            "lstm_kernel_size": 3,
            "epochs": 1,
        }
    )
    return params


def fake_input_batch(
    batch_size: int,
    height: int = _TEST_MAP_HEIGHT,
    width: int = _TEST_MAP_WIDTH,
) -> atmo_model.AtmoInput:
    """Creates a fake training batch for testing.

    Args:
        batch_size: Batch size.
        height: Optional height.
        width: Optional width.

    Returns:
        A dictionary of model inputs, with the following key/value pairs:
        - "spatial": Required. A rank-4 tensor [batch, height, width, features].
        - "spatiotemporal": Required. A rank-5 tensor
            [batch, time, height, width, features].
    """
    spatial = tf.random.normal((batch_size, height, width, _TEST_SPATIAL_FEATURES))
    spatiotemporal = tf.random.normal(
        (
            batch_size,
            constants.INPUT_TIME_STEPS,
            height,
            width,
            _TEST_SPATIOTEMPORAL_FEATURES,
        )
    )

    return {
        "spatial": spatial,
        "spatiotemporal": spatiotemporal,
    }


def test_atmo_convlstm():
    """Tests the AtmoConvLSTM model call."""
    batch_size = 4
    params = pytest_model_params()

    fake_input = fake_input_batch(batch_size)

    model = atmo_model.AtmoConvLSTM(
        params,
        spatial_dims=(_TEST_MAP_HEIGHT, _TEST_MAP_WIDTH),
        num_spatial_features=_TEST_SPATIAL_FEATURES,
        num_spatiotemporal_features=_TEST_SPATIOTEMPORAL_FEATURES,
    )
    prediction = model(fake_input)

    expected_output_shape = (
        batch_size,
        constants.OUTPUT_TIME_STEPS,
        _TEST_MAP_HEIGHT,
        _TEST_MAP_WIDTH,
        constants.OUTPUT_CHANNELS,  # T2, RH2, WSPD10, WDIR10_SIN, WDIR10_COS
    )

    assert prediction.shape == expected_output_shape

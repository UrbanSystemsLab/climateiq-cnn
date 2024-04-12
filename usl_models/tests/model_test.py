"""Tests for Flood model."""

import tensorflow as tf

from usl_models.flood_ml import constants
from usl_models.flood_ml import data_utils
from usl_models.flood_ml import model as flood_model

_MODEL_PARAMS = {
    "lstm_units": 32,
    "lstm_kernels": 4,
    "lstm_dropout": 0.2,
    "lstm_recurrent_dropout": 0.2,
    "m_rainfall": 3,
}


def fake_flood_input_batch(
    batch_size: int,
    height: int = constants.MAP_HEIGHT,
    width: int = constants.MAP_WIDTH,
    rainfall_duration: int = 24,
    flood_maps: int = constants.N_FLOOD_MAPS,
    include_flood_maps: bool = False,
) -> dict[str, tf.Tensor]:
    """Creates a fake training batch for testing.

    All dimensions can be specified via optional args, otherwise will use the values
    defined within constants.py.

    Args:
        batch_size: Batch size.
        height: Optional height. Defaults to MAP_HEIGHT.
        width: Optional width. Defaults to MAP_WIDTH.
        rainfall_duration: Optional rainfall duration.
        flood_maps: Optional number of flood maps. Defaults to N_FLOOD_MAPS.
            Only used if include_flood_maps=True.
        include_flood_maps: Whether or not to pass in flood maps as
            spatiotemporal inputs.

    Returns:
        A dictionary of model inputs, with the following keys:
        - "geospatial": Required. A 4D tensor [batch, height, width, features].
        - "temporal": Required. A 2D tensor [batch, rainfall duration].
        - "spatiotemporal": Optional. A 5D tensor of time series flood maps
            [batch, time, height, width, 1].
    """

    geospatial = tf.random.normal((batch_size, height, width, constants.GEO_FEATURES))
    temporal = tf.random.normal((batch_size, rainfall_duration))

    input = {"geospatial": geospatial, "temporal": temporal}

    if include_flood_maps:
        spatiotemporal = tf.random.normal((batch_size, flood_maps, height, width, 1))
        input["spatiotemporal"] = spatiotemporal

    return input


def test_forward():
    """Tests a single pass (prediction) of the model."""
    batch_size = 4
    height, width = 100, 100
    flood_maps = 5

    fake_input = fake_flood_input_batch(
        batch_size,
        height=height,
        width=width,
        flood_maps=flood_maps,
        include_flood_maps=True,
    )
    geo_input = fake_input["geospatial"]
    st_input = fake_input["spatiotemporal"]
    temp_input = data_utils.temporal_window_view(
        fake_input["temporal"], _MODEL_PARAMS["m_rainfall"]
    )
    # The forward function assumes temp_input has been clipped to the same
    # time slice as the flood maps. We take the first <flood_maps> values.
    temp_input = temp_input[:, :flood_maps, :]

    model = flood_model.FloodConvLSTM(_MODEL_PARAMS)
    prediction = model.forward(st_input, geo_input, temp_input)
    assert prediction.shape == (batch_size, height, width, 1)

"""Tests for Flood model."""

import tensorflow as tf

from usl_models.flood_ml import constants
from usl_models.flood_ml import data_utils
from usl_models.flood_ml import model as flood_model
from usl_models.flood_ml import model_params


def fake_flood_input_batch(
    batch_size: int,
    height: int = constants.MAP_HEIGHT,
    width: int = constants.MAP_WIDTH,
    rainfall_duration: int = 24,
    include_flood_map: bool = False,
) -> dict[str, tf.Tensor]:
    """Creates a fake training batch for testing.

    All dimensions can be specified via optional args, otherwise will use the values
    defined within constants.py.

    Args:
        batch_size: Batch size.
        height: Optional height. Defaults to MAP_HEIGHT.
        width: Optional width. Defaults to MAP_WIDTH.
        rainfall_duration: Optional rainfall duration.
        include_flood_map: Whether or not to pass in a single flood map as the
            spatiotemporal input.

    Returns:
        A dictionary of model inputs, with the following keys:
        - "geospatial": Required. A 4D tensor [batch, height, width, features].
        - "temporal": Required. A 2D tensor [batch, rainfall duration].
        - "spatiotemporal": Optional. A 4D tensor of time series flood maps
            [batch, height, width, 1].
    """
    geospatial = tf.random.normal((batch_size, height, width, constants.GEO_FEATURES))
    temporal = tf.random.normal((batch_size, rainfall_duration))

    input = {"geospatial": geospatial, "temporal": temporal}

    if include_flood_map:
        spatiotemporal = tf.random.normal((batch_size, height, width, 1))
        input["spatiotemporal"] = spatiotemporal

    return input


def test_convlstm_forward():
    """Tests a single pass (prediction) of the model.

    Expected input shapes:
        st_input: [B, n, H, W, 1]
        geo_input: [B, H, W, f]
        temp_input: [B, n, m]

    Expected output shape: [B, H, W, 1]
    """
    batch_size = 4
    height, width = 100, 100
    flood_maps = 5
    params = model_params.test_model_params

    st_input = tf.random.normal((batch_size, flood_maps, height, width, 1))
    geo_input = tf.random.normal((batch_size, height, width, constants.GEO_FEATURES))
    # The forward function assumes temp_input has been clipped to the same
    # time slice as the flood maps, so we only create <flood_maps> values.
    temp_input = tf.random.normal((batch_size, flood_maps))
    temp_input = data_utils.temporal_window_view(temp_input, params.m_rainfall)

    model = flood_model.FloodConvLSTM(params, spatial_dims=(height, width))
    prediction = model.forward(st_input, geo_input, temp_input)
    assert prediction.shape == (batch_size, height, width, 1)


def test_convlstm_call():
    """Tests the FloodConvLSTM model call.

    Expected input shapes:
        spatiotemporal: [B, H, W, 1)
        geospatial: [B, H, W, f]
        temporal: [B, T_max, m]

    Expected output shape: [B, T, H, W]
    """
    batch_size = 4
    height, width = 100, 100
    params = model_params.test_model_params

    # The FloodConvLSTM model expects the data to have been preprocessed, such
    # that it receives the full temporal inputs and a single flood map.
    fake_input = fake_flood_input_batch(
        batch_size,
        height=height,
        width=width,
        include_flood_map=True,
    )
    full_temp_input = data_utils.temporal_window_view(
        fake_input["temporal"], params.m_rainfall
    )
    fake_input["temporal"] = full_temp_input

    storm_duration = 12
    model = flood_model.FloodConvLSTM(
        params, n_predictions=storm_duration, spatial_dims=(height, width)
    )
    prediction = model(fake_input)
    assert prediction.shape == (batch_size, storm_duration, height, width)


def test_train():
    """Tests the model training.

    Expected training inputs:
        spatiotemporal: Optional[tf.Tensor[shape=(B, H, W, 1)]]
        geospatial: tf.Tensor[shape=(B, H, W, f)]
        temporal: tf.Tensor[shape=(B, T_max)]

    Expected labels and outputs: tf.Tensor[shape=(B, T, H, W)]
    """
    batch_size = 16
    height, width = 100, 100
    rainfall_duration = 12
    params = model_params.test_model_params

    model = flood_model.FloodModel(params, spatial_dims=(height, width))

    storm_1 = 4
    fake_input_1 = fake_flood_input_batch(
        batch_size,
        height=height,
        width=width,
        rainfall_duration=rainfall_duration,
    )
    fake_labels_1 = tf.random.normal((batch_size, storm_1, height, width))
    history_1 = model.train(fake_input_1, fake_labels_1, storm_1)
    assert "loss" in history_1.history

    # Also ensure model can continue training with varying storm durations.
    # Note that this storm is shorter than the total rainfall data.
    storm_2 = 8
    fake_input_2 = fake_flood_input_batch(
        batch_size // 2,
        height=height,
        width=width,
        rainfall_duration=rainfall_duration,
    )
    fake_labels_2 = tf.random.normal((batch_size // 2, storm_2, height, width))
    history_2 = model.train(fake_input_2, fake_labels_2, storm_2)
    assert "loss" in history_2.history

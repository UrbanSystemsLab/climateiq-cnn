"""Tests for Flood model."""

import dataclasses
import pytest

import tensorflow as tf

from usl_models.flood_ml import data_utils
from usl_models.flood_ml import model as flood_model
from usl_models.flood_ml import model_params

# Use smaller feature spaces for testing, rather than the values provided in
# constants.py.
_TEST_GEO_FEATURES = 8
_TEST_MAP_HEIGHT = 100
_TEST_MAP_WIDTH = 100
_TEST_MAX_RAINFALL = 36


def fake_flood_input_batch(
    batch_size: int,
    height: int = _TEST_MAP_HEIGHT,
    width: int = _TEST_MAP_WIDTH,
    storm_duration: int = 2,
    include_flood_map: bool = False,
    include_labels: bool = False,
) -> dict[str, tf.Tensor]:
    """Creates a fake training batch for testing.

    All dimensions can be specified via optional args, otherwise will use the values
    defined within constants.py.

    Args:
        batch_size: Batch size.
        height: Optional height.
        width: Optional width.
        storm_duration: Storm duration.
        include_flood_map: Whether or not to pass in a single flood map as the
            spatiotemporal input.
        include_labels: Whether or not to include labels associated with the
            inputs.

    Returns:
        A dictionary of model inputs, with the following keys:
        - "geospatial": Required. A 4D tensor [batch, height, width, features].
        - "temporal": Required. A 2D tensor [batch, MAX_RAINFALL].
        - "spatiotemporal": Optional. A 4D tensor of time series flood maps
            [batch, height, width, 1].
    """
    geospatial = tf.random.normal((batch_size, height, width, _TEST_GEO_FEATURES))
    temporal = tf.random.normal((batch_size, _TEST_MAX_RAINFALL))

    input = {"geospatial": geospatial, "temporal": temporal}

    if include_flood_map:
        spatiotemporal = tf.random.normal((batch_size, height, width, 1))
        input["spatiotemporal"] = spatiotemporal

    if include_labels:
        input["labels"] = tf.random.normal((batch_size, storm_duration, height, width))

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
    geo_input = tf.random.normal((batch_size, height, width, _TEST_GEO_FEATURES))
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
    storm_duration = 12
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
    params = model_params.test_model_params

    model = flood_model.FloodModel(params, spatial_dims=(height, width))

    storm_1 = 4
    fake_input_1 = fake_flood_input_batch(
        batch_size,
        height=height,
        width=width,
        storm_duration=storm_1,
        include_labels=True,
    )
    fake_data_1 = flood_model.FloodModelData(storm_duration=storm_1, **fake_input_1)

    storm_2 = 8
    fake_input_2 = fake_flood_input_batch(
        batch_size // 2,
        height=height,
        width=width,
        storm_duration=storm_2,
        include_labels=True,
    )
    fake_data_2 = flood_model.FloodModelData(storm_duration=storm_2, **fake_input_2)

    history = model.train([fake_data_1, fake_data_2])
    assert len(history) == 2


def test_bad_labels():
    """Tests validation of bad labels."""
    batch_size = 8
    height, width = 100, 100
    params = model_params.test_model_params

    model = flood_model.FloodModel(params, spatial_dims=(height, width))

    storm_duration = 4
    fake_input = fake_flood_input_batch(
        batch_size,
        height=height,
        width=width,
        storm_duration=8,  # labels do not match storm_duration
        include_labels=True,
    )
    fake_data = flood_model.FloodModelData(storm_duration=storm_duration, **fake_input)

    with pytest.raises(AssertionError):
        model.train([fake_data])


def test_early_stopping():
    """Tests early stopping during model training.

    Since inputs are random, the model will not continually improve (there's no real
    training). Setting a reasonably large number of epochs makes it likely that the
    validation loss stops improving before training is complete. Enabling early stopping
    with a patience of 1 epoch means that the model should stop training as soon as the
    validaton loss goes up.
    """
    # To ensure determinism, we set a random seed so the model is reproducible.
    tf.keras.utils.set_random_seed(1)

    batch_size = 16
    height, width = 100, 100
    # Set a large number of epochs to increase the odds of triggering early stopping.
    params = dataclasses.replace(model_params.test_model_params, epochs=20)

    model = flood_model.FloodModel(params, spatial_dims=(height, width))

    storm_duration = 4
    fake_input = fake_flood_input_batch(
        batch_size,
        height=height,
        width=width,
        storm_duration=storm_duration,
        include_labels=True,
    )
    fake_data = flood_model.FloodModelData(storm_duration=storm_duration, **fake_input)

    history = model.train([fake_data], early_stopping=1)
    # Check whether the model history indicates early stopping.
    assert len(history[0].history["val_loss"]) < params.epochs

"""Tests for Flood model."""

import tempfile

import numpy as np
import tensorflow as tf

from usl_models.flood_ml import model as flood_model
from usl_models.flood_ml import model_params
from tests.mock_dataset import mock_dataset


def test_convlstm_call():
    """Tests a single pass (prediction) of the model.

    Expected input shapes:
        st_input: [B, n, H, W, 1]
        geo_input: [B, H, W, f]
        temp_input: [B, n, m]

    Expected output shape: [B, H, W, 1]
    """
    batch_size = 4
    height, width = 100, 100
    params = model_params.test_model_params()

    input, _ = next(
        iter(mock_dataset(params, batch_size=batch_size, height=height, width=width))
    )
    model = flood_model.FloodConvLSTM(params, spatial_dims=(height, width))
    prediction = model.call(input)
    assert prediction.shape == (batch_size, height, width, 1)


def test_convlstm_call_n():
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
    params = model_params.test_model_params()

    # The FloodConvLSTM model expects the data to have been preprocessed, such
    # that it receives the full temporal inputs and a single flood map.
    input, _ = next(
        iter(
            mock_dataset(
                params,
                height=height,
                width=width,
                batch_size=batch_size,
                n=storm_duration,
            )
        )
    )
    model = flood_model.FloodConvLSTM(params, spatial_dims=(height, width))
    prediction = model.call_n(input, n=storm_duration)
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
    params = model_params.test_model_params()

    model = flood_model.FloodModel(params, spatial_dims=(height, width))
    epochs = 2
    dataset = mock_dataset(
        params, height=height, width=width, batch_size=batch_size, batch_count=epochs
    )
    history = model.fit(dataset, epochs=epochs, steps_per_epoch=1)
    assert len(history.history["loss"]) == epochs


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

    batch_size = 4
    height, width = 100, 100
    # Set a large number of epochs to increase the odds of triggering early stopping.
    params = model_params.test_model_params()
    epochs = 20
    model = flood_model.FloodModel(params, spatial_dims=(height, width))

    dataset = mock_dataset(
        params,
        height=height,
        width=width,
        batch_size=batch_size,
        batch_count=epochs,
    )
    history = model.fit(dataset, early_stopping=1, epochs=epochs, steps_per_epoch=1)
    # Check whether the model history indicates early stopping.
    assert len(history.history["loss"]) < epochs


def test_model_checkpoint():
    """Tests saving and loading a model checkpoint."""
    batch_size = 16
    height, width = 100, 100
    params = model_params.test_model_params()

    model = flood_model.FloodModel(params, spatial_dims=(height, width))

    dataset = mock_dataset(
        params,
        height=height,
        width=width,
        batch_size=batch_size,
        batch_count=1,
    )
    model.fit(dataset, steps_per_epoch=1)

    with tempfile.NamedTemporaryFile(suffix=".keras") as tmp:
        model.save_model(tmp.name, overwrite=True)

        new_model = flood_model.FloodModel(params, spatial_dims=(height, width))
        new_model.load_model(tmp.name)

    old_weights = model._model.get_weights()
    new_weights = new_model._model.get_weights()
    for old, new in zip(old_weights, new_weights):
        np.testing.assert_array_equal(old, new)


def test_get_temporal_window():
    """Tests copmuting the temporal window."""
    B, T_MAX, M = 1, 16, 6
    temporal = tf.random.normal((B, T_MAX, M))

    t = 1  # Timestep
    n = 5  # Window size
    actual = flood_model.FloodConvLSTM._get_temporal_window(temporal, t=t, n=n)
    assert actual.shape == (B, n, M)
    expected = tf.concat([np.zeros((B, n - t, M)), temporal[:, 0:t]], axis=1)
    np.testing.assert_array_equal(actual, expected)

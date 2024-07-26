"""Tests for Flood model."""

import tempfile

import numpy as np
import tensorflow as tf

from usl_models.flood_ml import model as flood_model
from usl_models.flood_ml import model_params
from tests.flood_ml.mock_dataset import mock_dataset, mock_prediction_dataset


def pytest_model_params() -> model_params.FloodModelParams:
    """Defines FloodModelParams for testing."""
    params = model_params.default_params()
    params.update(
        {
            "batch_size": 4,
            "m_rainfall": 3,
            "n_flood_maps": 3,
            "lstm_units": 32,
            "lstm_kernel_size": 3,
        }
    )
    return params


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
    params = pytest_model_params()

    input, _ = next(
        iter(mock_dataset(params, batch_size=batch_size, height=height, width=width))
    )
    model = flood_model.FloodConvLSTM(params, spatial_dims=(height, width))
    prediction = model.call(input)
    assert prediction.shape == (batch_size, height, width, 1)


def test_convlstm_call_n():
    """Tests the FloodConvLSTM model call.

    Expected input shapes:
        spatiotemporal: [B, n, H, W, 1)
        geospatial: [B, H, W, f]
        temporal: [B, T_max, m]

    Expected output shape: [B, T, H, W]
    """
    batch_size = 4
    height, width = 100, 100
    storm_duration = 12
    params = pytest_model_params()

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


def test_batch_predict_n():
    """Tests the FloodConvLSTM model batch predict.

    Expected input shapes:
        spatiotemporal: [B, n, H, W, 1)
        geospatial: [B, H, W, f]
        temporal: [B, T_max, m]

    Expected output shape: [B, T, H, W]
    """
    batch_size = 4
    height, width = 100, 100
    storm_duration = 3
    params = pytest_model_params()

    dataset = mock_prediction_dataset(
        params,
        height=height,
        width=width,
        batch_size=batch_size,
        n=storm_duration,
    )
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = flood_model.FloodModel(params, spatial_dims=(height, width))
        for results in model.batch_predict_n(dataset, n=storm_duration):
            assert len(results) == batch_size
            for result in results:
                assert result["prediction"].shape == (height, width)


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
    params = pytest_model_params()

    model = flood_model.FloodModel(params, spatial_dims=(height, width))
    epochs = 2
    train_dataset = mock_dataset(
        params, height=height, width=width, batch_size=batch_size, batch_count=epochs
    )
    val_dataset = mock_dataset(
        params, height=height, width=width, batch_size=batch_size, batch_count=epochs
    )
    history = model.fit(
        train_dataset, val_dataset=val_dataset, epochs=epochs, steps_per_epoch=1
    )
    assert len(history.history["loss"]) == epochs
    # Also check that the model is calculating validation metrics
    assert "val_loss" in history.history


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
    params = pytest_model_params()
    epochs = 20
    model = flood_model.FloodModel(params, spatial_dims=(height, width))

    train_dataset = mock_dataset(
        params,
        height=height,
        width=width,
        batch_size=batch_size,
        batch_count=epochs,
    )
    val_dataset = mock_dataset(
        params,
        height=height,
        width=width,
        batch_size=batch_size,
        batch_count=epochs,
    )
    history = model.fit(
        train_dataset,
        val_dataset=val_dataset,
        early_stopping=1,
        epochs=epochs,
        steps_per_epoch=1,
    )
    # Check whether the model history indicates early stopping.
    assert len(history.history["loss"]) < epochs


def test_model_checkpoint():
    """Tests saving and loading a model checkpoint."""
    batch_size = 16
    height, width = 100, 100
    params = pytest_model_params()

    model = flood_model.FloodModel(params, spatial_dims=(height, width))

    train_dataset = mock_dataset(
        params,
        height=height,
        width=width,
        batch_size=batch_size,
        batch_count=1,
    )
    val_dataset = mock_dataset(
        params,
        height=height,
        width=width,
        batch_size=batch_size,
        batch_count=1,
    )
    model.fit(train_dataset, val_dataset=val_dataset, steps_per_epoch=1)

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
    B, T_MAX, M = 1, 8, 6
    temporal = tf.constant([[[t + 1] * M for t in range(T_MAX)]], dtype=tf.float32)

    t = 3  # Timestep
    n = 5  # Window size
    actual = flood_model.FloodConvLSTM._get_temporal_window(temporal, t=t, n=n)
    assert actual.shape == (B, n, M)
    expected = [
        [
            np.zeros((M)),
            np.zeros((M)),
            [1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3],
        ],
    ]
    np.testing.assert_array_equal(actual, expected)

"""Tests for Flood model."""

import tempfile
import unittest

import keras
import numpy as np
import tensorflow as tf

from usl_models.flood_ml import model as flood_model
from tests.flood_ml.mock_dataset import mock_dataset, mock_prediction_dataset
from usl_models.flood_ml.model import SpatialAttention, FloodConvLSTM


class FloodModelTest(unittest.TestCase):
    """Basic tests for Flood Model."""

    def setUp(self):
        """Set up test fixture."""
        keras.utils.set_random_seed(1)
        self._params = flood_model.FloodModel.Params(
            lstm_units=32,
            lstm_kernel_size=3,
            lstm_dropout=0.2,
            lstm_recurrent_dropout=0.2,
            m_rainfall=3,
            n_flood_maps=3,
            num_features=22,
        )

    def test_convlstm_call(self):
        """Tests a single pass (prediction) of the model.

        Expected input shapes:
            st_input: [B, n, H, W, 1]
            geo_input: [B, H, W, f]
            temp_input: [B, n, m]

        Expected output shape: [B, H, W, 1]
        """
        batch_size = 4
        height, width = 100, 100

        input, _ = next(
            iter(
                mock_dataset(
                    self._params, batch_size=batch_size, height=height, width=width
                )
            )
        )
        model = flood_model.FloodConvLSTM(self._params, spatial_dims=(height, width))
        prediction = model.call(input)
        assert prediction.shape == (batch_size, height, width, 1)

    def test_convlstm_call_n(self):
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

        # The FloodConvLSTM model expects the data to have been preprocessed, such
        # that it receives the full temporal inputs and a single flood map.
        input, _ = next(
            iter(
                mock_dataset(
                    self._params,
                    height=height,
                    width=width,
                    batch_size=batch_size,
                    n=storm_duration,
                )
            )
        )
        model = flood_model.FloodConvLSTM(self._params, spatial_dims=(height, width))
        prediction = model.call_n(input, n=storm_duration)
        assert prediction.shape == (batch_size, storm_duration, height, width)

    def test_batch_predict_n(self):
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

        dataset = mock_prediction_dataset(
            self._params,
            height=height,
            width=width,
            batch_size=batch_size,
            n=storm_duration,
        )
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = flood_model.FloodModel(self._params, spatial_dims=(height, width))
            for results in model.batch_predict_n(dataset, n=storm_duration):
                assert len(results) == batch_size
                for result in results:
                    assert result["prediction"].shape == (height, width)

    def test_train(self):
        """Tests the model training.

        Expected training inputs:
            spatiotemporal: tf.Tensor[shape=(B, H, W, 1)] | None
            geospatial: tf.Tensor[shape=(B, H, W, f)]
            temporal: tf.Tensor[shape=(B, T_max)]

        Expected labels and outputs: tf.Tensor[shape=(B, T, H, W)]
        """
        batch_size = 16
        height, width = 100, 100

        model = flood_model.FloodModel(self._params, spatial_dims=(height, width))
        epochs = 2
        train_dataset = mock_dataset(
            self._params,
            height=height,
            width=width,
            batch_size=batch_size,
            batch_count=epochs,
        )
        val_dataset = mock_dataset(
            self._params,
            height=height,
            width=width,
            batch_size=batch_size,
            batch_count=epochs,
        )
        history = model.fit(
            train_dataset, val_dataset=val_dataset, epochs=epochs, steps_per_epoch=1
        )
        assert len(history.history["loss"]) == epochs
        # Also check that the model is calculating validation metrics
        assert "val_loss" in history.history

    def test_early_stopping(self):
        """Tests early stopping during model training.

        Since inputs are random, the model will not continually improve (there's no real
        training). Setting a reasonably large number of epochs makes it likely that the
        validation loss stops improving before training is complete. Enabling early
        stopping with a patience of 1 epoch means that the model should stop training
        as soon as the validaton loss goes up.
        """
        batch_size = 4
        height, width = 100, 100
        # Set a large number of epochs to increase the odds of triggering early
        # stopping.
        epochs = 20
        model = flood_model.FloodModel(self._params, spatial_dims=(height, width))

        train_dataset = mock_dataset(
            self._params,
            height=height,
            width=width,
            batch_size=batch_size,
            batch_count=epochs,
        )
        val_dataset = mock_dataset(
            self._params,
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

    def test_model_checkpoint(self):
        """Tests saving and loading a model checkpoint."""
        batch_size = 16
        height, width = 100, 100

        model = flood_model.FloodModel(self._params, spatial_dims=(height, width))

        train_dataset = mock_dataset(
            self._params,
            height=height,
            width=width,
            batch_size=batch_size,
            batch_count=1,
        )
        val_dataset = mock_dataset(
            self._params,
            height=height,
            width=width,
            batch_size=batch_size,
            batch_count=1,
        )
        model.fit(train_dataset, val_dataset=val_dataset, steps_per_epoch=1)

        with tempfile.NamedTemporaryFile(suffix=".keras") as tmp:
            model.save_model(tmp.name, overwrite=True)

            # Load the entire model, not just weights
            loaded_model = keras.models.load_model(
                tmp.name,
                custom_objects={
                    "SpatialAttention": SpatialAttention,
                    "FloodConvLSTM": FloodConvLSTM,
                },
            )

            # Check weights equality
            old_weights = model._model.get_weights()
            new_weights = loaded_model.get_weights()
        for old, new in zip(old_weights, new_weights):
            np.testing.assert_array_equal(old, new)

    def test_get_temporal_window(self):
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

"""Tests for Atmo model."""

import tempfile
import unittest

import keras
import tensorflow as tf
import numpy as np

from usl_models.atmo_ml import model as atmo_model
from usl_models.atmo_ml import constants

_TEST_MAP_HEIGHT = 50
_TEST_MAP_WIDTH = 50
_TEST_SPATIAL_FEATURES = 22  # lu_index is now separate
_TEST_SPATIOTEMPORAL_FEATURES = 12
_LU_INDEX_VOCAB_SIZE = 61


class AtmoModelTest(unittest.TestCase):
    """Basic tests for AtmoModel."""

    def setUp(self):
        """Set up test fixture."""
        keras.utils.set_random_seed(1)
        self._params = atmo_model.AtmoModel.Params(
            output_timesteps=2,
            lstm_units=32,
            lstm_kernel_size=3,
            optimizer=keras.optimizers.Adam(
                learning_rate=1e-3,
            ),
        )

    def fake_input_batch(
        self,
        batch_size: int,
        height: int = _TEST_MAP_HEIGHT,
        width: int = _TEST_MAP_WIDTH,
    ) -> atmo_model.AtmoModel.Input:
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
        lu_index = tf.random.uniform(
            (batch_size, height, width),
            minval=0,
            maxval=_LU_INDEX_VOCAB_SIZE,
            dtype=tf.int32,
        )
        return atmo_model.AtmoModel.Input(
            spatial=spatial,
            spatiotemporal=spatiotemporal,
            lu_index=lu_index,
            sim_name=tf.constant(["test"] * batch_size),
            date=tf.constant(["test"] * batch_size),
        )

    def test_atmo_convlstm(self):
        """Tests the AtmoConvLSTM model call."""
        batch_size = 4

        fake_input = self.fake_input_batch(batch_size)

        model = atmo_model.AtmoConvLSTM(self._params)
        prediction = model(fake_input)

        expected_output_shape = (
            batch_size,
            self._params.output_timesteps,
            _TEST_MAP_HEIGHT,
            _TEST_MAP_WIDTH,
            len(self._params.sto_vars),  # T2, RH2, WSPD10
        )

        assert prediction.shape == expected_output_shape

    def test_train(self):
        """Tests the AtmoModel training.

        Expected training inputs:
            spatial: tf.Tensor[shape=(B, H, W, num_spatial_features)]
            spatiotemporal: tf.Tensor[shape=(B, T, H, W, num_spatiotemporal_features)]

        Expected labels and outputs: tf.Tensor[shape=(B, T, H, W, output_channels)]
        """
        batch_size = 8
        epochs = 2

        model = atmo_model.AtmoModel(self._params)

        # Create fake training and validation datasets
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                self.fake_input_batch(batch_size),
                tf.random.normal(
                    [
                        batch_size,
                        constants.OUTPUT_TIME_STEPS,
                        _TEST_MAP_HEIGHT,
                        _TEST_MAP_WIDTH,
                        len(self._params.sto_vars),
                    ]
                ),
            )
        ).batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices(
            (
                self.fake_input_batch(batch_size),
                tf.random.normal(
                    [
                        batch_size,
                        constants.OUTPUT_TIME_STEPS,
                        _TEST_MAP_HEIGHT,
                        _TEST_MAP_WIDTH,
                        len(self._params.sto_vars),
                    ]
                ),
            )
        ).batch(batch_size)

        history = model.fit(
            train_dataset,
            val_dataset=val_dataset,
            epochs=epochs,
            steps_per_epoch=1,
        )
        assert len(history.history["loss"]) == epochs
        assert "val_loss" in history.history

    def test_early_stopping(self):
        """Tests early stopping during model training."""
        batch_size = 4
        epochs = 20

        model = atmo_model.AtmoModel(self._params)

        # Create fake training and validation datasets
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                self.fake_input_batch(batch_size),
                tf.random.normal(
                    [
                        batch_size,
                        constants.OUTPUT_TIME_STEPS,
                        _TEST_MAP_HEIGHT,
                        _TEST_MAP_WIDTH,
                        len(self._params.sto_vars),
                    ]
                ),
            )
        ).batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices(
            (
                self.fake_input_batch(batch_size),
                tf.random.normal(
                    [
                        batch_size,
                        constants.OUTPUT_TIME_STEPS,
                        _TEST_MAP_HEIGHT,
                        _TEST_MAP_WIDTH,
                        len(self._params.sto_vars),
                    ]
                ),
            )
        ).batch(batch_size)

        history = model.fit(
            train_dataset,
            val_dataset=val_dataset,
            early_stopping=1,
            epochs=epochs,
            steps_per_epoch=1,
        )
        assert len(history.history["loss"]) <= epochs

    def test_model_checkpoint(self):
        """Tests saving and loading a model checkpoint."""
        batch_size = 16
        model = atmo_model.AtmoModel(self._params)

        # Create fake training and validation datasets
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (
                self.fake_input_batch(batch_size),
                tf.random.normal(
                    [
                        batch_size,
                        constants.OUTPUT_TIME_STEPS,
                        _TEST_MAP_HEIGHT,
                        _TEST_MAP_WIDTH,
                        len(self._params.sto_vars),
                    ]
                ),
            )
        ).batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices(
            (
                self.fake_input_batch(batch_size),
                tf.random.normal(
                    [
                        batch_size,
                        constants.OUTPUT_TIME_STEPS,
                        _TEST_MAP_HEIGHT,
                        _TEST_MAP_WIDTH,
                        len(self._params.sto_vars),
                    ]
                ),
            )
        ).batch(batch_size)

        model.fit(train_dataset, val_dataset=val_dataset, steps_per_epoch=1)
        with tempfile.TemporaryDirectory(suffix="model") as tmp:
            model.save_model(tmp)
            loaded_model = atmo_model.AtmoModel.from_checkpoint(tmp)

            for weights, loaded_weights in zip(
                model._model.get_weights(), loaded_model._model.get_weights()
            ):
                np.testing.assert_array_equal(weights, loaded_weights)

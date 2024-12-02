"""Tests for Atmo model."""

import tensorflow as tf
import tempfile
import numpy as np
from usl_models.atmo_ml import model as atmo_model
from usl_models.atmo_ml import constants
from usl_models.atmo_ml import model_params

_TEST_MAP_HEIGHT = 200
_TEST_MAP_WIDTH = 200
_TEST_SPATIAL_FEATURES = 22  # lu_index is now separate
_TEST_SPATIOTEMPORAL_FEATURES = 12
_LU_INDEX_VOCAB_SIZE = 61
_EMBEDDING_DIM = 8


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
    num_spatial_features: int = _TEST_SPATIAL_FEATURES,
    num_spatiotemporal_features: int = _TEST_SPATIOTEMPORAL_FEATURES,
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
    return {
        "spatial": spatial,
        "spatiotemporal": spatiotemporal,
        "lu_index": lu_index,
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
        lu_index_vocab_size=_LU_INDEX_VOCAB_SIZE,  # Added for lu_index
        embedding_dim=_EMBEDDING_DIM,  # Added for lu_index embedding
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


def test_handle_missing_spatial_features():
    """Tests model handling when some spatial features are missing."""
    batch_size = 4
    params = pytest_model_params()

    # Create input with missing spatial features
    missing_spatial_features = _TEST_SPATIAL_FEATURES - 5
    fake_input = fake_input_batch(
        batch_size, num_spatial_features=missing_spatial_features
    )

    model = atmo_model.AtmoConvLSTM(
        params,
        spatial_dims=(_TEST_MAP_HEIGHT, _TEST_MAP_WIDTH),
        num_spatial_features=_TEST_SPATIAL_FEATURES,
        num_spatiotemporal_features=_TEST_SPATIOTEMPORAL_FEATURES,
        lu_index_vocab_size=_LU_INDEX_VOCAB_SIZE,
        embedding_dim=_EMBEDDING_DIM,
    )
    prediction = model(fake_input)

    expected_output_shape = (
        batch_size,
        constants.OUTPUT_TIME_STEPS,
        _TEST_MAP_HEIGHT,
        _TEST_MAP_WIDTH,
        constants.OUTPUT_CHANNELS,
    )

    assert prediction.shape == expected_output_shape


def test_handle_missing_spatiotemporal_features():
    """Tests model handling when some spatiotemporal features are missing."""
    batch_size = 4
    params = pytest_model_params()

    # Create input with missing spatiotemporal features
    missing_spatiotemporal_features = _TEST_SPATIOTEMPORAL_FEATURES - 3
    fake_input = fake_input_batch(
        batch_size, num_spatiotemporal_features=missing_spatiotemporal_features
    )

    model = atmo_model.AtmoConvLSTM(
        params,
        spatial_dims=(_TEST_MAP_HEIGHT, _TEST_MAP_WIDTH),
        num_spatial_features=_TEST_SPATIAL_FEATURES,
        num_spatiotemporal_features=_TEST_SPATIOTEMPORAL_FEATURES,
        lu_index_vocab_size=_LU_INDEX_VOCAB_SIZE,
        embedding_dim=_EMBEDDING_DIM,
    )
    prediction = model(fake_input)

    expected_output_shape = (
        batch_size,
        constants.OUTPUT_TIME_STEPS,
        _TEST_MAP_HEIGHT,
        _TEST_MAP_WIDTH,
        constants.OUTPUT_CHANNELS,
    )

    assert prediction.shape == expected_output_shape


def test_train():
    """Tests the AtmoModel training.

    Expected training inputs:
        spatial: tf.Tensor[shape=(B, H, W, num_spatial_features)]
        spatiotemporal: tf.Tensor[shape=(B, T, H, W, num_spatiotemporal_features)]

    Expected labels and outputs: tf.Tensor[shape=(B, T, H, W, output_channels)]
    """
    batch_size = 16
    epochs = 2
    params = pytest_model_params()

    model = atmo_model.AtmoModel(
        params,
        spatial_dims=(_TEST_MAP_HEIGHT, _TEST_MAP_WIDTH),
        lu_index_vocab_size=_LU_INDEX_VOCAB_SIZE,
        embedding_dim=_EMBEDDING_DIM,
    )

    # Create fake training and validation datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (
            fake_input_batch(batch_size),
            tf.random.normal(
                [
                    batch_size,
                    constants.OUTPUT_TIME_STEPS,
                    _TEST_MAP_HEIGHT,
                    _TEST_MAP_WIDTH,
                    constants.OUTPUT_CHANNELS,
                ]
            ),
        )
    ).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (
            fake_input_batch(batch_size),
            tf.random.normal(
                [
                    batch_size,
                    constants.OUTPUT_TIME_STEPS,
                    _TEST_MAP_HEIGHT,
                    _TEST_MAP_WIDTH,
                    constants.OUTPUT_CHANNELS,
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


def test_early_stopping():
    """Tests early stopping during model training."""
    tf.keras.utils.set_random_seed(1)

    batch_size = 4
    params = pytest_model_params()
    epochs = 20

    model = atmo_model.AtmoModel(
        params,
        spatial_dims=(_TEST_MAP_HEIGHT, _TEST_MAP_WIDTH),
        lu_index_vocab_size=_LU_INDEX_VOCAB_SIZE,
        embedding_dim=_EMBEDDING_DIM,
    )

    # Create fake training and validation datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (
            fake_input_batch(batch_size),
            tf.random.normal(
                [
                    batch_size,
                    constants.OUTPUT_TIME_STEPS,
                    _TEST_MAP_HEIGHT,
                    _TEST_MAP_WIDTH,
                    constants.OUTPUT_CHANNELS,
                ]
            ),
        )
    ).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (
            fake_input_batch(batch_size),
            tf.random.normal(
                [
                    batch_size,
                    constants.OUTPUT_TIME_STEPS,
                    _TEST_MAP_HEIGHT,
                    _TEST_MAP_WIDTH,
                    constants.OUTPUT_CHANNELS,
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
    assert len(history.history["loss"]) < epochs


def test_model_checkpoint():
    """Tests saving and loading a model checkpoint."""
    batch_size = 16
    params = pytest_model_params()

    model = atmo_model.AtmoModel(
        params,
        spatial_dims=(_TEST_MAP_HEIGHT, _TEST_MAP_WIDTH),
        num_spatial_features=_TEST_SPATIAL_FEATURES,
        num_spatiotemporal_features=_TEST_SPATIOTEMPORAL_FEATURES,
        lu_index_vocab_size=_LU_INDEX_VOCAB_SIZE,
        embedding_dim=_EMBEDDING_DIM,
    )

    # Create fake training and validation datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (
            fake_input_batch(batch_size),
            tf.random.normal(
                [
                    batch_size,
                    constants.OUTPUT_TIME_STEPS,
                    _TEST_MAP_HEIGHT,
                    _TEST_MAP_WIDTH,
                    constants.OUTPUT_CHANNELS,
                ]
            ),
        )
    ).batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (
            fake_input_batch(batch_size),
            tf.random.normal(
                [
                    batch_size,
                    constants.OUTPUT_TIME_STEPS,
                    _TEST_MAP_HEIGHT,
                    _TEST_MAP_WIDTH,
                    constants.OUTPUT_CHANNELS,
                ]
            ),
        )
    ).batch(batch_size)

    model.fit(train_dataset, val_dataset=val_dataset, steps_per_epoch=1)

    with tempfile.NamedTemporaryFile(suffix=".keras") as tmp:
        model._model.save(tmp.name, overwrite=True)
        new_model = atmo_model.AtmoModel(
            params,
            spatial_dims=(_TEST_MAP_HEIGHT, _TEST_MAP_WIDTH),
            num_spatial_features=_TEST_SPATIAL_FEATURES,
            num_spatiotemporal_features=_TEST_SPATIOTEMPORAL_FEATURES,
            lu_index_vocab_size=_LU_INDEX_VOCAB_SIZE,
            embedding_dim=_EMBEDDING_DIM,
        )
        new_model._model = tf.keras.models.load_model(tmp.name, compile=False)

        # Recompile the model with the optimizer and loss
        new_model._model.compile(
            optimizer=tf.keras.optimizers.get(params["optimizer_config"]),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(),
                tf.keras.metrics.RootMeanSquaredError(),
            ],
        )

    old_weights = model._model.get_weights()
    new_weights = new_model._model.get_weights()
    for old, new in zip(old_weights, new_weights):
        np.testing.assert_array_equal(old, new)

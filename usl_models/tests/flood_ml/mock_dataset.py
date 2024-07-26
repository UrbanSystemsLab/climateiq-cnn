"""Construct mock datasets."""

import tensorflow as tf

from usl_models.flood_ml import model as flood_model
from usl_models.flood_ml import model_params
from usl_models.flood_ml import constants


def input_signature(
    params: model_params.FloodModelParams,
    height: int = 100,
    width: int = 100,
    n: int | None = None,
) -> dict[str, tf.TensorSpec]:
    """Returns the input signature for a dataset with n timesteps."""
    # If the datset specifies a number of timesteps, temporal tensor is of max size.
    temporal_duration = (
        constants.MAX_RAINFALL_DURATION if n is not None else params["n_flood_maps"]
    )
    return dict(
        geospatial=tf.TensorSpec(
            shape=(height, width, constants.GEO_FEATURES), dtype=tf.float32
        ),
        temporal=tf.TensorSpec(
            shape=(temporal_duration, params["m_rainfall"]), dtype=tf.float32
        ),
        spatiotemporal=tf.TensorSpec(
            shape=(params["n_flood_maps"], height, width, 1), dtype=tf.float32
        ),
    )


def label_signature(
    height: int = 100, width: int = 100, n: int | None = None
) -> tf.TensorSpec:
    """Returns the label signature for a dataset with n timesteps."""
    shape = (n, height, width) if n is not None else (height, width)
    return tf.TensorSpec(shape=shape, dtype=tf.float32)


def metadata_signature() -> tf.TensorSpec:
    """Returns the metadata signature describing a specific simulation chunk."""
    return dict(
        feature_chunk=tf.TensorSpec(shape=(), dtype=tf.string),
        rainfall=tf.TensorSpec(shape=(), dtype=tf.int32),
    )


def mock_dataset(
    params: model_params.FloodModelParams,
    height: int = 100,
    width: int = 100,
    batch_count: int = 1,
    batch_size: int | None = 1,
    n: int | None = None,
) -> tf.data.Dataset:
    """Constructs a dataset of random mock data.

    Args:
        params: Flood model params
        height: spatial map height
        width: spatial map width
        batch_count: number of batches to produce
        batch_size: size of each batch
        n: optional number of timesteps.
            Required for producing data for use with `model.call_n`.

    Returns:
        The TF dataset.
    """
    input_sig = input_signature(params, height, width, n)
    label_sig = label_signature(height, width, n)

    def generator():
        """Generate random tensors."""
        batch_size_int = 1 if batch_size is None else batch_size
        for i in range(batch_size_int * batch_count):
            yield flood_model.FloodModel.Input(
                geospatial=tf.random.normal(shape=input_sig["geospatial"].shape),
                temporal=tf.random.normal(shape=input_sig["temporal"].shape),
                spatiotemporal=tf.random.normal(
                    shape=input_sig["spatiotemporal"].shape
                ),
            ), tf.random.normal(label_sig.shape)

    dataset = tf.data.Dataset.from_generator(
        generator=generator, output_signature=(input_sig, label_sig)
    )
    if batch_size:
        dataset = dataset.batch(batch_size)
    return dataset


def mock_prediction_dataset(
    params: model_params.FloodModelParams,
    height: int = 100,
    width: int = 100,
    batch_count: int = 1,
    batch_size: int | None = 1,
    n: int | None = None,
) -> tf.data.Dataset:
    """Constructs a dataset of random mock prediction data.

    Args:
        params: Flood model params
        height: spatial map height
        width: spatial map width
        batch_count: number of batches to produce
        batch_size: size of each batch. If None, the data is not batched (no batch dimension).
        n: optional number of timesteps.
            Required for producing data for use with `model.call_n`.

    Returns:
        The TF dataset.
    """
    input_sig = input_signature(params, height, width, n)
    metadata_sig = metadata_signature()

    def generator():
        """Generate random tensors."""
        batch_size_int = 1 if batch_size is None else batch_size
        for i in range(batch_size_int * batch_count):
            yield flood_model.FloodModel.Input(
                geospatial=tf.random.normal(shape=input_sig["geospatial"].shape),
                temporal=tf.random.normal(shape=input_sig["temporal"].shape),
                spatiotemporal=tf.random.normal(
                    shape=input_sig["spatiotemporal"].shape
                ),
            ), dict(feature_chunk=str(i), rainfall=i)

    dataset = tf.data.Dataset.from_generator(
        generator=generator, output_signature=(input_sig, metadata_sig)
    )
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    return dataset

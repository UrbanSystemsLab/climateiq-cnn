import tensorflow as tf
from usl_models.atmo_ml import dataset, constants


def test_process_dataset():
    """Test to ensure proper input-output sequence generation."""
    # Create fake inputs for the dataset (Spatial, Spatiotemporal, LU Index, Labels)
    batch_size = 4
    height = constants.MAP_HEIGHT
    width = constants.MAP_WIDTH
    spatial_features = constants.num_spatial_features
    spatiotemporal_features = constants.num_spatiotemporal_features
    time_steps = constants.INPUT_TIME_STEPS
    output_channels = constants.OUTPUT_CHANNELS

    # Simulating fake inputs (spatial, spatiotemporal, lu_index) and labels
    spatial_input = tf.random.normal((batch_size, height, width, spatial_features))
    spatiotemporal_input = tf.random.normal(
        (batch_size, time_steps, height, width, spatiotemporal_features)
    )
    lu_index_input = tf.random.uniform(
        (batch_size, height, width),
        minval=0,
        maxval=constants.lu_index_vocab_size,
        dtype=tf.int32,
    )
    labels = tf.random.normal(
        (batch_size, constants.OUTPUT_TIME_STEPS, height, width, output_channels)
    )

    # Creating inputs as dictionary similar to model input
    inputs = {
        "spatial": spatial_input,
        "spatiotemporal": spatiotemporal_input,
        "lu_index": lu_index_input,
    }

    # Create a fake dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(
        batch_size
    )

    # Process the dataset
    processed_train_dataset = dataset.process_dataset(train_dataset)

    # Fetch one batch to check its structure
    for batch in processed_train_dataset.take(1):
        processed_inputs, processed_labels = batch

        # Assert input-output shapes
        assert "spatial" in processed_inputs
        assert "spatiotemporal" in processed_inputs
        assert "lu_index" in processed_inputs

        # Expected shapes
        expected_spatial_shape = (
            batch_size,
            height,
            width,
            spatial_features + constants.embedding_dim,
        )
        assert processed_inputs["spatial"].shape == expected_spatial_shape, (
            f"Expected spatial shape {expected_spatial_shape} but got "
            f"{processed_inputs['spatial'].shape}"
        )

        expected_spatiotemporal_shape = (
            batch_size,
            time_steps,
            height,
            width,
            spatiotemporal_features,
        )
        assert (
            processed_inputs["spatiotemporal"].shape == expected_spatiotemporal_shape
        ), (
            f"Expected spatiotemporal shape {expected_spatiotemporal_shape} but got "
            f"{processed_inputs['spatiotemporal'].shape}"
        )

        expected_output_shape = (
            batch_size,
            constants.OUTPUT_TIME_STEPS,
            height,
            width,
            output_channels,
        )
        assert processed_labels.shape == expected_output_shape, (
            f"Expected output shape {expected_output_shape} but got "
            f"{processed_labels.shape}"
        )

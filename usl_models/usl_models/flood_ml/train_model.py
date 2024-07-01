import sys, os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the desired directory (where 'usl_models' is located)
usl_models_dir = os.path.join(current_dir, "../..")  # go one directory up

# Append the path to sys.path
sys.path.append(usl_models_dir)
print(sys.path)


import tensorflow as tf

from usl_models.flood_ml.model import FloodModel, FloodModelParams
from usl_models.flood_ml.trainingdataset import IncrementalTrainDataGenerator
from usl_models.flood_ml.featurelabelchunks import GenerateFeatureLabelChunks
from usl_models.flood_ml import constants

"""
THis is a method that is used to train the flood model.
This method is called from Docker container by the Vertex
hyperparameter tuning job.

Args:
    model_dir: The GCS path where the model will be saved.

    The hyperparameters:
        batch_size: The batch size.
        lstm_units: The number of units in the LSTM layer.
        lstm_kernel_size: The kernel size for the LSTM layer.
        lstm_dropout: The dropout rate for the LSTM layer.
        lstm_recurrent_dropout: The recurrent dropout rate for the LSTM layer.
        learning_rate: The learning rate.
        epochs: The number of epochs.

Returns:
    A list of Keras history objects.

"""


def verify_labels_shape(flood_model_data_list):
    """
    Verify that the labels tensor shape matches the storm duration.
    """
    for data in flood_model_data_list:
        # Get the shape of the first label
        first_label = next(iter(data.labels.take(1)))[0]  # Get the first label batch
        expected_label_shape = list(first_label.shape)
        expected_label_shape[-1] = data.storm_duration

        # Labels must match the storm duration.
        assert first_label.shape[-1] == data.storm_duration, (
            "Provided labels are inconsistent with storm duration. "
            f"Labels are expected to have shape {expected_label_shape}. "
            f"Actual shape: {first_label.shape}."
        )
        print("**** Labels are consistent with storm duration. ***")
        print("-" * 20)


def simple_training(
    model_dir="gs://usl_models_bucket/flood_model",
    batch_size=1,
    lstm_units=128,
    lstm_kernel_size=3,
    lstm_dropout=0.2,
    lstm_recurrent_dropout=0.2,
    epochs=1,
):

    # Create FloodModelParams from hyperparameters
    model_params = FloodModelParams(
        batch_size=batch_size,
        lstm_units=lstm_units,
        lstm_kernel_size=lstm_kernel_size,
        lstm_dropout=lstm_dropout,
        lstm_recurrent_dropout=lstm_recurrent_dropout,
        epochs=epochs,
    )
    model_history = []

    # Instantiate FloodModel class
    model = FloodModel(model_params)
    # model = model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate))

    sim_names = ["Manhattan-config_v1%2FRainfall_Data_2.txt"]

    generator = IncrementalTrainDataGenerator(
        batch_size=batch_size,
    )

    # # download npy label chunks locally
    # generator.download_numpy_files(sim_names, "label")

    # # download npy feature chunks locally
    # generator.download_numpy_files(sim_names, "feature")

    # # download npy temporal chunks locally
    # generator.download_numpy_files(sim_names, "temporal")

    # compare feature and label chunks

    # def set_shapes(features, labels):
    #     features["geospatial"].set_shape([None, 1000, 1000, 8])
    #     features["temporal"].set_shape([None, 864, constants.M_RAINFALL])
    #     features["spatiotemporal"].set_shape([None, 19, 1000, 1000, 1])
    #     labels.set_shape([None, 19, 1000, 1000])
    #     return features, labels

    for sim_name in sim_names:
        dataset, storm_duration = generator.get_dataset_from_tensors(sim_name)
        print("BEFORE BATCH: Dataset element spec:", dataset.element_spec)

        dataset = dataset.batch(batch_size)

        print("AFTER BATCH: Dataset element spec:", dataset.element_spec)
        print(type(dataset))
        print(storm_duration)
        # Alternatively, use the `element_spec` to understand the structure
    
    print("Dataset generation completed, will hand over to model training..")
    print("\n")
    print("#######  Training model ##########")
    model_history = model.train(dataset, storm_duration)
    print("Training complete")
    return model_history


def full_training(
    sim_names, generator,
    model_dir="gs://usl_models_bucket/flood_model",
    batch_size=1,
    lstm_units=128,
    lstm_kernel_size=3,
    lstm_dropout=0.2,
    lstm_recurrent_dropout=0.2,
    epochs=1,
):
    model_params = FloodModelParams(
        batch_size=batch_size,
        lstm_units=lstm_units,
        lstm_kernel_size=lstm_kernel_size,
        lstm_dropout=lstm_dropout,
        lstm_recurrent_dropout=lstm_recurrent_dropout,
        epochs=epochs,
    )
    model_history = []

    model = FloodModel(model_params)

    sim_names_len = len(sim_names)
    print(f"Number of simulations: {sim_names_len}")

    # Create a dataset that yields data from all simulations
    def data_generator():
        for sim_name in sim_names:
            dataset, storm_duration = generator.get_dataset_from_tensors(sim_name)
            for element in dataset:
                yield element, storm_duration

    # Create a tf.data.Dataset from the generator
    full_dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            {
                'geospatial': tf.TensorSpec(shape=(1000, 1000, 8), dtype=tf.float32),
                'temporal': tf.TensorSpec(shape=(864, constants.M_RAINFALL), dtype=tf.float32),
                'spatiotemporal': tf.TensorSpec(shape=(1000, 1000, 1), dtype=tf.float32)
            },
            tf.TensorSpec(shape=(None, 1000, 1000), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)  # for storm_duration
        )
    )

    # Apply batching, shuffling, and prefetching
    full_dataset = full_dataset.batch(batch_size)
    full_dataset = full_dataset.shuffle(buffer_size=1000)
    full_dataset = full_dataset.prefetch(tf.data.AUTOTUNE)

    print("Dataset preparation completed, starting model training...")

    print("#######  Training the model ##########")
    model_history = model.train(full_dataset)
    print("Training complete")
    
    return model_history


def check_generator_tensor_shape(generator):
    """
    Check the shape of the tensors returned by the generator.
    """
    # Check shape of each generator
    sim_names = ["Manhattan-config_v1%2FRainfall_Data_2.txt"]
    input_shape = [1000, 1000, 1]
    for sim_name in sim_names:
        geospatial_tensor = generator._generate_feature_tensors(sim_name)
        temporal_tensor = generator._generate_temporal_tensors(sim_name)
        label_tensor = generator._generate_label_tensors(sim_name)
        geotemporal_tensor = generator._generate_spatiotemporal_tensor(input_shape)

        # Get one object from each generator
        geospatial_tensor = next(geospatial_tensor)
        temporal_tensor = next(temporal_tensor)
        label_tensor = next(label_tensor)
        geotemporal_tensor = next(geotemporal_tensor)

        print("Geospatial tensor shape:", geospatial_tensor.shape)
        print("Temporal tensor shape:", temporal_tensor.shape)
        print("Label tensor shape:", label_tensor.shape)
        print("Geotemporal tensor shape:", geotemporal_tensor.shape)


def check_tensorflow_dataset(generator):
    sim_name = "Manhattan-config_v1%2FRainfall_Data_2.txt"
    dataset, storm_duration = generator.get_dataset_from_tensors(
        sim_name, batch_size=11
    )
    print(type(dataset))
    print(storm_duration)

    print(
        f"Number of elements in dataset: {tf.data.experimental.cardinality(dataset).numpy()}"
    )

    # Iterate through the entire dataset
    count = 0
    for features, labels in dataset:
        print(f"Element {count}:")
        for key, value in features.items():
            print(f"  {key} shape: {value.shape}")
        print(f"  Labels shape: {labels.shape}")
        count += 1
        if count >= 66:  # or some other large number
            break

    print(f"Total elements in dataset: {count}")


def test_feature_label_tensors(generator):
    sim_name = "Manhattan-config_v1%2FRainfall_Data_2.txt"
    generator = generator._generate_feature_label_tensors(sim_name)
    for feature_tensor, label_tensor in generator:
        print("Feature tensor shape:", feature_tensor.shape)
        print("Label tensor shape:", label_tensor.shape)


def test_temporal_tensor(generator):
    sim_name = "Manhattan-config_v1%2FRainfall_Data_2.txt"
    generator = generator._generate_temporal_tensors(sim_name)
    for temporal_tensor in generator:
        print("Temporal tensor shape:", temporal_tensor.shape)


def investigate_dataset(generator, sim_name):
    try:
        # Get the dataset without specifying a batch size
        dataset, _ = generator.get_dataset_from_tensors(sim_name)

        print(f"Successfully created dataset for {sim_name}")

        # Try to get the first element
        try:
            first_element = next(iter(dataset))
            if isinstance(first_element, tuple) and len(first_element) == 2:
                features, labels = first_element
                print("First element shapes:")
                if isinstance(features, dict):
                    for key, value in features.items():
                        print(f"  {key} shape: {value.shape}")
                        print(f"  {key} dtype: {value.dtype}")
                else:
                    print(f"  Features shape: {features.shape}")
                    print(f"  Features dtype: {features.dtype}")
                print(f"  Labels shape: {labels.shape}")
                print(f"  Labels dtype: {labels.dtype}")
            else:
                print(f"Unexpected element structure: {type(first_element)}")
        except tf.errors.OutOfRangeError:
            print("Dataset is empty or cannot access first element")
        except Exception as e:
            print(f"Error accessing first element: {str(e)}")

        # Estimate total elements (this might be slow for large datasets)
        try:
            total_elements = dataset.cardinality().numpy()
            if total_elements == tf.data.INFINITE_CARDINALITY:
                print("Dataset is infinite")
            elif total_elements == tf.data.UNKNOWN_CARDINALITY:
                print("Dataset cardinality is unknown")
            else:
                print(f"Total number of elements in dataset: {total_elements}")
        except Exception as e:
            print(f"Error estimating dataset size: {str(e)}")

        # Suggest a conservative batch size
        suggested_batch_size = 1
        print(f"Suggested starting batch size: {suggested_batch_size}")
        print(
            "Note: Increase batch size gradually in your training loop, monitoring for OOM errors"
        )

        return suggested_batch_size
    except Exception as e:
        print(f"Error investigating dataset: {str(e)}")
        return 0


def set_tf_gpu():
    print(tf.__version__)
    # Check if TensorFlow was built with CUDA
    print("TensorFlow built with CUDA support:", tf.test.is_built_with_cuda())

    # List available GPUs
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print("Num GPUs Available: ", len(gpus))
    else:
        print("No GPUs found")

    # Configure TensorFlow for GPU usage
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Use only the first GPU (T4 in your case)
            tf.config.set_visible_devices(gpus[0], "GPU")
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(
                len(gpus),
                "Physical GPUs,",
                len(logical_gpus),
                "Logicalexpand_more GPUs",
            )
        except RuntimeError as e:
            print(e)  # Handle potent


def download_sims_locally(class_label, generator, sim_names):
    print("**** Length of sim_names: ", len(sim_names))
    print("Checking if feature and label chunks are equal")
    # compare feature and label chunks
    for sim_name in sim_names:
        print(f"Comparing feature and label chunks for {sim_name}")
        label_compare = class_label.compare_study_area_sim_chunks(sim_name)
        if label_compare:
            print("Feature and label chunks are equal for", sim_name)
            print("")
        else:
            print("Feature and label chunks are not equal")
            sim_names.remove(sim_name)

    print("**** Length of *valid* sim_names: ", len(sim_names))
    print("Printing valid sims in metastore")
    for sim_name in sim_names:
        print(f"Index: {sim_names.index(sim_name)} , Sim name: {sim_name}")
        print("")

    # download npy temporal chunks locally
    print("Downloading numpy temporal chunks locally")
    generator.download_numpy_files(sim_names, "temporal")
    print("Temporals downloaded successfully")

    # download npy label chunks locally
    print("Downloading numpy label chunks locally")
    generator.download_numpy_files(sim_names, "label")
    print("Labels downloaded successfully")

    # download npy feature chunks locally
    print("Downloading numpy feature chunks locally")
    generator.download_numpy_files(sim_names, "feature")
    print("Features downloaded successfully")


def main():
    batch_size = 1
    sim_name = "Manhattan-config_v1%2FRainfall_Data_2.txt"

    generator = IncrementalTrainDataGenerator(
        batch_size=batch_size,
    )

    # set_tf_gpu()

    # check_generator_tensor_shape(generator)

    # test_temporal_tensor(generator)

    # test_feature_label_tensors(generator)

    # check_tensorflow_dataset(generator)

    # # Use this function with your generator and sim_name
    # max_batch = investigate_dataset(generator, sim_name)
    # print(f"**** Recommended batch size for {sim_name} is {max_batch}")

    # model_histories = simple_training()

    # for history in model_histories:
    #     print(history)

    class_label = GenerateFeatureLabelChunks()

    sim_names = [
        "Manhattan-config_v1%2FRainfall_Data_1.txt",
        "Manhattan-config_v1%2FRainfall_Data_2.txt",
        "Manhattan-config_v1%2FRainfall_Data_3.txt",
        "Manhattan-config_v1%2FRainfall_Data_4.txt",
        "Manhattan-config_v1%2FRainfall_Data_5.txt",
        "Manhattan-config_v1%2FRainfall_Data_7.txt",
        "Manhattan-config_v1%2FRainfall_Data_8.txt",
        "Manhattan-config_v1%2FRainfall_Data_13.txt",
        "Manhattan-config_v1%2FRainfall_Data_14.txt",
        "Manhattan-config_v1%2FRainfall_Data_15.txt",
        "Manhattan-config_v1%2FRainfall_Data_17.txt",
        "Manhattan-config_v1%2FRainfall_Data_22.txt",
        "Manhattan-config_v1%2FRainfall_Data_24.txt"    
    ]
    
    # download_sims_locally(class_label, generator, sim_names)

    model_histories = full_training(sim_names, generator)
    for history in model_histories:
        print(history)


# Run the main function
if __name__ == "__main__":
    main()

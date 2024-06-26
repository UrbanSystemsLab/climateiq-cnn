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


# def train_model(
#     model_dir: str,
#     batch_size: int,
#     lstm_units: int,
#     lstm_kernel_size: int,
#     lstm_dropout: float,
#     lstm_recurrent_dropout: float,
#     epochs: int,
# ):

#     # Create FloodModelParams from hyperparameters
#     model_params = FloodModelParams(
#         batch_size=batch_size,
#         lstm_units=lstm_units,
#         lstm_kernel_size=lstm_kernel_size,
#         lstm_dropout=lstm_dropout,
#         lstm_recurrent_dropout=lstm_recurrent_dropout,
#         epochs=epochs,
#     )
#     model_history = []

#     # Instantiate FloodModel class
#     model = FloodModel(model_params)
#     # model = model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate))

#     class_label = GenerateFeatureLabelChunks()
#     sim_names = class_label._get_sim_names_from_metastore()
#     print(sim_names)
#     batch_size = 32

#     generator = IncrementalTrainDataGenerator(
#         batch_size=batch_size,
#     )

#     # compare feature and label chunks
#     for sim_name in sim_names:
#         print(f"Comparing feature and label chunks for {sim_name}")
#         label_compare = class_label.compare_study_area_sim_chunks(sim_name)
#         if label_compare:
#             print("Feature and label chunks are equal for", sim_name)
#             print("")
#         else:
#             print("Feature and label chunks are not equal")
#             sim_names.remove(sim_name)

#     print("**** Length of sim_names: ", len(sim_names))

#     # Get the next batch of data
#     if len(sim_names) > 0:
#         data = generator.get_next_batch(sim_names, batch_size)
#         print("\n")
#         print("Dataset generation completed, will hand over to model training..")
#         print("\n")
#         print("Training model")
#         model_history.append(model.train(data))
#         print("Training complete")

#     else:
#         print("No valid sims found for training data generation")

#     return model_history


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
    batch_size=32,
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

    for sim_name in sim_names:
        floodModelDataset, storm_duration = generator.get_dataset_from_tensors(sim_name, batch_size=batch_size)
        print(type(floodModelDataset))
        print(storm_duration)

    #    # Iterate through the dataset using an iterator
    #     for element in floodModelDataset.take(2):  # Take only the first 10 elements
    #         # Access the elements within the tuple
    #         features, labels = element
    #         print("Features:", features)
    #         print("Labels:", labels)
    #         print("-" * 20)

        # Alternatively, use the `element_spec` to understand the structure
        print("Dataset element spec:", floodModelDataset.element_spec)

    print("Dataset generation completed, will hand over to model training..")
    print("\n")
    print("#######  Training model ##########")
    model_history.append(model.train(floodModelDataset, storm_duration))
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
    dataset, storm_duration = generator.get_dataset_from_tensors(sim_name, batch_size=32)
    print(type(dataset))
    print(storm_duration)

    print(f"Number of elements in dataset: {tf.data.experimental.cardinality(dataset).numpy()}")

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

    # # Iterate through the dataset using an iterator
    # for element in dataset.take(2):  # Take only the first 2 elements
    #     # Access the elements within the tuple
    #     features, labels = element
    #     print("Features:", features)
    #     print("Labels:", labels)
    #     print("-" * 20)

    #    # Print the shapes of the tensors within the features dictionary
    #     for key, value in features.items():
    #         print(f"Feature {key} shape: {value.shape}")
    #     print("-" * 20)

    #     # Print the shape of the labels tensor
    #     print("Labels shape:", labels.shape)
    #     print("-" * 20)


def set_tf_gpu():
    print(tf.__version__)
    # Check if TensorFlow was built with CUDA
    print("TensorFlow built with CUDA support:", tf.test.is_built_with_cuda())

    # List available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("Num GPUs Available: ", len(gpus))
    else:
        print("No GPUs found")

    # Configure TensorFlow for GPU usage
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Use only the first GPU (T4 in your case)
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logicalexpand_more GPUs")
        except RuntimeError as e:
            print(e)  # Handle potent


def main():
    batch_size = 32

    generator = IncrementalTrainDataGenerator(
        batch_size=batch_size,
    )

    # set_tf_gpu()
    
    #check_generator_tensor_shape(generator)

    #check_tensorflow_dataset(generator)
    
    #

    model_histories = simple_training()

    for history in model_histories:
        print(history.history)
   
    

    


    # generator = IncrementalTrainDataGenerator(
    #     batch_size=batch_size,
    # )

    # sim_names = [
    #     "Manhattan-config_v1%2FRainfall_Data_2.txt",
    #     "Manhattan-config_v1%2FRainfall_Data_13.txt",
    #     "Manhattan-config_v1%2FRainfall_Data_14.txt",
    #     "Manhattan-config_v1%2FRainfall_Data_15.txt",
    #     "Manhattan-config_v1%2FRainfall_Data_17.txt",
    # ]
    # # compare feature and label chunks
    # # for sim_name in sim_names:
    # #     print(f"Comparing feature and label chunks for {sim_name}")
    # #     label_compare = class_label.compare_study_area_sim_chunks(sim_name)
    # #     if label_compare:
    # #         print("Feature and label chunks are equal for", sim_name)
    # #         print("")
    # #     else:
    # #         print("Feature and label chunks are not equal")
    # #         sim_names.remove(sim_name)

    # print("**** Length of sim_names: ", len(sim_names))
    # print("Printing valid sims in metastore")
    # for sim_name in sim_names:
    #     print(f"Index: {sim_names.index(sim_name)} , Sim name: {sim_name}")
    #     print("")

    # # first_sim_name = sim_names[0]

    # # slice sim_names to get the first two elements
    # sim_names_2 = sim_names[:2]

    # # # download npy label chunks locally
    # # generator.download_numpy_files(sim_names_2, "label")

    # # download npy feature chunks locally
    # generator.download_numpy_files(sim_names_2, "feature")

    # # # download npy temporal chunks locally
    # # generator.download_numpy_files(sim_names_2, "temporal")

    # # print("Printing valid sims in metastore"))

    # for sim_name in sim_names_2:
    #     print(f"Index: {sim_names_2.index(sim_name)} , Sim name: {sim_name}")
    #     print("")

    # # Get the next batch of data for one simulation
    # data_batch = generator.get_next_batch(sim_names_2, batch_size)

    # print("\n")

    # # print size of data_batch
    # print("Data batch size:", len(data_batch))
    # print("\n")

    # # Print the shapes of the tensors in the batch
    # for data in data_batch:
    #     # Take the first element from tf dataset iterator and print shape of each tensors
    #     print("_" * 10, "\n\n")
    #     print("Temporal data shape:", next(iter(data.temporal.take(1))).shape)
    #     print("_" * 10, "\n\n")
    #     print("Labels shape:", next(iter(data.labels.take(1))).shape)
    #     print("_" * 10, "\n\n")
    #     print(
    #         "geospatial Feature tensor shape:",
    #         next(iter(data.geospatial.take(1))).shape,
    #     )
    #     print("_" * 10, "\n\n")
    #     print("Rainfall duration:", data.storm_duration)
    #     print("")
    #     print("-" * 20)

    #     # # Validate the shape of the labels tensor
    #     # first_label = next(iter(data.take(1)))[0]  # Get the first label batch
    #     # expected_label_shape = list(first_label.shape)
    #     # expected_label_shape[-1] = rainfall_duration

    #     # # Labels must match the storm duration.
    #     # assert first_label.shape[-1] == rainfall_duration, (  # Compare the shape of the first label tensor
    #     #     "Provided labels are inconsistent with storm duration. "
    #     #     f"Labels are expected to have shape {expected_label_shape}. "
    #     #     f"Actual shape: {first_label.shape}."  # Use first_label.shape
    #     # )
    #     # print("Labels are consistent with storm duration.")
    #     # print("-" * 20)




if __name__ == "__main__":
    main()

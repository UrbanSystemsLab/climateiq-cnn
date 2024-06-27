from typing import List
import os
import glob
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import functools
import re

import numpy as np
import tensorflow as tf
from google.cloud import firestore, storage

from usl_models.flood_ml.metastore import FirestoreDataHandler
from usl_models.flood_ml.settings import Settings
from usl_models.flood_ml.featurelabelchunks import GenerateFeatureLabelChunks
from usl_models.flood_ml import constants

"""
    This class is used to generate the training data for the flood model.
    Input:
        study_areas: List of study areas
        batch_size: Batch size
    Output:
        A generator that yields a batch of data.

    Note:
    This class uses Tensorflow TFRecords and TFExamples.
    TFRecords store data in a binary format and are optimized for reading and processinglarge datasets,
    making training fast. TFExamples are protocol buffers to store the data.

    tf.train.Feature object, which represents a single feature in the TFRecord example.
    The bytes_list is used to store the serialized NumPy array

    TFRecords are ideal for distributed training scenarios where we need to split data across
    multiple machines.

"""


class IncrementalTrainDataGenerator:
    def __init__(
        self,
        batch_size: int,
        settings: Settings = None,
        firestore_client: firestore.Client = None,
        storage_client: storage.Client = None,
        metastore: FirestoreDataHandler = None,
        featurelabelchunksgenerator: GenerateFeatureLabelChunks = None,
    ):
        print("Initializing IncrementalTrainDataGenerator...")
        self.current_index = 0  # Add an instance variable to track the current index
        self.common_indices = []
        self.feature_dict = {}
        self.label_dict = {}
        # Load settings
        self.settings = settings or Settings()

        self.firestore_client = firestore_client or firestore.Client()
        self.storage_client = storage_client or storage.Client()
        # self.firestore_collection = self.config.get("firestore_collection")
        # self.batch_size = batch_size
        self._chunks_iterator = None
        # self.rainfall_duration = None

        # instantiate metastore class
        self.metastore = metastore or FirestoreDataHandler(
            firestore_client=self.firestore_client, settings=self.settings
        )

        # instantiate featurelabelchunks class
        self.featurelabelchunksgenerator = (
            featurelabelchunksgenerator or GenerateFeatureLabelChunks()
        )

        # print(f"Firestore Collection: {self.firestore_collection}")
        print(f"Local Numpy Directory: {self.settings.LOCAL_NUMPY_DIR}")
        if self.settings.LOCAL_NUMPY_DIR is None:
            # set a default local directory
            self.settings.LOCAL_NUMPY_DIR = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "numpy_data"
            )
            print("Local Numpy Directory is not set, will create the directory.")
            os.makedirs(self.settings.LOCAL_NUMPY_DIR, exist_ok=True)

    # def _download_npy_from_gcs_in_memory(self, gcs_url):
    #     """
    #     Download a numpy file from GCS and return it as a BytesIO object.
    #     """
    #     if gcs_url is None:
    #         raise ValueError("GCS URL is empty.")
    #     bucket_name, blob_name = gcs_url.replace("gs://", "").split("/", 1)
    #     bucket = self.storage_client.bucket(bucket_name)
    #     blob = bucket.blob(blob_name)
    #     file_obj = BytesIO()
    #     blob.download_to_file(file_obj)
    #     file_obj.seek(0)
    #     return file_obj

    def download_numpy_files_in_dir(self, gcs_urls, dir_name, max_workers=5):
        """
        Download all numpy files from a directory in GCS to local storage.
        """
        if dir_name is None:
            print("GCS directory name is empty, returning...")
            return

        if os.path.exists(dir_name):
            npy_files = glob.glob(os.path.join(dir_name, "*.npy"))
            if npy_files:
                print(
                    f"Directory '{dir_name}' already contains .npy files, skipping download."
                )
                return
        else:
            os.makedirs(dir_name)
            print(f"Directory '{dir_name}' did not exist and was created.")

        print(f"Downloading numpy files from GCS directory to: {dir_name}")

        if len(gcs_urls) > 1:
            max_workers = min(self.settings.MAX_WORKERS, len(gcs_urls))
        else:
            max_workers = 1

        def _download_file(gcs_url, local_dir):
            def download_single_file(single_gcs_url):
                try:
                    print(f"Downloading file from GCS URL: {single_gcs_url}")
                    bucket_name, blob_name = single_gcs_url.replace("gs://", "").split(
                        "/", 1
                    )
                    bucket = self.storage_client.bucket(bucket_name)
                    blob = bucket.blob(blob_name)
                    local_path = os.path.join(local_dir, os.path.basename(blob_name))
                    blob.download_to_filename(local_path)
                    # print(f"Downloaded {single_gcs_url} to {local_path}")
                except Exception as e:
                    print(f"Error downloading file {single_gcs_url}: {e}")

            if gcs_url is None:
                return

            if isinstance(gcs_url, str):
                download_single_file(gcs_url)
            elif isinstance(gcs_url, list):
                for url in gcs_url:
                    download_single_file(url)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_download_file, gcs_url, dir_name)
                for gcs_url in gcs_urls
            ]

            for future in futures:
                try:
                    future.result()  # Wait for all futures to complete
                except Exception as e:
                    print(f"Error downloading chunk numpy files: {e}")
        print("Finished downloading numpy files.")


    def _generate_rainfall_duration(self, sim_name):
        print("Get rainfall duration...")
        rainfall_duration, rainfaill_dict = (
            self.featurelabelchunksgenerator.get_rainfall_config(sim_name)
        )
        if rainfall_duration is None:
            print(f"No rainfall duration found for sim {sim_name}.")
            return None
        return rainfall_duration

    def _generate_temporal_tensors(self, sim_name):
        """
        Create temporal tensors from the temporal chunks.
        """
        print("Generating temporal tensors...")
        # Get GCS URLs for npy chunks
        temporal_chunks, sim_dict = (
            self.featurelabelchunksgenerator.get_temporal_chunks(sim_name)
        )

        if temporal_chunks:
            if self.settings.DEBUG == 2:
                print(f"GCS URLs for sim {sim_name}: {temporal_chunks}")
        
        temporal_dir = sim_name + "_temporal"

        if os.path.exists(temporal_dir):
            print(f"Loading temporal tensors from {temporal_dir}...")
            for file_name in os.listdir(temporal_dir):
                if file_name.endswith(".npy"):
                    file_path = os.path.join(temporal_dir, file_name)
                    temporal_npy = np.load(file_path)
                    temporal_npy_tiled = np.tile(temporal_npy, (6, 1)).T
                    temporal_tensor = tf.convert_to_tensor(
                        temporal_npy_tiled,
                        dtype=tf.float32,
                    )
                    # Yield the same tensor infinitely
                    while True:
                        yield temporal_tensor
        else:
            print(f"No temporal chunks found for sim {sim_name}.")
            return None

    def download_numpy_files(self, sim_names, chunktype):
        """
        Download numpy files for the given sim_names.
        """
        for sim_name in sim_names:
            print(f"Downloading numpy files for sim: {sim_name}")

            # Get GCS URLs for npy chunks
            feature_chunks, sim_dict = (
                self.featurelabelchunksgenerator.get_feature_chunks(sim_name)
            )
            label_chunks, sim_dict = self.featurelabelchunksgenerator.get_label_chunks(
                sim_name
            )
            temporal_chunks = self.featurelabelchunksgenerator.get_temporal_chunks(
                sim_name
            )

            if chunktype == "feature":
                dir_name = sim_name + "_feature"
                for sim_name in sim_names:
                    self.download_numpy_files_in_dir(feature_chunks, dir_name)
            if chunktype == "label":
                dir_name = sim_name + "_label"
                for sim_name in sim_names:
                    self.download_numpy_files_in_dir(label_chunks, dir_name)
            if chunktype == "temporal":
                dir_name = sim_name + "_temporal"
                for sim_name in sim_names:
                    self.download_numpy_files_in_dir(temporal_chunks, dir_name)

    def rainfall_duration_generator(self, sim_name):
        print(f"Generating rainfall duration for sim_name: {sim_name}")
        return self._generate_rainfall_duration(sim_name)

    # # create a dummy dataset for Spatiotemporal tensors
    # def _create_dummy_dataset(self, input_shape):
    #     # Create a dummy dataset with the correct shape
    #     # Adjust the shape and dtype according to your needs
    #     return tf.data.Dataset.from_tensor_slices(tf.zeros(input_shape))

    # create a dummy *generator* for Spatiotemporal tensors
    def _generate_spatiotemporal_tensor(self, input_shape):
        while True:
            yield tf.zeros((input_shape))

    def _extract_index(self, file_name):
        match = re.search(r"(\d+_\d+)", file_name)
        if match:
            return match.group(1)
        return None

    def _list_numpy_files_sorted(self, directory_path):
        """Lists NumPy files in a directory, sorted alphabetically."""

        # List all files in the directory
        all_files = os.listdir(directory_path)

        # Filter for files with the '.npy' extension (NumPy's default format)
        numpy_files = [f for f in all_files if f.endswith(".npy")]

        # Sort the NumPy files alphabetically
        numpy_files.sort()

        return numpy_files

    def _generate_feature_label_tensors(self, sim_name):
        """
        Create feature and label tensors from the feature and label chunks.
        """
        print("Generating feature and label tensors...")

        rainfall_duration = self._generate_rainfall_duration(sim_name)
        if rainfall_duration is None:
            return None

        print(
            "The output label tensor will initially be of shape: ",
            [1000, 1000, rainfall_duration],
        )
        if not self.common_indices:
            feature_chunks, _ = self.featurelabelchunksgenerator.get_feature_chunks(sim_name)
            label_chunks, _ = self.featurelabelchunksgenerator.get_label_chunks(sim_name)

            if not feature_chunks or not label_chunks:
                print(f"No chunks found for sim {sim_name}.")
                return None

            # Define directories containing label and feature chunks
            feature_dir = sim_name + "_feature"
            label_dir = sim_name + "_label"

            # Create a sorted list of feature and label chunks from the contents of these directories on the file system
            feature_chunks = self._list_numpy_files_sorted(feature_dir)
            label_chunks = self._list_numpy_files_sorted(label_dir)

            # Create dictionaries to map indices to file paths
            self.feature_dict = {self._extract_index(f): f for f in feature_chunks}
            self.label_dict = {self._extract_index(f): f for f in label_chunks}

            # Ensure both dictionaries have the same keys (i.e., they are matched)
            self.common_indices = sorted(set(self.feature_dict.keys()).intersection(set(self.label_dict.keys())))
            print("Length of common indices:", len(self.common_indices))

            if len(self.common_indices) != len(feature_chunks) or len(self.common_indices) != len(label_chunks):
                raise ValueError("Number of matching feature chunks and label chunks do not match.")

        # Generate feature and label tensors in matched pairs
            for _ in range(len(self.common_indices)):
                # Get current index
                if self.current_index >= len(self.common_indices):
                    self.current_index = 0  # Reset index if it exceeds the number of available pairs

                index = self.common_indices[self.current_index]
                self.current_index += 1

                print(f"Using index: {index} (current_index: {self.current_index - 1})")  # Print the current index being used
                feature_dir = sim_name + "_feature"
                label_dir = sim_name + "_label"

                feature_path = os.path.join(feature_dir, self.feature_dict[index])
                label_path = os.path.join(label_dir, self.label_dict[index])
                
                print(f"Feature file: {self.feature_dict[index]}")
                print(f"Label file: {self.label_dict[index]}")

                # Load and yield feature and label tensors in matched pairs
                feature_tensor = tf.convert_to_tensor(
                    np.load(feature_path),
                    dtype=tf.float32,
                )
                label_tensor = tf.convert_to_tensor(
                    np.load(label_path), dtype=tf.float32
                )
                reshaped_label_tensor = tf.transpose(label_tensor, perm=[2, 0, 1])

                yield feature_tensor, reshaped_label_tensor

    def get_dataset_from_tensors(self, sim_name):
        """
        Get the dataset for training.
        """
        # Get the next batch of feature tensors
        print("Generating training data for sim_name: ", sim_name)

        storm_duration = self.rainfall_duration_generator(sim_name)
        print(f"Storm duration: {storm_duration}")

        # Define output signature for features and labels
        output_signature = (
            {
                "geospatial": tf.TensorSpec(shape=(1000, 1000, 8), dtype=tf.float32),
                "temporal": tf.TensorSpec(
                    shape=(864, constants.M_RAINFALL), dtype=tf.float32
                ),
                "spatiotemporal": tf.TensorSpec(
                    shape=(1000, 1000, 1), dtype=tf.float32
                ),
            },
            tf.TensorSpec(shape=(storm_duration, 1000, 1000), dtype=tf.float32),
        )

        def combined_generator(sim_name):  # Pass 'sim_name' as an argument
            feature_label_generator = self._generate_feature_label_tensors(sim_name)
            temporal_tensor_generator = self._generate_temporal_tensors(sim_name)

            spatiotemporal_tensor_generator = self._generate_spatiotemporal_tensor(
                [1000, 1000, 1]
            )
            print("Starting combined generator")
            count = 0
            print(f"Storm duration: {storm_duration}")

            # Combine the generators into a single generator
            for (geo, labels), temp, spatemp in zip(
                feature_label_generator,
                temporal_tensor_generator,
                spatiotemporal_tensor_generator
            ):
                yield (
                    {
                        "geospatial": geo,
                        "temporal": temp,
                        "spatiotemporal": spatemp,
                    },
                    labels,
                )
                count += 1
            print(f"Combined generator finished after {count} elements")

        # Create the dataset
        dataset = tf.data.Dataset.from_generator(
            functools.partial(
                combined_generator, sim_name
            ),  # Pass 'sim_name' to 'combined_generator'
            output_signature=output_signature,
        )
        # dataset = dataset.batch(batch_size)
        print("Ended creating dataset, returning..")
        # Return the batched dataset and storm duration
        return dataset, storm_duration

    # def get_next_batch(self, sim_names, batch_size) -> List[FloodModelData]:
    #     """
    #     Get the next batch of data for training.
    #     """
    #     # Get the next batch of feature tensors

    #     flood_model_data_list = []

    #     for sim_name in sim_names:
    #         print("\n")
    #         print(
    #             f"------ Starting training data generation:{sim_names.index(sim_name)} :: {sim_name} ------------"
    #         )
    #         print("Generating training data for sim_names: ", sim_names)

    #         # Lazy loading of datasets
    #         feature_dataset = tf.data.Dataset.from_generator(
    #             functools.partial(self._generate_feature_tensors, sim_name),
    #             output_types=tf.float32,
    #             # output_shapes=[1000, 1000, 8],
    #         ).batch(batch_size)

    #         label_dataset = tf.data.Dataset.from_generator(
    #             functools.partial(self._generate_label_tensors, sim_name),
    #             output_types=tf.float32,
    #             # output_shapes=[1000, 1000, rainfall_duration],
    #         ).batch(batch_size)

    #         temporal_dataset = tf.data.Dataset.from_generator(
    #             functools.partial(self._generate_temporal_tensors, sim_name),
    #             output_types=tf.float32,
    #             # output_shapes=[864],
    #         ).batch(batch_size)

    #         storm_duration = self.rainfall_duration_generator(sim_name)

    #         print(f"Storm duration: {storm_duration}")

    #         spatiotemporal_tensor = self._create_dummy_dataset([1, 1000, 1000, 1])

    #         # print type of these datasets
    #         print(f"feature_dataset type: {type(feature_dataset)}")
    #         print(f"label_dataset type: {type(label_dataset)}")
    #         print(f"temporal_dataset type: {type(temporal_dataset)}")
    #         print(f"spatiotemporal_tensor type: {type(spatiotemporal_tensor)}")

    #         if (
    #             label_dataset is None
    #             or feature_dataset is None
    #             or temporal_dataset is None
    #         ):
    #             print(f"Error: Missing data for simulation {sim_name}")
    #             continue  # Skip to the next simulation if any data is missing
    #         else:
    #             # Create a FloodModelData object
    #             flood_model_data = FloodModelData(
    #                 geospatial=feature_dataset,
    #                 labels=label_dataset,
    #                 temporal=temporal_dataset,
    #                 storm_duration=storm_duration,
    #                 spatiotemporal=spatiotemporal_tensor
    #             )
    #             flood_model_data_list.append(flood_model_data)

    #         print(
    #             "------ Finished training data generation for this simulation ------------"
    #         )

    #     # Return the FloodModelData object
    #     return flood_model_data_list

    # def _create_tfrecord_from_numpy(self, name, dir_name) -> List[tf.train.Example]:
    #     """
    #     Create TFRecord from numpy files.

    #     Args:
    #         name (str): Name of the feature in the TFRecord.

    #     Returns:
    #     List[tf.train.Example]: List of TFRecord examples.

    #     """
    #     print(
    #         f"Creating TFRecord from numpy files for: {name} from directory: {dir_name}"
    #     )
    #     serialized_examples_list = []

    #     # check if the LOCAL_NUMPY_DIR is not empty
    #     if not os.listdir(dir_name):
    #         print(f"{dir_name} is empty, will exit.")
    #         return []

    #     for file_name in os.listdir(dir_name):
    #         if file_name.endswith(".npy"):
    #             try:
    #                 file_path = os.path.join(dir_name, file_name)
    #                 data = np.load(file_path, mmap_mode="r")

    #                 # reshape temporal npy to match expected shape of
    #                 # constants.MAX_RAINFALL_DURATION x constants.M_RAINFALL
    #                 if name == "temporal_feature":
    #                     data = np.tile(data, (6, 1)).T
    #                     print("Temporal npy modified, shape: ", data.shape)

    #                 # Flatten the numpy array and convert it to a list of floats
    #                 data_list = data.flatten().tolist()

    #                 feature = {
    #                     name: tf.train.Feature(
    #                         float_list=tf.train.FloatList(value=data_list)
    #                     )
    #                 }

    #                 # Create an example protocol buffer
    #                 example = tf.train.Example(
    #                     features=tf.train.Features(feature=feature)
    #                 )

    #                 # Serialize to string and add to the list
    #                 serialized_examples_list.append(example.SerializeToString())

    #             except Exception as e:
    #                 print(f"Error processing numpy file {file_name}: {e}")
    #     if self.settings.DEBUG:
    #         print(f"Size of serialized examples list: {len(serialized_examples_list)}")

    #     # clean-up downloaded files from local storage. Uncomment this code once finalized.

    #     # for file_name in os.listdir(self.settings.LOCAL_NUMPY_DIR):
    #     #     if file_name.endswith(".npy"):
    #     #         os.remove(os.path.join(self.settings.LOCAL_NUMPY_DIR, file_name))

    #     print(
    #         "Finished creating seriliazed TFRecord from numpy files, files cleaned up from local storage. \n \n"
    #     )
    #     return serialized_examples_list

    # @staticmethod
    # def _parse_tf_example(serialized_example, feature_description):
    #     # Before parsing, log the serialized example to inspect its structure (for debugging purposes)
    #     print("Inspecting serialized example structure and data type...")
    #     print(f"Feature description: {feature_description}")
    #     # Attempt to parse the serialized example
    #     try:
    #         example = tf.io.parse_single_example(
    #             serialized_example, feature_description
    #         )
    #         return example
    #     except tf.errors.InvalidArgumentError as e:
    #         print(f"Failed to parse serialized example: {e}")
    #         # Optionally, add more detailed logging or error handling here
    #         raise

    # def _create_tfrecord_dataset(
    #     self, serialized_examples_list, feature_description, name
    # ) -> tf.data.Dataset:
    #     """
    #     Create a TensorFlow Dataset from a list of serialized TFRecord examples.
    #     """
    #     print(f"Creating TFRecord dataset for feature: {name}")
    #     dataset = tf.data.Dataset.from_tensor_slices(serialized_examples_list)
    #     dataset = dataset.map(lambda x: self._parse_tf_example(x, feature_description))
    #     return dataset

    # def _generate_feature_tensors(self, sim_name):
    #     """
    #     Create feature tensors from the feature chunks.
    #     """
    #     print("Generating feature tensors...")
    #     # Get GCS URLs for npy chunks
    #     feature_chunks, sim_dict = self.featurelabelchunksgenerator.get_feature_chunks(
    #         sim_name
    #     )

    #     if feature_chunks:
    #         if self.settings.DEBUG == 2:
    #             print(f"GCS URLs for sim {sim_name}: {feature_chunks}")

    #         # Download feature chunks npy files from GCS. Uncomment this when finalized.
    #         # self._download_numpy_files(feature_chunks)

    #         # Create TFRecord from numpy files
    #         serialized_examples_list = self._create_tfrecord_from_numpy(
    #             "geospatial_feature", sim_name + "_feature"
    #         )

    #         # Yield individual feature tensors
    #         for serialized_example in serialized_examples_list:
    #             feature_tensor = self._parse_tf_example(
    #                 serialized_example,
    #                 {
    #                     "geospatial_feature": tf.io.FixedLenFeature(
    #                         [1000, 1000, 8], tf.float32
    #                     ),
    #                 },
    #             )[
    #                 "geospatial_feature"
    #             ]  # Extract the 'feature' tensor from the dictionary
    #             yield feature_tensor  # Yield only the feature tensor
    #     else:
    #         print(f"No feature chunks found for sim {sim_name}.")
    #         return None

    # def _generate_label_tensors(self, sim_name):
    #     """
    #     Create label tensors from the label chunks.
    #     """
    #     print("Generating label tensors...")
    #     # get the storm duration for the simulation
    #     rainfall_duration = self._generate_rainfall_duration(sim_name)
    #     if rainfall_duration is None:
    #         return None

    #     print(
    #         "The output label tensor will initially be of shape: ",
    #         [1000, 1000, rainfall_duration],
    #     )

    #     # Get GCS URLs for npy chunks
    #     label_chunks, sim_dict = self.featurelabelchunksgenerator.get_label_chunks(
    #         sim_name
    #     )

    #     if label_chunks:
    #         if self.settings.DEBUG == 2:
    #             print(f"GCS URLs for sim {sim_name}: {label_chunks}")

    #         # This step is done as preprocessed before calling this function
    #         # self._download_numpy_files(label_chunks)

    #         # Create TFRecord from numpy files
    #         serialized_examples_list = self._create_tfrecord_from_numpy(
    #             "label", sim_name + "_label"
    #         )

    #         # Yield individual label tensors
    #         for serialized_example in serialized_examples_list:
    #             label_tensor = self._parse_tf_example(
    #                 serialized_example,
    #                 {
    #                     "label": tf.io.FixedLenFeature(
    #                         [1000, 1000, rainfall_duration], tf.float32
    #                     )
    #                 },
    #             )[
    #                 "label"
    #             ]  # Extract the 'label' tensor from the dictionary

    #             # Reshape the tensor to (rainfall_duration, 1000, 1000)
    #             reshaped_label_tensor = tf.transpose(label_tensor, perm=[2, 0, 1])
    #             yield reshaped_label_tensor  # Yield only the reshaped label tensor
    #     else:
    #         print(f"No label chunks found for sim {sim_name}.")
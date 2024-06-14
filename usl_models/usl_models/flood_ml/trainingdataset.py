from typing import List
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import tensorflow as tf
from google.cloud import firestore, storage

from usl_models.flood_ml.data_utils import FloodModelData
from usl_models.flood_ml.metastore import FirestoreDataHandler
from usl_models.flood_ml.settings import Settings
from usl_models.flood_ml.featurelabelchunks import GenerateFeatureLabelChunks

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

        # Load settings
        self.settings = settings or Settings()

        self.firestore_client = firestore_client or firestore.Client()
        self.storage_client = storage_client or storage.Client()
        # self.firestore_collection = self.config.get("firestore_collection")
        self.batch_size = batch_size
        self._chunks_iterator = None
        self.rainfall_duration = None

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

    def _download_npy_from_gcs_in_memory(self, gcs_url):
        """
        Download a numpy file from GCS and return it as a BytesIO object.
        """
        if gcs_url is None:
            raise ValueError("GCS URL is empty.")
        bucket_name, blob_name = gcs_url.replace("gs://", "").split("/", 1)
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        file_obj = BytesIO()
        blob.download_to_file(file_obj)
        file_obj.seek(0)
        return file_obj

    def _download_numpy_files(self, gcs_urls, max_workers=5):
        """
        Download numpy files from GCS to local storage. This is a multi-threaded approach to download files.

        Args:
            gcs_urls: List of GCS URLs to download.
            max_workers: Maximum number of workers to use for downloading.

        Returns:
            None
        """
        if gcs_urls is None or len(gcs_urls) == 0:
            print("GCS URLs are empty.")
            return

        print("Downloading numpy files...", self.settings.LOCAL_NUMPY_DIR)
        if len(gcs_urls) > 1:
            max_workers = min(self.settings.MAX_WORKERS, len(gcs_urls))
        else:
            max_workers = 1

        def _download_file(gcs_url, local_dir):

            if gcs_url is None:
                return

            try:
                # print(f"Downloading file from GCS URL: {gcs_url}")
                bucket_name, blob_name = gcs_url.replace("gs://", "").split("/", 1)
                bucket = self.storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                local_path = os.path.join(local_dir, os.path.basename(blob_name))
                blob.download_to_filename(local_path)
                # print(f"Downloaded {gcs_url} to {local_path}")
            except Exception as e:
                print(f"Error downloading file {gcs_url}: {e}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_download_file, gcs_url, self.settings.LOCAL_NUMPY_DIR)
                for gcs_url in gcs_urls
            ]

            for future in futures:
                try:
                    future.result()  # Wait for all futures to complete
                except Exception as e:
                    print(f"Error downloading chunk numpy files: {e}")
        print("Finished downloading numpy files.")

    def _create_tfrecord_from_numpy(self, name) -> List[tf.train.Example]:
        """
        Create TFRecord from numpy files.

        Args:
            name (str): Name of the feature in the TFRecord.

        Returns:
        List[tf.train.Example]: List of TFRecord examples.

        """
        print("Creating TFRecord from numpy files...")
        serialized_examples_list = []

        # check if the LOCAL_NUMPY_DIR is not empty
        if not os.listdir(self.settings.LOCAL_NUMPY_DIR):
            print("LOCAL_NUMPY_DIR is empty.")
            return []

        for file_name in os.listdir(self.settings.LOCAL_NUMPY_DIR):
            if file_name.endswith(".npy"):
                try:
                    file_path = os.path.join(self.settings.LOCAL_NUMPY_DIR, file_name)
                    data = np.load(file_path)
                    # print(f"Loaded numpy file: {file_path}")

                    # **Shape Validation**
                    # Get the shape of the loaded NumPy array
                    data_shape = data.shape

                    # Flatten the numpy array and convert it to a list of floats
                    data_list = data.flatten().tolist()

                    feature = {
                        name: tf.train.Feature(
                            float_list=tf.train.FloatList(value=data_list)
                        )
                    }

                    # Create an example protocol buffer
                    example = tf.train.Example(
                        features=tf.train.Features(feature=feature)
                    )

                    # Serialize to string and add to the list
                    serialized_examples_list.append(example.SerializeToString())
                    # print(f"Serialized example for file: {file_name}")
                except Exception as e:
                    print(f"Error processing numpy file {file_name}: {e}")
        if self.settings.DEBUG:
            print(f"Size of serialized examples list: {len(serialized_examples_list)}")

        # clean-up downloaded files from local storage. Uncomment this code once finalized.

        for file_name in os.listdir(self.settings.LOCAL_NUMPY_DIR):
            if file_name.endswith(".npy"):
                os.remove(os.path.join(self.settings.LOCAL_NUMPY_DIR, file_name))

        print(
            "Finished creating seriliazed TFRecord from numpy files, files cleaned up from local storage. \n \n"
        )
        return serialized_examples_list

    @staticmethod
    @staticmethod
    def _parse_tf_example(serialized_example, feature_description, name):
        """
        Parse TFRecord example given a feature description and serialized example.
        Example refers to a single example in the TFRecord tf.train.Example protocol buffer.

        This method is static since it parses the example using TensorFlow's
        tf.io.parse_single_example function, and returns the parsed example.
        """
        print(f"Parsing TFRecord example for feature: {name}")

        # Parse the serialized example using the feature description
        example = tf.io.parse_single_example(serialized_example, feature_description)

        label_tensor = example[name]

        return label_tensor

    def _create_tfrecord_dataset(
        self, serialized_examples_list, feature_description, name
    ) -> tf.data.Dataset:
        """
        Create a TensorFlow Dataset from a list of serialized TFRecord examples.
        """
        print(f"Creating TFRecord dataset for feature: {name}")
        dataset = tf.data.Dataset.from_tensor_slices(serialized_examples_list)
        dataset = dataset.map(
            lambda x: self._parse_tf_example(x, feature_description, name)
        )
        return dataset

    # def _create_feature_tensors(self, batch, sim_name):
    #     """
    #     Create feature tensors from the feature chunks.
    #     """
    #     # Get GCS URLs for npy chunks
    #     feature_chunks, sim_dict = self.featurelabelchunksgenerator.get_feature_chunks(
    #         sim_name
    #     )

    #     if feature_chunks:
    #         if self.settings.DEBUG == 2:
    #             print(f"GCS URLs for sim {sim_name}: {feature_chunks}")

    #         # Download label chunks npy files from GCS. Uncomment this when finalized.
    #         self._download_numpy_files(feature_chunks)

    #         # Create TFRecord from numpy files
    #         serialized_examples_list = self._create_tfrecord_from_numpy(
    #             "geospatial_feature"
    #         )

    #         # Create a TensorFlow Dataset from the serialized examples
    #         feature_dataset = self._create_tfrecord_dataset(
    #             serialized_examples_list,
    #             {
    #                 "geospatial_feature": tf.io.FixedLenFeature(
    #                     [1000, 1000, 8], tf.float32
    #                 ),
    #             },
    #             "geospatial_feature",
    #         )
    #         # Batch the dataset
    #         batched_feature_dataset = feature_dataset.batch(batch)

    #         return batched_feature_dataset
    #     else:
    #         print(f"No feature chunks found for sim {sim_name}.")
    #         return None

    # def _create_label_tensors(self, batch, sim_name):
    #     """
    #     Create label tensors from the label chunks.
    #     """
    #     # Get GCS URLs for npy chunks
    #     label_chunks, sim_dict = self.featurelabelchunksgenerator.get_label_chunks(
    #         sim_name
    #     )

    #     if label_chunks:
    #         if self.settings.DEBUG == 2:
    #             print(f"GCS URLs for sim {sim_name}: {label_chunks}")

    #         # Download label chunks npy files from GCS. Uncomment this when finalized.
    #         self._download_numpy_files(label_chunks)

    #         # Create TFRecord from numpy files
    #         serialized_examples_list = self._create_tfrecord_from_numpy("label")

    #         # Create a TensorFlow Dataset from the serialized examples
    #         label_dataset = self._create_tfrecord_dataset(
    #             serialized_examples_list,
    #             {
    #                 "label": tf.io.FixedLenFeature(
    #                     [1000, 1000, self.rainfall_duration], tf.float32
    #                 )
    #             },
    #             "label",
    #         )

    #         # Batch the dataset
    #         batched_label_dataset = label_dataset.batch(batch)

    #         return batched_label_dataset
    #     else:
    #         print(f"No label chunks found for sim {sim_name}.")
    #         return None

    # def _create_temporal_tensors(self, sim_name):
    #     """
    #     Create temporal tensors from the temporal chunks.
    #     """
    #     # Get GCS URLs for npy chunks
    #     temporal_chunks, sim_dict = (
    #         self.featurelabelchunksgenerator.get_temporal_chunks(sim_name)
    #     )

    #     if temporal_chunks:
    #         if self.settings.DEBUG == 2:
    #             print(f"GCS URLs for sim {sim_name}: {temporal_chunks}")

    #         # Download label chunks npy files from GCS. Uncomment this when finalized.
    #         self._download_numpy_files(temporal_chunks)

    #         # Create TFRecord from numpy files
    #         serialized_examples_list = self._create_tfrecord_from_numpy(
    #             "temporal_feature"
    #         )

    #         # Create a TensorFlow Dataset from the serialized examples
    #         temporal_dataset = self._create_tfrecord_dataset(
    #             serialized_examples_list,
    #             {"temporal_feature": tf.io.FixedLenFeature([864], tf.float32)},
    #             "temporal_feature",
    #         )

    #         return temporal_dataset
    #     else:
    #         print(f"No temporal chunks found for sim {sim_name}.")
    #         return None

    def _generate_rainfall_duration(self, sim_name):
        rainfall_duration, rainfaill_dict = (
            self.featurelabelchunksgenerator.get_rainfall_config(sim_name)
        )
        if rainfall_duration is None:
            print(f"No rainfall duration found for sim {sim_name}.")
            return None
        return rainfall_duration

    def _generate_feature_tensors(self, sim_name):
        """
        Create feature tensors from the feature chunks.
        """
        # Get GCS URLs for npy chunks
        feature_chunks, sim_dict = self.featurelabelchunksgenerator.get_feature_chunks(
            sim_name
        )

        if feature_chunks:
            if self.settings.DEBUG == 2:
                print(f"GCS URLs for sim {sim_name}: {feature_chunks}")

            # Download label chunks npy files from GCS. Uncomment this when finalized.
            self._download_numpy_files(feature_chunks)

            # Create TFRecord from numpy files
            serialized_examples_list = self._create_tfrecord_from_numpy(
                "geospatial_feature"
            )

            # Yield batches of data
            for i in range(0, len(serialized_examples_list), self.batch_size):
                yield [
                    self._parse_tf_example(
                        serialized_examples_list[j],
                        {
                            "geospatial_feature": tf.io.FixedLenFeature(
                                [1000, 1000, 8], tf.float32
                            ),
                        },
                        "geospatial_feature",
                    )
                    for j in range(i, min(i + self.batch_size, len(serialized_examples_list)))
                ]
        else:
            print(f"No feature chunks found for sim {sim_name}.")
            return None

    def _generate_label_tensors(self, sim_name, rainfall_duration):
        """
        Create label tensors from the label chunks.
        """
        print("The output label tensor will be of shape: ", [1000, 1000, rainfall_duration])

        # Get GCS URLs for npy chunks
        label_chunks, sim_dict = self.featurelabelchunksgenerator.get_label_chunks(
            sim_name
        )

        if label_chunks:
            if self.settings.DEBUG == 2:
                print(f"GCS URLs for sim {sim_name}: {label_chunks}")

            # Download label chunks npy files from GCS. Uncomment this when finalized.
            self._download_numpy_files(label_chunks)

            # Create TFRecord from numpy files
            serialized_examples_list = self._create_tfrecord_from_numpy("label")

            # Yield batches of data
            for i in range(0, len(serialized_examples_list), self.batch_size):
                for j in range(i, min(i + self.batch_size, len(serialized_examples_list))):
                    # Yield the label tensor directly
                    yield self._parse_tf_example(
                        serialized_examples_list[j],
                        {
                            "label": tf.io.FixedLenFeature(
                                [1000, 1000, rainfall_duration], tf.float32
                            )
                        },
                        "label",
                    )
        else:
            print(f"No label chunks found for sim {sim_name}.")
            return None

    def _generate_temporal_tensors(self, sim_name):
        """
        Create temporal tensors from the temporal chunks.
        """
        # Get GCS URLs for npy chunks
        temporal_chunks, sim_dict = (
            self.featurelabelchunksgenerator.get_temporal_chunks(sim_name)
        )

        if temporal_chunks:
            if self.settings.DEBUG == 2:
                print(f"GCS URLs for sim {sim_name}: {temporal_chunks}")

            # Download label chunks npy files from GCS. Uncomment this when finalized.
            self._download_numpy_files(temporal_chunks)

            # Create TFRecord from numpy files
            serialized_examples_list = self._create_tfrecord_from_numpy(
                "temporal_feature"
            )

            # Yield batches of data
            for i in range(0, len(serialized_examples_list), self.batch_size):
                yield [
                    self._parse_tf_example(
                        serialized_examples_list[j],
                        {"temporal_feature": tf.io.FixedLenFeature([864], tf.float32)},
                        "temporal_feature",
                    )
                    for j in range(i, min(i + self.batch_size, len(serialized_examples_list)))
                ]
        else:
            print(f"No temporal chunks found for sim {sim_name}.")
            return None
    
    def get_next_batch(self, sim_names, batch) -> List[FloodModelData]:
        """
        Get the next batch of data for training.
        """
        # Get the next batch of feature tensors

        flood_model_data_list = []

        for sim_name in sim_names:
            print("\n")
            print(f"------ Starting training data generation:{sim_names.index(sim_name)} :: {sim_name} ------------")
            print("Generating training data for sim_names: ", sim_names)
            # Get the rainfall duration for the current simulation
            rainfall_duration = self._generate_rainfall_duration(sim_name)
            

            # Lazy loading of datasets
            feature_dataset = tf.data.Dataset.from_generator(
                lambda: self._generate_feature_tensors(sim_name),
                output_types=tf.float32,
                output_shapes=[1000, 1000, 8],
            )
            label_dataset = tf.data.Dataset.from_generator(
                lambda: self._generate_label_tensors(sim_name, rainfall_duration),
                output_types=tf.float32,
                output_shapes=[1000, 1000, rainfall_duration],
            )
            temporal_dataset = tf.data.Dataset.from_generator(
                lambda: self._generate_temporal_tensors(sim_name),
                output_types=tf.float32,
                output_shapes=[864],
            )

            # print type of these datasets
            print(f"feature_dataset type: {type(feature_dataset)}")
            print(f"label_dataset type: {type(label_dataset)}")
            print(f"temporal_dataset type: {type(temporal_dataset)}")

            if (
                label_dataset is None
                or feature_dataset is None
                or temporal_dataset is None
            ):
                print(f"Error: Missing data for simulation {sim_name}")
                continue  # Skip to the next simulation if any data is missing
            else:
                # Create a FloodModelData object
                flood_model_data = FloodModelData(
                    geospatial=feature_dataset,
                    labels=label_dataset,
                    temporal=temporal_dataset,
                    storm_duration=rainfall_duration,
                )
                flood_model_data_list.append(flood_model_data)

            print(
                "------ Finished training data generation for this simulation ------------"
            )

        # Return the FloodModelData object
        return flood_model_data_list





    # def get_next_batch(self, sim_names, batch) -> List[FloodModelData]:
    #     """
    #     Get the next batch of data for training.
    #     """
    #     # Get the next batch of feature tensors

    #     flood_model_data_list = []

    #     for sim_name in sim_names:
    #         print("\n")
    #         print("------ Starting training data generation ------------")
    #         print("Generating training data for sim_names: ", sim_names)
    #         # Get the rainfall duration for the current simulation
    #         rainfall_duration = self._get_rainfall_duration(sim_name)
    #         if rainfall_duration:
    #             self.rainfall_duration = rainfall_duration
    #         else:
    #             print(f"No rainfall duration found for sim {sim_name}.")

    #         feature_dataset = self._create_feature_tensors(batch, sim_name)
    #         label_dataset = self._create_label_tensors(
    #             batch,
    #             sim_name,
    #         )
    #         temporal_dataset = self._create_temporal_tensors(sim_name)

    #         # print type of these datasets
    #         print(f"feature_dataset type: {type(feature_dataset)}")
    #         print(f"label_dataset type: {type(label_dataset)}")
    #         print(f"temporal_dataset type: {type(temporal_dataset)}")

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
    #                 storm_duration=rainfall_duration,
    #             )
    #             flood_model_data_list.append(flood_model_data)

    #         print(
    #             "------ Finished training data generation for this simulation ------------"
    #         )

    #     # Return the FloodModelData object
    #     return flood_model_data_list

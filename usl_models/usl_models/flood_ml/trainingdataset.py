from typing import List
import os
import glob
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import functools

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

    def download_numpy_files_in_dir(self, gcs_urls, dir_name, max_workers=5):
        """
        Download all numpy files from a directory in GCS to local storage.
        """
        if dir_name is None:
            print("GCS directory name is empty, returning...")
            return

        if os.path.exists(dir_name):
            npy_files = glob.glob(os.path.join(dir_name, '*.npy'))
            if npy_files:
                print(f"Directory '{dir_name}' already contains .npy files, skipping download.")
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
                    # print(f"Downloading file from GCS URL: {single_gcs_url}")
                    bucket_name, blob_name = single_gcs_url.replace("gs://", "").split("/", 1)
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

    def _create_tfrecord_from_numpy(self, name, dir_name) -> List[tf.train.Example]:
        """
        Create TFRecord from numpy files.

        Args:
            name (str): Name of the feature in the TFRecord.

        Returns:
        List[tf.train.Example]: List of TFRecord examples.

        """
        print("Creating TFRecord from numpy files for: ", dir_name)
        serialized_examples_list = []

        # check if the LOCAL_NUMPY_DIR is not empty
        if not os.listdir(dir_name):
            print(f"{dir_name} is empty, will exit.")
            return []

        for file_name in os.listdir(dir_name):
            if file_name.endswith(".npy"):
                try:
                    file_path = os.path.join(dir_name, file_name)
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

        # for file_name in os.listdir(self.settings.LOCAL_NUMPY_DIR):
        #     if file_name.endswith(".npy"):
        #         os.remove(os.path.join(self.settings.LOCAL_NUMPY_DIR, file_name))

        print(
            "Finished creating seriliazed TFRecord from numpy files, files cleaned up from local storage. \n \n"
        )
        return serialized_examples_list

    @staticmethod
    def _parse_tf_example(serialized_example, feature_description, name):
        """
        Parse TFRecord example given a feature description and serialized example.
        Example refers to a single example in the TFRecord tf.train.Example protocol buffer.

        This method is static since it parses the example using TensorFlow's
        tf.io.parse_single_example function, and returns the parsed example.
        """
        #print(f"Parsing TFRecord example for feature: {name}")

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

            # Download feature chunks npy files from GCS. Uncomment this when finalized.
            #self._download_numpy_files(feature_chunks)

            # Create TFRecord from numpy files
            serialized_examples_list = self._create_tfrecord_from_numpy(
                "geospatial_feature", sim_name+"_feature"
            )

            # Yield individual feature tensors
            for serialized_example in serialized_examples_list:
                yield self._parse_tf_example(
                    serialized_example,
                    {
                        "geospatial_feature": tf.io.FixedLenFeature(
                            [1000, 1000, 8], tf.float32
                        ),
                    },
                    "geospatial_feature",
                )
        else:
            print(f"No feature chunks found for sim {sim_name}.")
            return None

    def _generate_label_tensors(self, sim_name):
        """
        Create label tensors from the label chunks.
        """
        # get the storm duration for the simulation
        rainfall_duration = self._generate_rainfall_duration(sim_name)
        if rainfall_duration is None:
            return None

        print(
            "The output label tensor will be of shape: ",
            [1000, 1000, rainfall_duration],
        )

        # Get GCS URLs for npy chunks
        label_chunks, sim_dict = self.featurelabelchunksgenerator.get_label_chunks(
            sim_name
        )

        if label_chunks:
            if self.settings.DEBUG == 2:
                print(f"GCS URLs for sim {sim_name}: {label_chunks}")

            # This step is done as preprocessed before calling this function
           # self._download_numpy_files(label_chunks)

            # Create TFRecord from numpy files
            serialized_examples_list = self._create_tfrecord_from_numpy("label", sim_name+"_label")

            # Yield individual label tensors
            for serialized_example in serialized_examples_list:
                yield self._parse_tf_example(
                    serialized_example,
                    {
                        "label": tf.io.FixedLenFeature(
                            [1000, 1000, rainfall_duration], tf.float32
                        )
                    },
                    "label",
                )
        else:
            print(f"No label chunks found for sim {sim_name}.")

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
            #self._download_numpy_files(temporal_chunks)

            # Create TFRecord from numpy files
            serialized_examples_list = self._create_tfrecord_from_numpy(
                "temporal_feature", sim_name+"_temporal"
            )

            for serialized_example in serialized_examples_list:
                yield self._parse_tf_example(
                    serialized_example,
                    {"temporal_feature": tf.io.FixedLenFeature([864], tf.float32)},
                    "temporal_feature",
                )

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
            feature_chunks, sim_dict = self.featurelabelchunksgenerator.get_feature_chunks(
                sim_name
            )
            label_chunks, sim_dict = self.featurelabelchunksgenerator.get_label_chunks(
                sim_name
            )
            temporal_chunks = self.featurelabelchunksgenerator.get_temporal_chunks(
                sim_name
            )

            if chunktype == "feature":
                dir_name = sim_name+"_feature"
                for sim_name in sim_names:
                    self.download_numpy_files_in_dir(feature_chunks, dir_name)
            if chunktype == "label":
                dir_name = sim_name+"_label"
                for sim_name in sim_names:
                    self.download_numpy_files_in_dir(label_chunks, dir_name)
            if chunktype == "temporal":
                dir_name = sim_name+"_temporal"
                for sim_name in sim_names:
                    self.download_numpy_files_in_dir(temporal_chunks, dir_name)
            
    def rainfall_duration_generator(self, sim_name):
        print(f"Generating rainfall duration for sim_name: {sim_name}")
        return self._generate_rainfall_duration(sim_name)

    def get_next_batch(self, sim_names, batch_size) -> List[FloodModelData]:
        """
        Get the next batch of data for training.
        """
        # Get the next batch of feature tensors

        flood_model_data_list = []

        for sim_name in sim_names:
            print("\n")
            print(
                f"------ Starting training data generation:{sim_names.index(sim_name)} :: {sim_name} ------------"
            )
            print("Generating training data for sim_names: ", sim_names)

            # Lazy loading of datasets
            feature_dataset = tf.data.Dataset.from_generator(
                functools.partial(self._generate_feature_tensors, sim_name),
                output_types=tf.float32,
                # output_shapes=[1000, 1000, 8],
            ).batch(batch_size)

            label_dataset = tf.data.Dataset.from_generator(
                functools.partial(self._generate_label_tensors, sim_name),
                output_types=tf.float32,
                # output_shapes=[1000, 1000, rainfall_duration],
            ).batch(batch_size)

            temporal_dataset = tf.data.Dataset.from_generator(
                functools.partial(self._generate_temporal_tensors, sim_name),
                output_types=tf.float32,
                # output_shapes=[864],
            ).batch(batch_size)

            storm_duration = self.rainfall_duration_generator(sim_name)

            print(f"Storm duration: {storm_duration}")

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
                    storm_duration=storm_duration,
                )
                flood_model_data_list.append(flood_model_data)

            print(
                "------ Finished training data generation for this simulation ------------"
            )

        # Return the FloodModelData object
        return flood_model_data_list

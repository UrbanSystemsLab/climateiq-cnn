from typing import List
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import tensorflow as tf
from google.cloud import firestore, storage

from data_utils import FloodModelData
from metastore import FirestoreDataHandler
from settings import Settings
from geospatial import GeospatialTensor
from spatiotemporal import SpatialTemporalTensor


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
        study_areas: List[str],
        batch_size: int,
        settings: Settings = None,
        firestore_client: firestore.Client = None,
        storage_client: storage.Client = None,
        metastore: FirestoreDataHandler = None,
        geospatial_tensor: GeospatialTensor = None,
        spatial_temporal_tensor: SpatialTemporalTensor = None,
    ):
        print("Initializing IncrementalTrainDataGenerator...")

        # Load settings
        self.settings = settings or Settings()

        self.firestore_client = firestore_client or firestore.Client()
        self.storage_client = storage_client or storage.Client()
        # self.firestore_collection = self.config.get("firestore_collection")
        self.study_areas = study_areas
        self.batch_size = batch_size
        self._chunks_iterator = None

        # instantiate metastore class
        self.metastore = metastore or FirestoreDataHandler(
            firestore_client=self.firestore_client, settings=self.settings
        )

        # instantiate geospatial tensor class
        self.geospatial_tensor = geospatial_tensor or GeospatialTensor(
            settings=self.settings
        )

        # instantiate spatial temporal tensor class
        self.spatial_temporal_tensor = spatial_temporal_tensor or SpatialTemporalTensor(
            settings=self.settings
        )

        # Print initialization details
        print(f"Initialized with study_areas: {study_areas}, batch_size: {batch_size}")
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
            raise ValueError("GCS URLs are empty.")
        print("Downloading numpy files...", self.settings.LOCAL_NUMPY_DIR)
        max_workers = min(self.settings.MAX_WORKERS, len(gcs_urls))

        def _download_file(gcs_url, local_dir):
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

                    # Create a feature
                    feature = {
                        name: tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[tf.io.serialize_tensor(data).numpy()]
                            )
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
        # This should speed up testing.
        # for file_name in os.listdir(self.settings.LOCAL_NUMPY_DIR):
        #     if file_name.endswith(".npy"):
        #         os.remove(os.path.join(self.settings.LOCAL_NUMPY_DIR, file_name))

        print(
            "Finished creating seriliazed TFRecord from numpy files, files cleaned up from local storage. \n \n"
        )
        return serialized_examples_list

    @staticmethod
    def _parse_tf_example(serialized_example, feature_description):
        """
        Parse TFRecord example given a feature description and serialized example.
        Example refers to a single example in the TFRecord tf.train.Example protocol buffer.

        This method is static since it parses the example using TensorFlow's
        tf.io.parse_single_example function, and returns the parsed example.
        """
        example = tf.io.parse_single_example(serialized_example, feature_description)

        # Deserialize the labels tensor
        labels_tensor = tf.io.parse_tensor(example["labels"], out_type=tf.float32)

        # Reshape the labels tensor to match model's expected input shape
        # the model expects labels of shape (1000, 1000, 4)
        labels_tensor = tf.reshape(labels_tensor, (1000, 1000, 4))

        # Update the example with the reshaped labels tensor
        example["labels"] = labels_tensor

        return example

    def _create_tfrecord_dataset(self, serialized_examples_list, feature_description):
        """
        Create a TensorFlow Dataset from a list of serialized TFRecord examples.
        """
        dataset = tf.data.Dataset.from_tensor_slices(serialized_examples_list)
        dataset = dataset.map(lambda x: self._parse_tf_example(x, feature_description))
        return dataset

    def _retrieve_simulation_data(self, collections_info, study_areas):
        """Retrieves simulation document information for each study area."""
        simulation_data = {}
        for study_area in study_areas:
            simulation_document = self.metastore._find_simulation_document(
                collections_info, study_area
            )
            if simulation_document:
                print(f"Found simulation document for {study_area}:")
                print(f"Document ID: {simulation_document['document_id']}")
                print(f"Subcollections: {simulation_document['subcollections']}")

                simulation_data[study_area] = {
                    "document_id": simulation_document["document_id"],
                    "simulation_name": simulation_document["document_id"].split("%2F")[
                        0
                    ],
                }
            else:
                raise ValueError(f"No simulation document found for {study_area}")
        return simulation_data

    def _process_study_area_data(self, study_area, data, batch) -> List[FloodModelData]:
        """Processes data for a specific study area."""
        flood_model_data_list = []

        rainfall_info = self.metastore._extract_rainfall_info()

        for info in rainfall_info:
            if self.settings.DEBUG:
                print(f"Rainfall Duration: {info['rainfall_duration']}")
                print(f"As Vector GCS URI: {info['as_vector_gcs_uri']}")

            simulation_name = data["simulation_name"]
            parent_config_name = data["document_id"]

            # Get GCS URLs for npy chunks
            gcs_urls = self.metastore._get_label_chunks_urls(parent_config_name)

            if self.settings.DEBUG == 2:
                print(f"GCS URLs for simulation {simulation_name}: {gcs_urls}")

            # Download label chunks npy files from GCS. Uncomment this when finalized.
            # self._download_numpy_files(gcs_urls)

            # Create TFRecord from numpy files
            serialized_examples_list = self._create_tfrecord_from_numpy("labels")

            # Download temporal data
            temporal_data = np.load(
                self._download_npy_from_gcs_in_memory(info["as_vector_gcs_uri"])
            )
            if self.settings.DEBUG:
                print(f"Size of temporal data: {temporal_data.shape}")

            # Convert temporal data to tensor
            temporal_tensor = tf.convert_to_tensor(temporal_data)

            # rainfall duration
            rainfall_duration = info["rainfall_duration"]

            # Create a TensorFlow Dataset from the serialized examples
            dataset = self._create_tfrecord_dataset(
                serialized_examples_list,
                {"labels": tf.io.FixedLenFeature([], tf.string)},
            )

            # Batch the dataset
            dataset = dataset.batch(batch)

            # Process each batch
            flood_model_data_list.extend(
                self._process_batch(
                    dataset, study_area, rainfall_duration, temporal_tensor
                )
            )

        return flood_model_data_list

    def _generate_geospatial_tensor(self, study_area, data) -> tf.Tensor:
        """Generates geospatial tensor for a specific study area."""
        geospatial_tensor = self.geospatial_tensor.generate_geospatial_tensor(
            study_area, data
        )
        return geospatial_tensor

    def _generate_spatio_temporal_tensor(self, study_area, data) -> tf.Tensor:
        """Generates spatio-temporal tensor for a specific study area."""
        spatio_temporal_tensor = (
            self.spatial_temporal_tensor.generate_spatiotemporal_data(study_area, data)
        )
        return spatio_temporal_tensor

    def _process_batch(
        self, dataset, study_area, rainfall_duration, temporal_tensor=None
    ) -> List[FloodModelData]:
        """Processes a single batch of data."""
        flood_model_data_list = []
        for batch_data in dataset:
            # Access tensors from batch_data
            labels_tensor = batch_data["labels"]

            # To-Do : Generate geospatial_tensor from features, for now hardcode
            geospatial_tensor = self._generate_geospatial_tensor(study_area, dataset)
            # To-Do : Generate spatio_temporal_tensor from features, for now hardcode
            # Spatio-temporal tensor. A *single* flood map to specify the initial flood conditions. [H, W, 1]
            spatio_temporal_tensor = self._generate_spatio_temporal_tensor(
                study_area, dataset
            )

            # Create a FloodModelData instance
            flood_model_data = FloodModelData(
                storm_duration=rainfall_duration,  # integer values retrieved from firestore
                geospatial=geospatial_tensor,  # hardcoded tensors, to-do : generate from features
                temporal=temporal_tensor,  # this tensor is generated from the temporal data , as_vector_gcs_uri
                spatiotemporal=spatio_temporal_tensor,  # hardcoded tensors, to-do : generate from features
                labels=labels_tensor,  # this tensor is generated from the labels, the npy chunks created from
                # the raster
            )

            # Append the FloodModelData instance to the list
            flood_model_data_list.append(flood_model_data)

        return flood_model_data_list

    def get_next_batch(self, study_areas, batch) -> dict[str, FloodModelData]:
        """
        Get next batch of data.

        Args:
            study_areas: A list of study areas to get data for.
            batch: The batch size.

        Returns:
                A dictionary containing the study area and corresponding FloodModelData object.

        """
        print("Getting next batch...")

        if not study_areas or not study_areas[0]:
            raise ValueError("study_areas cannot be empty")
        if not isinstance(study_areas, list):
            raise ValueError("study_areas must be a list")
        if not isinstance(batch, int) or batch <= 1:
            raise ValueError(
                "batch must be a positive integer greater than or equal to 1"
            )

        collections_info = self.metastore._list_collections_and_print_documents()
        print(f"Number of collections: {len(collections_info)}")
        # Validate collections_info
        if not collections_info:
            raise ValueError("No collections found")

        study_area_data = {}  # Dictionary to store data for each study area

        # Retrieve Simulation Data
        simulation_data = self._retrieve_simulation_data(collections_info, study_areas)
        if not simulation_data:
            raise ValueError("No simulation data found for the given study areas")

        # Process Data for Each Study Area
        for study_area, data in simulation_data.items():
            print(f"Study Area: {study_area}")
            print(f"Document ID: {data['document_id']}")
            print(f"Simulation Name: {data['simulation_name']}")

            flood_model_data_list = self._process_study_area_data(
                study_area, data, batch
            )

            if flood_model_data_list:
                study_area_data[study_area] = flood_model_data_list
                print("Finished getting next batch for study area: ", study_area, "\n")
            else:
                print(
                    "Error: Error producing training data for study area: ", study_area
                )

        return study_area_data

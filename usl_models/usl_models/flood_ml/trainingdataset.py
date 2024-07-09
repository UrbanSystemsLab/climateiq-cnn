import os
import glob
from concurrent.futures import ThreadPoolExecutor
import functools
import re
import requests
from io import BytesIO
import random


import numpy as np
import tensorflow as tf
from google.cloud import firestore, storage

from usl_models.flood_ml.metastore import FirestoreDataHandler
from usl_models.flood_ml.settings import Settings
from usl_models.flood_ml.featurelabelchunks import GenerateFeatureLabelChunks
from usl_models.flood_ml import constants

from tqdm import tqdm
"""
    This class is used to generate the training data for the flood model.
    Input:
        study_areas: List of study areas
        batch_size: Batch size
    Output:
        A generator that yields a batch of data.

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
     
        self.settings = settings or Settings()

        self.firestore_client = firestore_client or firestore.Client()
        self.storage_client = storage_client or storage.Client()
       

        # instantiate metastore class
        self.metastore = metastore or FirestoreDataHandler(
            firestore_client=self.firestore_client, settings=self.settings
        )

        # instantiate featurelabelchunks class
        self.featurelabelchunksgenerator = (
            featurelabelchunksgenerator or GenerateFeatureLabelChunks()
        )

        # print(f"Firestore Collection: {self.firestore_collection}")
        # print(f"Local Numpy Directory: {self.settings.LOCAL_NUMPY_DIR}")
    
    def _generate_rainfall_duration(self, sim_name):
        # print("Get rainfall duration...")
        rainfall_duration, rainfaill_dict = (
            self.featurelabelchunksgenerator.get_rainfall_config(sim_name)
        )
        if rainfall_duration is None:
            print(f"No rainfall duration found for sim {sim_name}.")
            return None
        return rainfall_duration

    def _generate_temporal_tensors(self, sim_name):
        """
        Create temporal tensors from the temporal chunks using GCS URLs.
        """
        print("Generating temporal tensors...")
        # Get GCS URLs for npy chunks
        temporal_chunks, sim_dict = self.featurelabelchunksgenerator.get_temporal_chunks(sim_name)
        
        if not temporal_chunks:
            print(f"No temporal chunks found for sim {sim_name}.")
            return None

        if self.settings.DEBUG == 2:
            print(f"GCS URLs for sim {sim_name}: {temporal_chunks}")

        for temporal_url in temporal_chunks:
            print(f"Loading temporal tensor from {temporal_url}...")
            
            # Download data from GCS URL using Google Cloud Storage API
            bucket_name, blob_name = temporal_url.replace("gs://", "").split("/", 1)
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            temporal_data = blob.download_as_string()

            # Load the numpy array from the downloaded content
            temporal_npy = np.load(BytesIO(temporal_data))
            temporal_npy_tiled = np.tile(temporal_npy, (6, 1)).T
            temporal_tensor = tf.convert_to_tensor(
                temporal_npy_tiled,
                dtype=tf.float32,
            )
            print("Temporal tensor shape: ", temporal_tensor.shape)
            # Yield the same tensor infinitely
            while True:
                yield temporal_tensor


    def rainfall_duration_generator(self, sim_name):
        print(f"Generating rainfall duration for sim_name: {sim_name}")
        return self._generate_rainfall_duration(sim_name)

    # create a dummy *generator* for Spatiotemporal tensors
    def _generate_spatiotemporal_tensor(self, input_shape):
        print("Spatiotemporal tensor shape: ", input_shape)
        while True:
            yield tf.zeros((input_shape))

    def _extract_index(self, file_name):
        match = re.search(r"(\d+_\d+)", file_name)
        if match:
            return match.group(1)
        return None

    def _generate_feature_label_tensors(self, sim_name):
        """
        Create feature and label tensors from the feature and label chunks using GCS URLs.
        """
        print("Generating feature and label tensors...")

        rainfall_duration = self._generate_rainfall_duration(sim_name)
        if rainfall_duration is None:
            return None

        print(
            "The output label tensor will initially be of shape: ",
            [1000, 1000, rainfall_duration],
        )

        feature_chunks, _ = self.featurelabelchunksgenerator.get_feature_chunks(sim_name)
        label_chunks, _ = self.featurelabelchunksgenerator.get_label_chunks(sim_name)

        if not feature_chunks or not label_chunks:
            print(f"No chunks found for sim {sim_name}.")
            return None

        # Create dictionaries to map indices to GCS URLs (local variables)
        feature_dict = {self._extract_index(url): url for url in feature_chunks}
        label_dict = {self._extract_index(url): url for url in label_chunks}

        # Ensure both dictionaries have the same keys (i.e., they are matched)
        common_indices = sorted(
            set(feature_dict.keys()).intersection(set(label_dict.keys()))
        )
        print("Length of common indices:", len(common_indices))

        # Shuffle the common indices
        # random.shuffle(common_indices)  # Shuffle directly within the function

        if len(common_indices) != len(feature_chunks) or len(common_indices) != len(label_chunks):
            raise ValueError(
                "Number of matching feature chunks and label chunks do not match."
            )

        # Generate feature and label tensors in matched pairs
        for index in common_indices:  # Iterate over local common_indices
            feature_url = feature_dict[index]  # Use local dictionaries
            label_url = label_dict[index]

            print(f"Feature file: {feature_url}")
            print(f"Label file: {label_url}")
           
             # Download data from GCS URLs using Google Cloud Storage API
            bucket_name, blob_name = feature_url.replace("gs://", "").split("/", 1)
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            feature_data = blob.download_as_string()

            bucket_name, blob_name = label_url.replace("gs://", "").split("/", 1)
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            label_data = blob.download_as_string()

            # Load and yield feature and label tensors in matched pairs
            feature_tensor = tf.convert_to_tensor(
                np.load(BytesIO(feature_data)),
                dtype=tf.float32,
            )
            label_tensor = tf.convert_to_tensor(
                np.load(BytesIO(label_data)),
                dtype=tf.float32
            )
            
            reshaped_label_tensor = tf.transpose(label_tensor, perm=[2, 0, 1])
            label_tensor = tf.ensure_shape(reshaped_label_tensor, (rainfall_duration, 1000, 1000))

            print("Finished creating feature and label tensors...")
            print(f"Feature tensor shape: {feature_tensor.shape}")
            print(f"Label tensor shape: {reshaped_label_tensor.shape}")
            print("\n")

            yield feature_tensor, reshaped_label_tensor


    def combined_generator(self, sim_name):  
        feature_label_generator = self._generate_feature_label_tensors(sim_name)
        temporal_tensor_generator = self._generate_temporal_tensors(sim_name)
        spatiotemporal_tensor_generator = self._generate_spatiotemporal_tensor([1000, 1000, 1])

        # Combine the generators into a single generator
        for (geo, labels), temp, spatemp in zip(
            feature_label_generator,
            temporal_tensor_generator,
            spatiotemporal_tensor_generator
        ):
            yield {
                "geospatial": geo,
                "temporal": temp,
                "spatiotemporal": spatemp,    
            } , labels

    def get_datasets_from_tensors(self, sim_names, show_progress=True, limit=1):
        """
        Get a generator that yields smaller chunks of datasets for each simulation for training. 
        The examples are generated from multiple simulations.
        """
        storm_durations = {}  # Create an empty dictionary to store storm durations
        for sim_name in sim_names:
            storm_duration = self.rainfall_duration_generator(sim_name)
            storm_durations[sim_name] = storm_duration  # Store the storm duration in the dictionary

        def dataset_generator(sim_name, storm_duration):
            print("Dataset generator: generating training data for sim_name: ", sim_name, "Storm Duration: ", storm_duration)

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

            # Create the dataset for this simulation
            full_dataset = tf.data.Dataset.from_generator(
                lambda: self.combined_generator(sim_name),
                output_signature=output_signature
            )

            # Set the chunk size for the dataset, and batch the dataset
            chunk_size = 1 
            full_dataset = full_dataset.batch(chunk_size)

            yield full_dataset, storm_duration

        return dataset_generator, storm_durations

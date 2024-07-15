import re
from typing import Tuple
from io import BytesIO

from typing import Iterator
import random

import numpy as np
import tensorflow as tf
from google.cloud import firestore, storage

from usl_models.flood_ml.metastore import FirestoreDataHandler
from usl_models.flood_ml.settings import Settings
from usl_models.flood_ml.featurelabelchunks import GenerateFeatureLabelChunks
from usl_models.flood_ml import constants
from usl_models.flood_ml.model import FloodModel

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
        settings: Settings = None,
        firestore_client: firestore.Client = None,
        storage_client: storage.Client = None,
        metastore: FirestoreDataHandler = None,
        featurelabelchunksgenerator: GenerateFeatureLabelChunks = None,
    ):
        # print("Initializing IncrementalTrainDataGenerator...")

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
        temporal_chunks, sim_dict = (
            self.featurelabelchunksgenerator.get_temporal_chunks(sim_name)
        )

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

        feature_chunks, _ = self.featurelabelchunksgenerator.get_feature_chunks(
            sim_name
        )
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
        random.shuffle(common_indices)  # Shuffle directly within the function

        if len(common_indices) != len(feature_chunks) or len(common_indices) != len(
            label_chunks
        ):
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
                np.load(BytesIO(label_data)), dtype=tf.float32
            )

            reshaped_label_tensor = tf.transpose(label_tensor, perm=[2, 0, 1])
            label_tensor = tf.ensure_shape(
                reshaped_label_tensor, (rainfall_duration, 1000, 1000)
            )

            print("Finished creating feature and label tensors...")
            print(f"Feature tensor shape: {feature_tensor.shape}")
            print(f"Label tensor shape: {reshaped_label_tensor.shape}")
            print("\n")

            yield feature_tensor, reshaped_label_tensor

    @staticmethod
    def extract_spatiotemporal(t: int, n: int, labels: tf.Tensor) -> tf.Tensor:
        """Extracts spatiotemporal tensor from labeled data.

        t is the current timestep.
        n is the window size.
        The result is the slice labels[t-n:t] with zero padding so the
        output tensor is always of shape (n, H, W, 1).
        """
        (_, H, W, *_) = labels.shape
        zeros = tf.zeros(shape=(max(n - t, 0), H, W), dtype=tf.float32)
        # print("labels:", labels.shape)
        data = labels[max(t - n, 0) : t]
        # print(f"labels[max(t - n, 0):{t}]:", data.shape)
        return tf.expand_dims(tf.concat([zeros, data], axis=0), axis=-1)

    @staticmethod
    def extract_temporal(t: int, n: int, temporal: tf.Tensor) -> tf.Tensor:
        (_, D) = temporal.shape
        zeros = tf.zeros(shape=(max(n - t, 0), D))
        data = temporal[max(t - n, 0) : t]
        return tf.concat([zeros, data], axis=0)

    @classmethod
    def generate_windows(
        cls, input: FloodModel.Input, labels: tf.Tensor, n: int = constants.N_FLOOD_MAPS
    ) -> Iterator[Tuple[FloodModel.Input, tf.Tensor]]:
        """Generate inputs for a sliding time window of length n timesteps."""
        (T_max, H, W, *_) = labels.shape
        for t in range(T_max):
            window_input = FloodModel.Input(
                geospatial=input["geospatial"],
                temporal=cls.extract_temporal(t, n, input["temporal"]),
                spatiotemporal=cls.extract_spatiotemporal(t, n, labels),
            )
            # print("input.geospatial:", input["geospatial"].shape)
            # print("input.temporal:", input["temporal"].shape)
            # print("input.spatiotemporal:", input["spatiotemporal"].shape)
            # print("print(tf.math.reduce_max(labels))", tf.math.reduce_max(labels))
            # print("print(tf.math.reduce_max(labels[t]))", tf.math.reduce_max(labels[t]))
            yield window_input, labels[t]

    def load_simulation_chunks(
        self, sim_name: str, max_chunks: int = 0
    ) -> Iterator[Tuple[FloodModel.Input, tf.Tensor]]:
        """Loads a single simulation's spatial chunks."""

        feature_label_gen = self._generate_feature_label_tensors(sim_name)
        temporal_gen = self._generate_temporal_tensors(sim_name)

        chunks = 0
        # For each spatial chunk in the simulation
        for (geospatial, labels), temporal in zip(feature_label_gen, temporal_gen):
            chunks += 1
            if max_chunks > 0 and chunks > max_chunks:
                break

            input = FloodModel.Input(
                temporal=temporal,
                geospatial=geospatial,
                spatiotemporal=tf.zeros(
                    shape=(
                        constants.N_FLOOD_MAPS,
                        constants.MAP_HEIGHT,
                        constants.MAP_WIDTH,
                        1,
                    )
                ),
            )
            yield input, labels

    def load_dataset(
        self, sim_names: list[str], batch_size: int = 1, max_chunks: int = 0
    ) -> tf.data.Dataset:
        """
        Get a generator that yields smaller chunks of datasets for each simulation for training.
        The examples are generated from multiple simulations.
        """

        def generator():
            for sim_name in sim_names:
                for input, labels in self.load_simulation_chunks(sim_name, max_chunks):
                    yield (input, labels)

        # Create the dataset for this simulation
        dataset = tf.data.Dataset.from_generator(
            generator=generator,
            output_signature=(
                dict(
                    geospatial=tf.TensorSpec(
                        shape=(
                            constants.MAP_HEIGHT,
                            constants.MAP_WIDTH,
                            constants.GEO_FEATURES,
                        ),
                        dtype=tf.float32,
                    ),
                    temporal=tf.TensorSpec(
                        shape=(constants.MAX_RAINFALL_DURATION, constants.M_RAINFALL),
                        dtype=tf.float32,
                    ),
                    spatiotemporal=tf.TensorSpec(
                        shape=(
                            constants.N_FLOOD_MAPS,
                            constants.MAP_HEIGHT,
                            constants.MAP_WIDTH,
                            1,
                        ),
                        dtype=tf.float32,
                    ),
                ),
                tf.TensorSpec(
                    shape=(None, constants.MAP_HEIGHT, constants.MAP_WIDTH),
                    dtype=tf.float32,
                ),
            ),
        )
        if batch_size:
            dataset = dataset.batch(batch_size)
        return dataset

    def load_dataset_windowed(
        self, sim_names: list[str], batch_size: int = 1, max_chunks: int = 0
    ) -> tf.data.Dataset:
        """
        Get a generator that yields smaller chunks of datasets for each simulation for training.
        The examples are generated from multiple simulations.
        Windowed for training on next-map prediction.
        """

        def generator():
            """Windowed generator for teacher-forcing training."""
            for sim_name in sim_names:
                for input, labels in self.load_simulation_chunks(
                    sim_name, max_chunks=max_chunks
                ):
                    for window_input, window_label in self.generate_windows(
                        input, labels
                    ):
                        yield (window_input, window_label)

        # Create the dataset for this simulation
        dataset = tf.data.Dataset.from_generator(
            generator=generator,
            output_signature=(
                dict(
                    geospatial=tf.TensorSpec(
                        shape=(
                            constants.MAP_HEIGHT,
                            constants.MAP_WIDTH,
                            constants.GEO_FEATURES,
                        ),
                        dtype=tf.float32,
                    ),
                    temporal=tf.TensorSpec(
                        shape=(constants.N_FLOOD_MAPS, constants.M_RAINFALL),
                        dtype=tf.float32,
                    ),
                    spatiotemporal=tf.TensorSpec(
                        shape=(
                            constants.N_FLOOD_MAPS,
                            constants.MAP_HEIGHT,
                            constants.MAP_WIDTH,
                            1,
                        ),
                        dtype=tf.float32,
                    ),
                ),
                tf.TensorSpec(
                    shape=(constants.MAP_HEIGHT, constants.MAP_WIDTH), dtype=tf.float32
                ),
            ),
        )
        if batch_size:
            dataset = dataset.batch(batch_size)
        return dataset

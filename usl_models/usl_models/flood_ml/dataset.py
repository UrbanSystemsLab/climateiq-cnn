from urllib.parse import urlparse
from typing import Optional, List, Dict
import random

import numpy as np
import tensorflow as tf
from google.cloud import firestore, storage
from dynaconf import Dynaconf

from usl_models.flood_ml.data_utils import FloodModelData


"""
    This class is used to generate the training data for the flood model.
    Input:
        study_areas: List of study areas
        batch_size: Batch size
        config_file: Path to the configuration file containing the parameters for
        the dataset.
    Output:
        A generator that yields a batch of data.

"""


class IncrementalTrainDataGenerator:
    def __init__(
        self,
        study_areas: List[str],
        batch_size: int,
        config_file: str = "config/dataset_config.json",
        firestore_client: firestore.Client = None,
        storage_client: storage.Client = None,
    ):
        self.config = Dynaconf(
            settings_files=[config_file], environments=True, load_dotenv=True
        )
        self.firestore_client = firestore_client or firestore.Client()
        self.storage_client = storage_client or storage.Client()
        self.firestore_collection = self.config.get("firestore_collection")
        self.study_areas = study_areas
        self.batch_size = batch_size
        self._chunks_iterator = None

    def _create_storage_client(self) -> storage.Client:
        """Create a storage client for querying the GCS bucket."""
        storage_client = storage.Client()
        return storage_client

    def _create_firestore_client(self) -> firestore.Client:
        """Create a Firestore client for querying the Firestore database."""
        firestore_client = firestore.Client()
        return firestore_client

    def _get_chunks_for_study_areas(self) -> List[str]:
        """Returns the chunks for the study areas specified."""
        chunks = []
        for study_area in self.study_areas:
            query = self.firestore_client.collection(self.firestore_collection).where(
                "study_area", "==", study_area
            )
            for doc in query.stream():
                gcs_urls = doc.to_dict().get("chunks", [])
                for gcs_url in gcs_urls:
                    try:
                        chunk = self._get_numpy_tensor_from_gcs(
                            gcs_url
                        )  # Call the GCS loading function
                        chunks.append(chunk)
                    except Exception as e:
                        # Add error handling for failed loads (e.g., logging, skipping, etc.)
                        print(f"Error loading chunk from {gcs_url}: {e}")
        return chunks

    def _get_next_chunk(self) -> Optional[str]:
        """Returns the next chunk of data for training."""
        if self._chunks_iterator is None:
            chunks = self._get_chunks_for_study_areas()
            self._chunks_iterator = iter(chunks)

        try:
            return next(self._chunks_iterator)
        except StopIteration:
            return None

    def _get_numpy_tensor_from_gcs(self, gcs_url: str) -> Dict[str, np.ndarray]:
        parsed_url = urlparse(gcs_url)
        bucket_name = parsed_url.netloc
        blob_name = parsed_url.path.lstrip("/")

        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Download the blob content as bytes
        tensor_bytes = blob.download_as_bytes()

        # Load the numpy array from the bytes
        tensor_dict = np.load(tensor_bytes, allow_pickle=True)

        return {key: tensor_dict[key] for key in tensor_dict}

    def _convert_to_train_data(self, numpy_arrays: List[dict]) -> List[FloodModelData]:
        """Convert the dictionary of numpy tensors to a list of
        FloodModelData objects."""
        train_data = [
            {
                key: np.stack([d[key] for d in numpy_arrays], axis=0)
                for key in numpy_arrays[0]
            }
            for _ in numpy_arrays
        ]

        return [
            FloodModelData(
                storm_duration=tf.convert_to_tensor(td["storm_duration"]),
                geospatial=tf.convert_to_tensor(td["geospatial"]),
                temporal=tf.convert_to_tensor(td["temporal"]),
                spatiotemporal=tf.convert_to_tensor(td.get("spatiotemporal", None)),
                labels=tf.convert_to_tensor(td.get("labels", None)),
            )
            for td in train_data
        ]

    def get_next_batch(self) -> Optional[List[FloodModelData]]:
        """Returns the next batch of data for training."""
        numpy_arrays = []

        for _ in range(self.batch_size):
            gcs_url = self._get_next_chunk()
            if not gcs_url:
                break  # End of data
            try:
                numpy_tensor = self._get_numpy_tensor_from_gcs(gcs_url)
                numpy_arrays.append(numpy_tensor)
            except Exception as e:
                print(f"Error fetching data: {e}")  # Log or handle error

        if not numpy_arrays:
            return None

        # Shuffle within the batch
        random.shuffle(numpy_arrays)

        train_data = self._convert_to_train_data(numpy_arrays)

        return train_data

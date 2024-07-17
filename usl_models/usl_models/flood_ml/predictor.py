import json

import numpy as np
import tensorflow as tf

from google.cloud.aiplatform.utils import prediction_utils
from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud import storage
from usl_models.flood_ml.model import FloodModel, constants


class FloodModelPredictor(Predictor):
    def __init__(self):
        """Initializes the FloodModelPredictor."""
        self._model = None

    def load(self, artifacts_uri: str) -> None:
        """Loads the model artifact.

        Args:
            artifacts_uri (str):
                Required. The value of the environment variable AIP_STORAGE_URI.

        Raises:
            ValueError: If there's no required model files provided in the artifacts
                uri.
        """
        prediction_utils.download_model_artifacts(artifacts_uri)
        # Load the saved model
        loaded_model = tf.saved_model.load('model')

        self._model = loaded_model

        # Get the serving_default signature
        serving_signature = self._model.signatures['serving_default']

        # Print model input signature
        print("Model input signature:", serving_signature.structured_input_signature)

        # Print model output signature
        print("\nModel output signature:")
        for output_name, output_tensor in serving_signature.structured_outputs.items():
            print(f"  {output_name}: {output_tensor.shape}")


    def preprocess(self, input_data) -> dict:
        """Loads and preprocesses data from either a GCS URL or a dictionary of instances.

        Args:
            input_data: Either a GCS URL string or a dictionary containing an "instances" key with a list of instances.

        Returns:
            A dictionary where keys are the field names (e.g., 'geospatial', 'temporal', 'spatiotemporal')
            and values are NumPy arrays with a batch dimension added.
        """
        data = {}

        if isinstance(input_data, str) and input_data.startswith("gs://"):
            # Input is a GCS URL
            storage_client = storage.Client()

            # Parse the GCS URL
            bucket_name = input_data[5:].split('/')[0]
            blob_name = '/'.join(input_data[5:].split('/')[1:])

            # Get the bucket and blob
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            # Download the blob content as a string
            blob_string = blob.download_as_string().decode('utf-8')

            # Iterate over lines in the blob string
            for line_num, line in enumerate(blob_string.splitlines()):
                item = json.loads(line)

                # Create NumPy arrays and add batch dimension directly
                for key in ['geospatial', 'temporal', 'spatiotemporal']:
                    arr = np.array(item[key], dtype=np.float32)
                    arr = np.expand_dims(arr, axis=0)

                    if key in data:
                        data[key] = np.concatenate([data[key], arr], axis=0)
                    else:
                        data[key] = arr

        elif isinstance(input_data, dict) and "instances" in input_data:
            # Input is a dictionary of instances
            instances = input_data["instances"]

            # Batch the instances
            for key in ['geospatial', 'temporal', 'spatiotemporal']:
                data[key] = np.stack([np.array(instance[key], dtype=np.float32) for instance in instances], axis=0)

        else:
            raise ValueError("Invalid input_data format. Expected a GCS URL or a dictionary with 'instances'.")

        # Print shapes for debugging
        print("Loaded data shapes:")
        for key, value in data.items():
            print(f"{key}: {value.shape}")

        return data

    @staticmethod
    def _get_temporal_window(temporal: tf.Tensor, t: int, n: int) -> tf.Tensor:
        """Returns a zero-padded n-sized window at timestep t."""

        B = tf.shape(temporal)[0]
        M = tf.shape(temporal)[2]
        
        # Use tf.maximum instead of max
        pad_size = tf.maximum(n - t, 0)
        start = tf.maximum(t - n, 0)
        
        return tf.concat(
            [tf.zeros(shape=(B, pad_size, M), dtype=temporal.dtype),
            temporal[:, start:t, :]],
            axis=1
        )

    def predict(self, data: dict, n=1):
        """Runs the entire autoregressive model.

            Args:
                data: A dictionary of input tensors.
                    While `call` expects only input data for a single context window,
                    `call_n` requires the full temporal tensor.
                n: Number of autoregressive iterations to run.

            Returns:
                A tensor of all the flood predictions: [B, n, H, W].
            """
        if self._model is None:
            raise ValueError("Model not loaded. Call load() first.")

        try:
            spatiotemporal = data["spatiotemporal"]
            geospatial = data["geospatial"]
            temporal = data["temporal"]

            B = spatiotemporal.shape[0]
            C = 1  # Channel dimension for spatiotemporal tensor
            F = constants.GEO_FEATURES
            N, M = constants.N_FLOOD_MAPS, constants.M_RAINFALL
            T_MAX = constants.MAX_RAINFALL_DURATION
            H, W = constants.MAP_HEIGHT, constants.MAP_WIDTH

            tf.ensure_shape(spatiotemporal, (B, N, H, W, C))
            tf.ensure_shape(geospatial, (B, H, W, F))
            tf.ensure_shape(temporal, (B, T_MAX, M))

            # This array stores the n predictions.
            predictions = tf.TensorArray(tf.float32, size=n)

            # We use 1-indexing for simplicity. Time step t represents the t-th flood
            # prediction.
            for t in range(1, n + 1):
                input = FloodModel.Input(
                    geospatial=geospatial,
                    temporal=self._get_temporal_window(temporal, t, N),
                    spatiotemporal=spatiotemporal,
                )
                 # Get the prediction function from loaded model
                predict_fn = self._model.signatures["serving_default"]

                # Make a prediction
                prediction_dict = predict_fn(**input)
                prediction = prediction_dict['output_1']

                predictions = predictions.write(t - 1, prediction)
                

                # Append new predictions along time axis, drop the first.
                spatiotemporal = tf.concat(
                    [spatiotemporal, tf.expand_dims(prediction, axis=1)], axis=1
                )[:, 1:]

            predictions = tf.stack(tf.unstack(predictions.stack()),axis=1)
            print("Prediction shape (before sequeeze): ", predictions.shape)
            # Drop channels dimension.
            return tf.squeeze(predictions, axis=-1)

        except Exception as e:
            raise RuntimeError(f"Error during prediction: {e}")

    def postprocess(self, prediction_results: np.ndarray) -> dict:
        """Converts numpy array to a dict.
        Args:
            prediction_results (np.ndarray):
                Required. The prediction results.
        Returns:
            The postprocessed prediction results.
        """
        return {tf.math.reduce_max(prediction_results, axis=1)}
    
    def health(self):
     return 'Healthy', 200, {'Content-Type': 'text/plain'}


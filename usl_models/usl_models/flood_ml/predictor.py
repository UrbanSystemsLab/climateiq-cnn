import logging

import numpy as np
import tensorflow as tf

from google.cloud.aiplatform.constants import prediction
from google.cloud.aiplatform.utils import prediction_utils
from google.cloud.aiplatform.prediction.predictor import Predictor
from usl_models.flood_ml.model import FloodModel, constants


class FloodModelPredictor(Predictor):
    """Default Predictor implementation for Sklearn models."""

    def __init__(self):
        """Initializes the FloodModelPredictor."""
        self._model = None
        logging.basicConfig(level=logging.INFO)

    def load(self, artifacts_uri: str) -> None:
        """Loads the model artifact.

        Args:
            artifacts_uri (str):
                Required. The value of the environment variable AIP_STORAGE_URI.

        Raises:
            ValueError: If there's no required model files provided in the artifacts
                uri.
        """
        #prediction_utils.download_model_artifacts(artifacts_uri)
         # Load the saved model
        #loaded_model = tf.saved_model.load('model')
        loaded_model = tf.saved_model.load('flood_model_tf213_local_1')

        # model_params = FloodModelParams()  # You may need to adjust this
        # self._model = FloodModel(model_params)
        self._model = loaded_model

        # Get the serving_default signature
        serving_signature = self._model.signatures['serving_default']

        # Print model input signature
        print("Model input signature:", serving_signature.structured_input_signature)

        # Print model output signature
        print("\nModel output signature:")
        for output_name, output_tensor in serving_signature.structured_outputs.items():
            print(f"  {output_name}: {output_tensor.shape}")

    def preprocess(self, prediction_input: dict) -> np.ndarray:
        """Converts the request body to a numpy array before prediction.
        Args:
            prediction_input (dict):
                Required. The prediction input that needs to be preprocessed.
        Returns:
            The preprocessed prediction input.
        """
        instances = prediction_input["instances"]
        return instances
    
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

    def predict(self, full_input, n=1):
        """Runs the entire autoregressive model.

            Args:
                full_input: A dictionary of input tensors.
                    While `call` expects only input data for a single context window,
                    `call_n` requires the full temporal tensor.
                n: Number of autoregressive iterations to run.

            Returns:
                A tensor of all the flood predictions: [B, n, H, W].
            """
        if self._model is None:
            raise ValueError("Model not loaded. Call load() first.")

        try:
            spatiotemporal = full_input["spatiotemporal"]
            geospatial = full_input["geospatial"]
            temporal = full_input["temporal"]

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
            print("Prediction size: ", predictions.size())

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
                print("Prediction shape: ", prediction.shape)

                # Append new predictions along time axis, drop the first.
                spatiotemporal = tf.concat(
                    [spatiotemporal, tf.expand_dims(prediction, axis=1)], axis=1
                )[:, 1:]

            predictions = tf.stack(tf.unstack(predictions.stack(), num=n ),axis=1)
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
        return {tf.math.reduce_max(prediction_results, axis=0)}

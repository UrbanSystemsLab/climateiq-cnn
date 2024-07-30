"""Utilities for running batch predictions on files stored in GCS.

See `BatchPredictor` for a full description of the approach.
"""

import dataclasses
import json
import logging
import urllib.parse

from google.cloud import firestore  # type:ignore[attr-defined]
from google.cloud import storage  # type:ignore[attr-defined]
import numpy as np
import tensorflow as tf

from usl_models.flood_ml import model
from usl_models.flood_ml import prediction_dataset


@dataclasses.dataclass
class BatchPredictor:
    """Runs batch predictions on files stored in GCS.

    Ouputs data based on the specified `output_bucket`, `model_id`,
    and `run_id`.

    To use this class to run batch prediction on "NYC" study area:
    ```py
    predictor = BatchPredictor(
        model=FloodModel.from_checkpoint("gs://my/model"),
        ...)
    predictor.predict_and_save_scenario(
        study_area_id="NYC",
        config_id="config_v1/Rainfall_Data_3.txt",
        scenario_id="flood_depth_1y",
    )
    ```
    """

    db: firestore.Client
    client: storage.Client
    output_bucket: str
    model_id: str
    model: model.FloodModel
    run_id: str
    batch_size: int = 2  # Ideally 2 X number of GPUs when using T4.

    def predict_and_save_scenario(
        self,
        study_area_id: str,
        config_id: str,
        scenario_id: str,
    ):
        """Runs predictions for a rainfall scenario and saves outputs to GCS.

        This method runs batch prediction over an entire study area with the given
        rainfall config chunk-by-chunk. Each chunk is stored in an intermediate .npy
        file. After all chuunks are processed, theya re bundled into a single .jsonl
        file as expected by the frontend and the .npy files are deleted.

        Args:
            study_area_id: Study area ID.
            config_id: Rainfall config ID.
            scenario_id: Scenario ID.
        """
        parsed_config_id = urllib.parse.quote_plus(config_id)
        config_dict = (
            self.db.collection("city_cat_rainfall_configs")
            .document(parsed_config_id)
            .get()
            .to_dict()
        )
        rainfall_duration = config_dict["rainfall_duration"]
        dataset = prediction_dataset.load_prediction_dataset(
            study_area=study_area_id,
            city_cat_config=parsed_config_id,
            batch_size=self.batch_size,
            storage_client=self.client,
        )
        chunk_ids = []
        for results in self.model.batch_predict_n(dataset, n=rainfall_duration):
            for result in results:
                chunk_id = result["chunk_id"]

                # When the dataset is distributed, strings are converted to
                # tensors, so we must decode them back.
                if isinstance(chunk_id, tf.Tensor):
                    chunk_id = chunk_id.numpy().decode("utf-8")

                chunk_ids.append(chunk_id)
                self.save_prediction_npy(
                    study_area_id=study_area_id,
                    config_id=config_id,
                    chunk_id=chunk_id,
                    prediction=result["prediction"].numpy(),
                )

        self.set_model_run_prediction_metadata(
            study_area_id=study_area_id,
            scenario_id=scenario_id,
            chunk_ids=chunk_ids,
        )

        self.bundle_predictions_to_jsonl(
            study_area_id=study_area_id,
            config_id=config_id,
            chunk_ids=chunk_ids,
            scenario_id=scenario_id,
        )

    def get_prediction_npy_path(
        self, study_area_id: str, config_id: str, chunk_id: str
    ) -> str:
        """Returns the GCS filepath for a prediction npy file."""
        return (
            f"{self.run_id}/flood/{self.model_id}/{study_area_id}"
            + f"/{config_id}/{chunk_id}"
        )

    def get_prediction_jsonl_path(self, study_area_id: str, scenario_id: str) -> str:
        """Returns the GCS filepath for a prediction jsonl file."""
        return (
            f"{self.run_id}/flood/{self.model_id}/{study_area_id}"
            + f"/{scenario_id}/prediction.results-1-of-1"
        )

    def save_prediction_npy(
        self, study_area_id: str, config_id: str, chunk_id: str, prediction: np.ndarray
    ) -> None:
        """Saves a prediction npy file to GCS."""
        path = self.get_prediction_npy_path(study_area_id, config_id, chunk_id)
        logging.info(f"Saving chunk to {path} with shape {prediction.shape}")
        blob = self.client.get_bucket(self.output_bucket).blob(path)
        with blob.open("wb") as fd:
            np.save(fd, prediction)

    def load_prediction_npy(
        self, study_area_id: str, config_id: str, chunk_id: str, delete: bool = False
    ) -> np.ndarray:
        """Loads a prediction npy file from GCS."""
        blob = self.client.get_bucket(self.output_bucket).blob(
            self.get_prediction_npy_path(study_area_id, config_id, chunk_id)
        )
        with blob.open("rb") as fd:
            prediction = np.load(fd)

        if delete:
            blob.delete()

        return prediction

    def set_model_run_metadata(self, scenario_ids: list[str]):
        """Sets model run metadata in firestore."""
        model_ref = self.db.collection("models").document(self.model_id)
        run_ref = model_ref.collection("runs").document(self.run_id)
        run_ref.set({"scenario_ids": scenario_ids})

    def set_model_run_prediction_metadata(
        self, study_area_id: str, scenario_id: str, chunk_ids: list[str]
    ):
        """Sets the model run prediction metadata in firestore."""
        model_ref = self.db.collection("models").document(self.model_id)
        run_ref = model_ref.collection("runs").document(self.run_id)
        prediction_id = f"Prediction-{study_area_id}-{scenario_id}"
        prediction_ref = run_ref.collection("predictions").document(prediction_id)

        chunks = self.get_chunk_metadata(study_area_id, scenario_id, chunk_ids)

        prediction_ref.set(
            {
                "study_area_id": study_area_id,
                "scenario_configuration_id": scenario_id,
                "chunks": chunks,
            }
        )

    def get_chunk_metadata(
        self, study_area_id: str, scenario_id: str, chunk_ids: list[str]
    ) -> list[dict]:
        """Returns a list of chunk metadata."""
        return [
            {
                "id": chunk_id,
                "path": self.get_prediction_jsonl_path(study_area_id, scenario_id),
            }
            for chunk_id in chunk_ids
        ]

    @staticmethod
    def prediction_to_json(prediction: np.ndarray, chunk_id: str) -> dict:
        """Converts a prediction array to JSON."""
        return {
            "instance": {
                "values": [1],
                "key": chunk_id,
            },
            "prediction": prediction.tolist(),
        }

    def bundle_predictions_to_jsonl(
        self,
        study_area_id: str,
        config_id: str,
        chunk_ids: list[str],
        scenario_id: str,
    ):
        """Convert npy files to a single jsonl file."""
        jsonl_path = self.get_prediction_jsonl_path(study_area_id, scenario_id)
        with self.client.bucket(self.output_bucket).blob(jsonl_path).open("w") as fd:
            for chunk_id in chunk_ids:
                path = self.get_prediction_npy_path(study_area_id, config_id, chunk_id)
                logging.info(f"Bundling {path}...")
                prediction = self.load_prediction_npy(
                    study_area_id, config_id, chunk_id, delete=True
                )
                json_data = self.prediction_to_json(prediction, chunk_id)
                json.dump(json_data, fd)
                fd.write("\n")

import sys, os
from pathlib import Path

from google.cloud.aiplatform.prediction import LocalModel
from usl_models.flood_ml.predictor import FloodModelPredictor 
from usl_models.flood_ml.dataset import load_dataset_windowed, load_dataset
import json
import numpy as np
from google.cloud import firestore
from google.cloud import storage
import tensorflow as tf

def create_cotainer():
    REGION="us-central1"
    PROJECT_ID="climateiq-test"
    PATH_TO_REQUIREMENTS_TXT="./usl_models/requirements.txt"
    PREDICTOR_CLASS=FloodModelPredictor
    PATH_TO_THE_SOURCE_DIR="./usl_models/"
    REPOSITORY="custom-flood-ml-prediction-container"
    IMAGE="flood-ml-cpr"

    print(PATH_TO_THE_SOURCE_DIR)

    local_model = LocalModel.build_cpr_model(
        PATH_TO_THE_SOURCE_DIR,
        f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{IMAGE}",
        predictor=PREDICTOR_CLASS,
        requirements_path=PATH_TO_REQUIREMENTS_TXT
    )

def tensor_to_json_serializable(tensor):
    if isinstance(tensor, tf.Tensor):
        return tensor.numpy().tolist()
    elif isinstance(tensor, np.ndarray):
        return tensor.tolist()
    else:
        return tensor

def create_jsonl_file():
    sim_names = ["Manhattan-config_v1/Rainfall_Data_1.txt", "Manhattan-config_v1/Rainfall_Data_2.txt"]
    batch_size = 0 # On A100, use batch_size = 4
    # dataset = load_dataset_windowed(sim_names, batch_size=batch_size, max_chunks=8,
    #                                 firestore_client=firestore.Client(project='climateiq-test'),
    #                                 storage_client=storage.Client(project='climateiq-test'))
    dataset = load_dataset(sim_names, batch_size=batch_size, max_chunks=8,
                                     firestore_client=firestore.Client(project='climateiq-test'),
                                     storage_client=storage.Client(project='climateiq-test'))

    inputs, labels = next(iter(dataset))
    print("Input shapes:")
    for key, value in inputs.items():
        print(f"{key}: {value.shape}")

    print("\nLabel shape:", labels.shape)

    outfile = 'batch_pred_6.jsonl'

    # Convert inputs to JSON serializable format
    json_serializable_inputs = {
        key: tensor_to_json_serializable(value)
        for key, value in inputs.items()
    }

    # Write to JSONL file
    with open(outfile, 'w') as f:
        json.dump(json_serializable_inputs, f)

    print("JSONL file created successfully.")


def load_jsonl_to_numpy(file_path):
    data = {}  # Initialize an empty dictionary

    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file):  # Enumerate to keep track of line number (batch index)
            item = json.loads(line)
            
            # Create NumPy arrays and add batch dimension directly
            for key in ['geospatial', 'temporal', 'spatiotemporal']:
                arr = np.array(item[key], dtype=np.float32)
                arr = np.expand_dims(arr, axis=0)  # Add batch dimension 
                
                if key in data:
                    data[key] = np.concatenate([data[key], arr], axis=0) 
                else:
                    data[key] = arr

    # Print shapes for debugging
    print("Loaded data shapes:")
    for key, value in data.items():
        print(f"{key}: {value.shape}")

    return data


def main():
    from usl_models.flood_ml.predictor import FloodModelPredictor 

    n = 9

    predictor = FloodModelPredictor()
    #create_jsonl_file()

    #predictor.load("gs://climateiq-vertexai/aiplatform-custom-training-2024-07-15-13:32:01.898/")
    predictor.load('./flood_model_tf213_local_1/')

    # Usage
    file_path = 'batch_pred_6.jsonl'
    instances = load_jsonl_to_numpy(file_path)

    predictor.predict(instances, n)

if __name__ == "__main__":
    main()
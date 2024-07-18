import sys, os
from pathlib import Path

from google.cloud import aiplatform
from google.cloud.aiplatform.prediction import LocalModel
from usl_models.flood_ml.predictor import FloodModelPredictor 
from usl_models.flood_ml.dataset import load_dataset
import json
import numpy as np
from google.cloud import firestore
from google.cloud import storage
import tensorflow as tf

def tensor_to_json_serializable(tensor):
    if isinstance(tensor, tf.Tensor):
        return tensor.numpy().tolist()
    elif isinstance(tensor, np.ndarray):
        return tensor.tolist()
    else:
        return tensor

def create_jsonl_file(sim_names: list):
    batch_size = 0 
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
            for key in ['geospatial', 'temporal', 'spatiotemporal', 'n']:
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
    predictor = FloodModelPredictor()
    sim_names = ["Manhattan-config_v1/Rainfall_Data_1.txt"]
    use_local = True
    # model_gcs_url = "gs://climateiq-vertexai/aiplatform-custom-training-2024-07-16-17:40:15.640/"
    # create prediction container
    #create_cotainer()
    # # load model , model can be in GCS or local, present in "model" directory
    # predictor.load(model_gcs_url)
    #create_jsonl_file(sim_names=sim_names)

    # # if not use_local:
    # #     predictor.load(model_gcs_url)
    # # else:
    # #     predictor.load('model/')
    predictor.load('model/')


    # # # load unbatched input data and batch, this is typically done by Vertex
    batch_data_file = "batch_pred_6.jsonl"
    instances_dict = load_jsonl_to_numpy(batch_data_file)
    
    print("Calling prediction..")

    # # call predict

    predictions = predictor.predict_now(instances_dict)
    print("Predicitions successful, shape:", predictions.shape)

if __name__ == "__main__":
    main()
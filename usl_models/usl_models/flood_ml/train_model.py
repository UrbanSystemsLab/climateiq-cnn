import sys, os

 # Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the desired directory (where 'usl_models' is located)
usl_models_dir = os.path.join(current_dir, '../..') # go one directory up

# Append the path to sys.path
sys.path.append(usl_models_dir)
print(sys.path)



import tensorflow as tf

from usl_models.flood_ml.model import FloodModel, FloodModelParams
from usl_models.flood_ml.trainingdataset import IncrementalTrainDataGenerator
from usl_models.flood_ml.featurelabelchunks import GenerateFeatureLabelChunks



"""
THis is a method that is used to train the flood model.
This method is called from Docker container by the Vertex
hyperparameter tuning job.

Args:
    model_dir: The GCS path where the model will be saved.

    The hyperparameters:
        batch_size: The batch size.
        lstm_units: The number of units in the LSTM layer.
        lstm_kernel_size: The kernel size for the LSTM layer.
        lstm_dropout: The dropout rate for the LSTM layer.
        lstm_recurrent_dropout: The recurrent dropout rate for the LSTM layer.
        learning_rate: The learning rate.
        epochs: The number of epochs.

Returns:
    A list of Keras history objects.

"""


def train_model(
    model_dir: str,
    batch_size: int,
    lstm_units: int,
    lstm_kernel_size: int,
    lstm_dropout: float,
    lstm_recurrent_dropout: float,
    epochs: int,
):

    # Create FloodModelParams from hyperparameters
    model_params = FloodModelParams(
        batch_size=batch_size,
        lstm_units=lstm_units,
        lstm_kernel_size=lstm_kernel_size,
        lstm_dropout=lstm_dropout,
        lstm_recurrent_dropout=lstm_recurrent_dropout,
        epochs=epochs,
    )
    model_history = []

    # Instantiate FloodModel class
    model = FloodModel(model_params)
    # model = model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate))

    class_label = GenerateFeatureLabelChunks()
    sim_names = class_label._get_sim_names_from_metastore()
    print(sim_names)
    batch_size = 32

    generator = IncrementalTrainDataGenerator(
        batch_size=batch_size,
    )


    # compare feature and label chunks
    for sim_name in sim_names:
        print(f"Comparing feature and label chunks for {sim_name}")
        label_compare = class_label.compare_study_area_sim_chunks(sim_name)
        if label_compare:
            print("Feature and label chunks are equal for", sim_name)
            print("")
        else:
            print("Feature and label chunks are not equal")
            sim_names.remove(sim_name)

    print("**** Length of sim_names: ", len(sim_names))

    # Get the next batch of data
    if len(sim_names) > 0:
        data = generator.get_next_batch(sim_names, batch_size)
        print("\n")
        print("Dataset generation completed, will hand over to model training..")
        print("\n")
        print("Training model")
        model_history.append(model.train(data))
        print("Training complete")

    else:
        print("No valid sims found for training data generation")

    return model_history


def main():

    print(tf.__version__)

    model_histories = train_model(
        model_dir="gs://usl_models_bucket/flood_model",
        batch_size=32,
        lstm_units=128,
        lstm_kernel_size=3,
        lstm_dropout=0.2,
        lstm_recurrent_dropout=0.2,
        epochs=1,
    )

    for history in model_histories:
        print(history.history)


if __name__ == "__main__":
    main()

import tensorflow as tf

from usl_models.flood_ml.model import FloodModel, FloodModelParams
from usl_models.flood_ml.dataset import IncrementalTrainDataGenerator


def train_model(
    model_dir: str,
    batch_size: int,
    lstm_units: int,
    lstm_kernel_size: int,
    lstm_dropout: float,
    lstm_recurrent_dropout: float,
    learning_rate: float,
    epochs: int,
) -> list[tf.keras.callbacks.History]:

    # Create FloodModelParams from hyperparameters
    model_params = FloodModelParams(
        batch_size=batch_size,
        lstm_units=lstm_units,
        lstm_kernel_size=lstm_kernel_size,
        lstm_dropout=lstm_dropout,
        lstm_recurrent_dropout=lstm_recurrent_dropout,
        epochs=epochs,
        learning_rate=learning_rate
    )
    model_history = []

    # Instantiate data generator
    data_generator = IncrementalTrainDataGenerator()

    # Instantiate FloodModel class
    model = FloodModel(model_params)
    model.train(data_generator.get_next_batch())
    # Train the model
    model_history = model.train(data_generator.get_next_batch())

    return model_history

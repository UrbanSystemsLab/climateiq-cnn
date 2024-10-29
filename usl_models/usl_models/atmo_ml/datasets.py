import tensorflow as tf
import numpy as np
from google.cloud import storage, firestore  # type: ignore
from typing import Iterator, Tuple, List
from usl_models.atmo_ml import constants
from usl_models.atmo_ml import cnn_inputs_outputs
from usl_models.atmo_ml import model

import tensorflow as tf
import numpy as np
from google.cloud import storage, firestore  # type: ignore
from typing import Iterator, Tuple, List
from usl_models.atmo_ml import constants
from usl_models.atmo_ml import cnn_inputs_outputs
from usl_models.atmo_ml import model

# Utility function to split full dataset if needed
def split_dataset(
    dataset: tf.data.Dataset, 
    split_ratios: Tuple[float, float, float]
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Splits a full dataset into train, validation, and test sets."""
    total_count = dataset.cardinality().numpy()
    train_count = int(split_ratios[0] * total_count)
    val_count = int(split_ratios[1] * total_count)

    train_dataset = dataset.take(train_count)
    val_dataset = dataset.skip(train_count).take(val_count)
    test_dataset = dataset.skip(train_count + val_count)

    print(f"Total Count: {total_count}, Train Count: {train_count}, Validation Count: {val_count}, Test Count: {dataset.cardinality().numpy() - train_count - val_count}")
    
    return train_dataset, val_dataset, test_dataset

def load_data(
    sim_names: List[str],
    dataset_split: str,
    batch_size: int = 4,
    max_chunks: int | None = None,
    firestore_client: firestore.Client = None,
    storage_client: storage.Client | None = None,
) -> tf.data.Dataset:
    """Loads a specific dataset split (train, val, or test) for the AtmoML model."""
    firestore_client = firestore_client or firestore.Client()
    storage_client = storage_client or storage.Client()

    def generator():
        """Generator for producing inputs and labels based on dataset split."""
        for sim_name in sim_names:
            print(f"Loading simulation data for: {sim_name}")
            for model_input, labels in _iter_model_inputs(
                firestore_client,
                storage_client,
                sim_name,
                max_chunks,
                dataset_split,
            ):
                yield model_input, labels

    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_signature=( 
            dict(
                spatiotemporal=_spatiotemporal_dataset_signature(),
                spatial=_spatial_dataset_signature(),
                lu_index=_lu_index_signature(),
            ),
            tf.TensorSpec(
                shape=(None, constants.MAP_HEIGHT, constants.MAP_WIDTH),
                dtype=tf.float32,
            ),
        ),
    )
    
    print(f"Dataset created with batch size before batching: {batch_size}")
    
    if batch_size:
        dataset = dataset.batch(batch_size)

    return dataset

def _iter_model_inputs(
    firestore_client: firestore.Client,
    storage_client: storage.Client,
    sim_name: str,
    max_chunks: int | None,
    dataset_split: str,
) -> Iterator[Tuple[model.AtmoModel.Input, tf.Tensor]]:
    """Yields model inputs and labels from Atmo simulation data."""
    spatiotemporal_data = _load_spatiotemporal_data(sim_name, dataset_split, storage_client, firestore_client)
    spatial_data = _load_spatial_data(sim_name, dataset_split, storage_client, firestore_client)
    lu_index_data = _load_lu_index_data(sim_name, dataset_split, storage_client, firestore_client)
    labels = load_labels_data(sim_name, dataset_split, storage_client, firestore_client)

    print(f"Loaded spatiotemporal data for {sim_name}: {spatiotemporal_data.shape}")
    print(f"Loaded spatial data for {sim_name}: {spatial_data.shape}")
    print(f"Loaded LU index data for {sim_name}: {lu_index_data.shape}")
    print(f"Loaded labels for {sim_name}: {labels.shape}")

    spatiotemporal_inputs, label_sequences = cnn_inputs_outputs.divide_into_days(spatiotemporal_data, labels)

    for st_input, label in zip(spatiotemporal_inputs, label_sequences):
        model_input = model.AtmoModel.Input(
            spatiotemporal=st_input,
            spatial=spatial_data,
            lu_index=lu_index_data,
        )
        yield model_input, label

def _load_spatiotemporal_data(
    sim_name: str,
    dataset_split: str,
    batch_size: int = 4,
    max_chunks: int | None = None,
    firestore_client: firestore.Client = None,
    storage_client: storage.Client = None,
) -> tf.Tensor:
    """Load spatiotemporal data with specified parameters."""
    spatiotemporal_data = load_data(sim_name, dataset_split, batch_size, max_chunks, firestore_client, storage_client)
    return tf.convert_to_tensor(spatiotemporal_data, dtype=tf.float32)

def load_labels_data(
    sim_name: str,
    dataset_split: str,
    batch_size: int = 4,
    max_chunks: int | None = None,
    firestore_client: firestore.Client = None,
    storage_client: storage.Client = None,
) -> tf.Tensor:
    """Load labels with specified parameters."""
    labels = load_data(sim_name, dataset_split, batch_size, max_chunks, firestore_client, storage_client)
    return tf.convert_to_tensor(labels, dtype=tf.float32)

def _load_lu_index_data(
    sim_name: str,
    dataset_split: str,
    batch_size: int = 4,
    max_chunks: int | None = None,
    firestore_client: firestore.Client = None,
    storage_client: storage.Client = None,
) -> tf.Tensor:
    """Load land use index with specified parameters."""
    lu_index_data = load_data(sim_name, dataset_split, batch_size, max_chunks, firestore_client, storage_client)
    return tf.convert_to_tensor(lu_index_data, dtype=tf.float32)

def _load_spatial_data(
    sim_name: str,
    dataset_split: str,
    batch_size: int = 4,
    max_chunks: int | None = None,
    firestore_client: firestore.Client = None,
    storage_client: storage.Client = None,
) -> tf.Tensor:
    """Load spatial data with specified parameters."""
    spatial_data = load_data(sim_name, dataset_split, batch_size, max_chunks, firestore_client, storage_client)
    return tf.convert_to_tensor(spatial_data, dtype=tf.float32)

def create_atmo_dataset(
    sim_names: List[str],
    batch_size: int = 4,
    split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    max_chunks: int | None = None,
    firestore_client: firestore.Client = None,
    storage_client: storage.Client = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Create, preprocess, and split an Atmo dataset for training, validation, and testing."""
    dataset = load_data(
        sim_names=sim_names,
        dataset_split='full',
        batch_size=batch_size,
        max_chunks=max_chunks,
        firestore_client=firestore_client,
        storage_client=storage_client,
    )
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, split_ratios=split_ratios)
    
    # Debugging output
    print(f"Train dataset size: {len(list(train_dataset.as_numpy_iterator()))}")
    print(f"Validation dataset size: {len(list(val_dataset.as_numpy_iterator()))}")
    print(f"Test dataset size: {len(list(test_dataset.as_numpy_iterator()))}")

    return train_dataset, val_dataset, test_dataset

def load_prediction_dataset(
    sim_names: List[str],
    batch_size: int = 4,
    max_chunks: int | None = None,
    firestore_client: firestore.Client = None,
    storage_client: storage.Client = None,
) -> tf.data.Dataset:
    """Creates a dataset for predictions without labels."""
    firestore_client = firestore_client or firestore.Client()
    storage_client = storage_client or storage.Client()

    def generator():
        """Generator for producing only inputs (no labels) for predictions."""
        for sim_name in sim_names:
            for model_input in _iter_model_inputs_for_prediction(
                firestore_client,
                storage_client,
                sim_name,
                max_chunks,
            ):
                yield model_input

    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_signature=dict(
            spatiotemporal=_spatiotemporal_dataset_signature(),
            spatial=_spatial_dataset_signature(),
            lu_index=_lu_index_signature(),
        ),
    )
    
    if batch_size:
        dataset = dataset.batch(batch_size)
    
    return dataset

def _iter_model_inputs_for_prediction(
    firestore_client: firestore.Client,
    storage_client: storage.Client,
    sim_name: str,
    max_chunks: int | None,
) -> Iterator[model.AtmoModel.Input]:
    """Yields model inputs for prediction, no labels."""
    spatiotemporal_data = _load_spatiotemporal_data(sim_name, dataset_split='full', storage_client=storage_client, firestore_client=firestore_client)
    spatial_data = _load_spatial_data(sim_name, dataset_split='full', storage_client=storage_client, firestore_client=firestore_client)
    lu_index_data = _load_lu_index_data(sim_name, dataset_split='full', storage_client=storage_client, firestore_client=firestore_client)

    print(f"Loaded prediction spatiotemporal data for {sim_name}: {spatiotemporal_data.shape}")

    spatiotemporal_inputs = cnn_inputs_outputs.divide_into_days(spatiotemporal_data)

    for st_input in spatiotemporal_inputs:
        model_input = model.AtmoModel.Input(
            spatiotemporal=st_input,
            spatial=spatial_data,
            lu_index=lu_index_data,
        )
        yield model_input


def create_prediction_dataset(
    sim_names: List[str],
    batch_size: int = 4,
    max_chunks: int | None = None,
    firestore_client: firestore.Client = None,
    storage_client: storage.Client | None = None,
) -> tf.data.Dataset:
    """Creates and prepares a dataset specifically for predictions."""
    return load_prediction_dataset(
        sim_names=sim_names,
        batch_size=batch_size,
        max_chunks=max_chunks,
        firestore_client=firestore_client,
        storage_client=storage_client,
    )


def make_predictions(model: tf.keras.Model, dataset: tf.data.Dataset) -> np.ndarray:
    """Make predictions using the AtmoML model on the provided dataset."""
    predictions = model.predict(dataset)
    return predictions


def _spatiotemporal_dataset_signature():
    """Defines the spatiotemporal dataset signature."""
    return tf.TensorSpec(
        shape=(constants.TIME_STEPS_PER_DAY, constants.MAP_HEIGHT, constants.MAP_WIDTH, constants.num_spatiotemporal_features),
        dtype=tf.float32,
    )


def _spatial_dataset_signature():
    """Defines the spatial dataset signature."""
    return tf.TensorSpec(
        shape=(constants.MAP_HEIGHT, constants.MAP_WIDTH, constants.num_spatial_features),
        dtype=tf.float32,
    )


def _lu_index_signature():
    """Defines the land-use index dataset signature."""
    return tf.TensorSpec(
        shape=(constants.MAP_HEIGHT * constants.MAP_WIDTH,),
        dtype=tf.int32,
    )
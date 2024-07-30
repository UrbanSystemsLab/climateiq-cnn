"""tf.data.Datasets for training FloodML model on CityCAT data."""

# We just import this function to allow callers to call:
# usl_models.flood_ml.prediction_dataset.load_prediction_dataset
from usl_models.flood_ml.dataset import load_prediction_dataset  # noqa: F401
